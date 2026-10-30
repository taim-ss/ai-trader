"""Conservative, durable DEMO limit-order execution; no automatic POST retries."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, time, timedelta
from math import isfinite
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4
from zoneinfo import ZoneInfo

import httpx

from ai_trader.broker import TRADING212_DEMO_URL, TRADING212_LIVE_URL
from ai_trader.execution_state import CashFlow, RealizedFill, RiskLedger, demo_run_lock
from ai_trader.rebalance import PlannedOrder, RebalancePlan
from ai_trader.risk import RiskLimits, validate_order

TERMINAL = frozenset({"FILLED", "CANCELLED", "REJECTED", "EXPIRED"})
FEE_RESERVE = 1.01
FLOW_TYPES = frozenset({"DEPOSIT", "WITHDRAW", "TRANSFER"})
INCOME_TYPES = frozenset({"FEE", "INTEREST_ON_FREE_CASH", "LENDING_INTEREST"})


class ExecutionBlocked(ValueError):
    """A sanitized, local validation failure safe to return to the caller."""


class DemoBroker(Protocol):
    @property
    def base_url(self) -> str: ...
    def account_summary(self) -> dict[str, Any]: ...
    def positions(self) -> list[dict[str, Any]]: ...
    def instruments(self) -> list[dict[str, Any]]: ...
    def pending_orders(self) -> list[dict[str, Any]]: ...
    def order(self, order_id: str) -> dict[str, Any]: ...
    def order_history(self) -> list[dict[str, Any]]: ...
    def transactions(self) -> list[dict[str, Any]]: ...
    def place_limit_order(self, ticker: str, quantity: float, limit_price: float) -> dict[str, Any]: ...


def _number(value: Any, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ExecutionBlocked("broker and plan numbers must be finite")
    result = float(value)
    if positive and result <= 0:
        raise ExecutionBlocked("expected a positive number")
    return result


def _identifier(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ExecutionBlocked("missing broker identifier")
    return str(value)


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ExecutionBlocked("missing broker timestamp")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ExecutionBlocked("broker timestamps must be timezone aware")
    return result.astimezone(UTC)


def _ticker(order: dict[str, Any]) -> str:
    instrument = order.get("instrument")
    nested = instrument.get("ticker") if isinstance(instrument, dict) else None
    legacy = order.get("ticker")
    if nested is not None and legacy is not None and nested != legacy:
        raise ExecutionBlocked("conflicting broker ticker fields")
    return _identifier(nested if nested is not None else legacy)


def _save(path: Path, journal: dict[str, Any]) -> None:
    """Flush file contents before atomically replacing the journal under the run lock."""
    descriptor, temporary = tempfile.mkstemp(prefix=".orders-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(journal, stream, allow_nan=False, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        marker = path.with_suffix(".initialized")
        if not marker.exists():
            with marker.open("x", encoding="utf-8") as stream:
                stream.write("demo order journal initialized\n")
                stream.flush()
                os.fsync(stream.fileno())
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load(path: Path, account: str) -> dict[str, Any]:
    if not path.exists():
        if path.with_suffix(".initialized").exists():
            raise ExecutionBlocked("order journal is missing after initialization")
        return {"version": 1, "account_id": account, "orders": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or data.get("version") != 1
            or data.get("account_id") != account or not isinstance(data.get("orders"), list)):
        raise ExecutionBlocked("invalid order journal or account mismatch")
    for record in data["orders"]:
        if (not isinstance(record, dict) or record.get("phase") not in
                {"intent", "unknown", "accepted", "terminal", "denied"}):
            raise ExecutionBlocked("invalid order journal record")
        for key in ("intent_id", "cycle_id", "ticker"):
            _identifier(record.get(key))
        _number(record.get("quantity"))
        _number(record.get("limit_price"), positive=True)
        if record["phase"] in {"accepted", "terminal"}:
            _identifier(record.get("order_id"))
        if record["phase"] == "terminal" and record.get("status") not in TERMINAL:
            raise ExecutionBlocked("invalid terminal status in order journal")
    return data


def _risk_events(history: list[dict[str, Any]], transactions: list[dict[str, Any]]) -> tuple[list[RealizedFill], list[CashFlow]]:
    fills: list[RealizedFill] = []
    flows: list[CashFlow] = []
    for row in history:
        order, fill = row["order"], row.get("fill")
        if order.get("side") not in {"BUY", "SELL"}:
            raise ExecutionBlocked("unknown history order side")
        if fill is None or order["side"] == "BUY":
            continue
        impact = fill["walletImpact"]
        if impact["currency"] != "EUR":
            raise ExecutionBlocked("realised P/L currency must be EUR")
        # Trading 212 reports sold quantities as negative numbers (seen on the live account 2026-10-30).
        if _number(fill["quantity"]) == 0:
            raise ExecutionBlocked("fill quantity must be nonzero")
        _number(fill["price"], positive=True)
        fills.append(RealizedFill(_identifier(fill["id"]), _timestamp(fill["filledAt"]),
                                  _number(impact["realisedProfitLoss"])))
    for row in transactions:
        kind = row["type"]
        if kind not in FLOW_TYPES | INCOME_TYPES:
            raise ExecutionBlocked("unknown account transaction type")
        if row["currency"] != "EUR":
            raise ExecutionBlocked("account transaction currency must be EUR")
        amount = _number(row["amount"])
        occurred = _timestamp(row["dateTime"])
        if kind in FLOW_TYPES:
            flows.append(CashFlow(_identifier(row["reference"]), occurred, amount))
    return fills, flows


def _reconcile(broker: DemoBroker, journal: dict[str, Any], history: list[dict[str, Any]],
               result: dict[str, Any]) -> None:
    for record in journal["orders"]:
        if record["phase"] in {"intent", "unknown", "denied"}:
            result["violations"].append("submission requires manual review before any retry")
            continue
        if record["phase"] != "accepted":
            continue
        try:
            observed = broker.order(record["order_id"])
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
            matches = [row for row in history if _identifier(row["order"]["id"]) == record["order_id"]]
            if not matches:
                result["violations"].append("accepted order absent from pending endpoint and history")
                continue
            observed = matches[0]["order"]
            statuses = {row["order"]["status"] for row in matches}
            if len(statuses) != 1:
                raise ExecutionBlocked("inconsistent historical order statuses")
        if (_identifier(observed["id"]) != record["order_id"] or _ticker(observed) != record["ticker"]):
            raise ExecutionBlocked("order reconciliation identity mismatch")
        if observed.get("side") != ("BUY" if record["quantity"] > 0 else "SELL"):
            raise ExecutionBlocked("order reconciliation side mismatch")
        if "quantity" in observed and abs(_number(observed["quantity"])) != abs(record["quantity"]):
            raise ExecutionBlocked("order reconciliation quantity mismatch")
        status = observed["status"]
        result["reconciled"].append({"order_id": record["order_id"], "status": status})
        record["status"] = status
        if status in TERMINAL:
            record["phase"] = "terminal"
        else:
            result["violations"].append("previously accepted demo order is not terminal")


def _validate_plan(plan: RebalancePlan, prices: Mapping[str, float], fx: float) -> None:
    _number(fx, positive=True)
    seen: set[str] = set()
    for order in plan.orders:
        ticker = order.api_ticker
        if not ticker or ticker != ticker.strip() or ticker in seen:
            raise ExecutionBlocked("invalid or duplicate planned ticker")
        seen.add(ticker)
        quantity = _number(order.quantity)
        if quantity == 0 or abs(quantity - round(quantity, 4)) > 1e-12:
            raise ExecutionBlocked("quantity must be nonzero with at most four decimals")
        value = _number(order.value_account, positive=True)
        limit = _number(prices[ticker], positive=True)
        # Caller supplies a bounded limit around its quote; reject obviously stale plans.
        if abs(abs(quantity) * limit * fx / value - 1) > 0.020001:
            raise ExecutionBlocked("limit notional differs from planned EUR value by more than 2%")


def _order_violations(order: PlannedOrder, positions: list[dict[str, Any]], instruments: list[dict[str, Any]],
                      summary: dict[str, Any], price: float, fx: float, limits: RiskLimits) -> list[str]:
    violations: list[str] = []
    matches = [item for item in instruments if item.get("ticker") == order.api_ticker]
    if len(matches) != 1 or matches[0].get("type") not in {"STOCK", "ETF"}:
        return ["planned ticker must uniquely identify a stock or ETF"]
    currency = matches[0].get("currencyCode")
    if currency not in {"EUR", "USD"} or (currency == "EUR" and abs(fx - 1) > 1e-12):
        return ["instrument currency and FX are unsupported or inconsistent"]
    holding: dict[str, Any] | None = None
    seen: set[str] = set()
    for position in positions:
        ticker = _ticker(position)
        if ticker in seen:
            raise ExecutionBlocked("duplicate broker position")
        seen.add(ticker)
        if ticker == order.api_ticker:
            holding = position
    quantity = _number(holding["quantity"]) if holding else 0.0
    available = _number(holding["quantityAvailableForTrading"]) if holding else 0.0
    if quantity < 0 or available < 0 or available > quantity + 1e-10:
        return ["position quantities are invalid for long-only execution"]
    existing_value = 0.0
    if holding:
        impact = holding["walletImpact"]
        if impact["currency"] != "EUR":
            return ["position valuation currency must be EUR"]
        existing_value = _number(impact["currentValue"])
        quote = _number(holding["currentPrice"], positive=True)
        if abs(price / quote - 1) > 0.020001:
            violations.append("limit price differs from refreshed position quote by more than 2%")
        instrument_currency = holding["instrument"].get("currency", holding["instrument"].get("currencyCode"))
        if instrument_currency != currency or existing_value < 0:
            violations.append("position valuation or instrument currency is inconsistent")
    notional = abs(order.quantity) * price * fx
    buffered = notional * FEE_RESERVE
    violations.extend(validate_order(buffered, limits))
    if order.quantity < 0:
        if -order.quantity > available + 1e-10:
            violations.append("sell quantity exceeds shares available for trading")
    else:
        cash = _number(summary["cash"]["availableToTrade"])
        if cash < buffered:
            violations.append("insufficient available EUR cash including 1% reserve")
        equity = _number(summary["totalValue"], positive=True)
        projected = max(existing_value, quantity * price * fx) + buffered
        if projected > equity * limits.max_position_weight + 1e-9:
            violations.append("projected position exceeds portfolio weight cap including reserve")
    return violations


def execute_demo_plan(
    broker: DemoBroker, plan: RebalancePlan, *, limits: RiskLimits, state_dir: Path,
    cycle_id: str, price_limits: Mapping[str, float], fx_native_to_eur: float,
    now: datetime | None = None, execute: bool = False,
    quote_times_utc: Mapping[str, datetime] | None = None,
    approved: tuple[str, float, float] | None = None,
    halt_file: Path | None = None,
) -> dict[str, Any]:
    """Demo environment only (virtual money). See `execute_plan` for the behaviour."""
    return execute_plan(broker, plan, environment="demo", live_blockers=None, limits=limits, state_dir=state_dir,
                        cycle_id=cycle_id, price_limits=price_limits, fx_native_to_eur=fx_native_to_eur, now=now,
                        execute=execute, quote_times_utc=quote_times_utc, approved=approved, halt_file=halt_file)


def execute_live_plan(
    broker: DemoBroker, plan: RebalancePlan, *, live_blockers: Sequence[str], limits: RiskLimits, state_dir: Path,
    cycle_id: str, price_limits: Mapping[str, float], fx_native_to_eur: float,
    now: datetime | None = None, execute: bool = False,
    quote_times_utc: Mapping[str, datetime] | None = None,
    approved: tuple[str, float, float] | None = None,
    halt_file: Path | None = None,
) -> dict[str, Any]:
    """REAL MONEY. Refuses unless the caller proves the live gate is open (empty blocker list), and refuses to
    submit unless the exact order was human-approved. The gate is evaluated by the caller from config
    that only the owner can change; this function re-checks the evidence it is handed."""
    return execute_plan(broker, plan, environment="live", live_blockers=live_blockers, limits=limits,
                        state_dir=state_dir, cycle_id=cycle_id, price_limits=price_limits,
                        fx_native_to_eur=fx_native_to_eur, now=now, execute=execute,
                        quote_times_utc=quote_times_utc, approved=approved, halt_file=halt_file)


def execute_plan(
    broker: DemoBroker, plan: RebalancePlan, *, environment: str, live_blockers: Sequence[str] | None,
    limits: RiskLimits, state_dir: Path,
    cycle_id: str, price_limits: Mapping[str, float], fx_native_to_eur: float,
    now: datetime | None = None, execute: bool = False,
    quote_times_utc: Mapping[str, datetime] | None = None,
    approved: tuple[str, float, float] | None = None,
    halt_file: Path | None = None,
) -> dict[str, Any]:
    """Refresh, reconcile and submit at most one DAY limit order to one exact environment.

    A HALT file (when `halt_file` is given) stops the run at the start and again immediately before the POST;
    only deleting the file resumes trading.

    Dry runs perform reads and persist the risk baseline/reconciliation, never POST.
    A crash after recording intent is deliberately unresolved until a human audits it.
    During US regular hours, quotes must be <=15 minutes old. Outside those hours,
    demo DAY limits may queue using a quote <=4 calendar days old. This is not a
    market-order or live-trading policy. Timestamps are rechecked just before POST.
    """
    result: dict[str, Any] = {"status": "blocked", "violations": [], "submitted": [], "reconciled": []}
    stage = "validation"
    try:
        if halt_file is not None and halt_file.exists():
            raise ExecutionBlocked(f"HALT file present ({halt_file.name}); delete it to resume trading")
        expected_url = {"demo": TRADING212_DEMO_URL, "live": TRADING212_LIVE_URL}.get(environment)
        if expected_url is None or broker.base_url != expected_url:
            raise ExecutionBlocked("executor only accepts the exact demo environment" if environment == "demo"
                                   else "executor environment does not match the broker")
        if environment == "live":
            if live_blockers is None or len(live_blockers) > 0:
                raise ExecutionBlocked("live gate is closed: " + ("; ".join(live_blockers) if live_blockers
                                                                  else "no gate evidence supplied"))
            if execute and approved is None:
                raise ExecutionBlocked("a live order must be an order a human approved")
        if type(execute) is not bool or not isinstance(cycle_id, str) or not cycle_id.strip() or len(cycle_id) > 128:
            raise ExecutionBlocked("invalid execution flag or cycle identifier")
        at = now or datetime.now(UTC)
        if at.tzinfo is None or at.utcoffset() is None:
            raise ExecutionBlocked("run time must be timezone aware")
        _validate_plan(plan, price_limits, fx_native_to_eur)
        result["violations"].extend(plan.violations)
        if result["violations"]:
            return result
        state_dir.mkdir(parents=True, exist_ok=True)
        with demo_run_lock(state_dir / "execution.lock"):
            stage = "account_snapshot"
            summary = broker.account_summary()
            account = _identifier(summary["id"])
            if summary["currency"] != "EUR":
                raise ExecutionBlocked("initial demo executor requires an EUR account")
            equity = _number(summary["totalValue"], positive=True)
            available = _number(summary["cash"]["availableToTrade"])
            reserved = _number(summary["cash"]["reservedForOrders"])
            if available < 0 or reserved < 0:
                raise ExecutionBlocked("account cash values must be nonnegative")
            positions, pending = broker.positions(), broker.pending_orders()
            instruments = broker.instruments()
            stage = "history_reads"
            history, transactions = broker.order_history(), broker.transactions()
            stage = "risk_events"
            fills, flows = _risk_events(history, transactions)
            stage = "risk_ledger"
            ledger = RiskLedger(state_dir / "risk.json", limits)
            result["violations"].extend(ledger.observe(account_id=account, currency="EUR", equity_eur=equity,
                                                       now=at, fills=fills, cashflows=flows))
            path = state_dir / "orders.json"
            stage = "order_reconciliation"
            journal = _load(path, account)
            _reconcile(broker, journal, history, result)
            _save(path, journal)
            if pending or reserved > 0:
                result["violations"].append("pending broker orders or reserved cash block new submissions")
            if result["violations"]:
                return result
            done = {record["ticker"] for record in journal["orders"]
                    if record["cycle_id"] == cycle_id and record["phase"] == "terminal"}
            remaining = sorted((order for order in plan.orders if order.api_ticker not in done),
                               key=lambda order: (order.quantity > 0, order.api_ticker))
            if not remaining:
                result["status"] = "completed"
                return result
            selected = remaining[0]
            stage = "order_validation"
            price = price_limits[selected.api_ticker]
            result["violations"].extend(_order_violations(selected, positions, instruments, summary,
                                                          price, fx_native_to_eur, limits))
            if result["violations"]:
                return result
            if approved is not None and (selected.api_ticker, selected.quantity, price) != approved:
                raise ExecutionBlocked("selected order differs from the order the human approved")
            if not execute:
                result["status"] = "dry-run"
                result["planned"] = {"ticker": selected.api_ticker, "quantity": selected.quantity,
                                     "limit_price": price, "time_validity": "DAY"}
                return result
            if halt_file is not None and halt_file.exists():  # checked again right before the POST
                raise ExecutionBlocked(f"HALT file present ({halt_file.name}); delete it to resume trading")
            send_at = now or datetime.now(UTC)
            ny = send_at.astimezone(ZoneInfo("America/New_York"))
            regular = ny.weekday() < 5 and time(9, 30) <= ny.time().replace(tzinfo=None) < time(16)
            maximum_age = timedelta(minutes=15) if regular else timedelta(days=4)
            if quote_times_utc is None:
                raise ExecutionBlocked("execution requires quote timestamps")
            for key in (selected.api_ticker, "EURUSD=X"):
                stamp = quote_times_utc.get(key)
                if (not isinstance(stamp, datetime) or stamp.tzinfo is None or stamp.utcoffset() is None
                        or stamp > send_at + timedelta(seconds=60) or send_at - stamp > maximum_age):
                    raise ExecutionBlocked("quote failed freshness check immediately before demo submission")
            record = {"intent_id": str(uuid4()), "cycle_id": cycle_id, "ticker": selected.api_ticker,
                      "quantity": selected.quantity, "limit_price": price, "phase": "intent",
                      "created_at": at.isoformat()}
            journal["orders"].append(record)
            _save(path, journal)  # Durable intent must precede the non-idempotent POST.
            try:
                response = broker.place_limit_order(selected.api_ticker, selected.quantity, price)
                record["order_id"] = _identifier(response["id"])
                if _ticker(response) != selected.api_ticker:
                    raise ExecutionBlocked("submission response ticker mismatch")
                record["phase"] = "accepted"
                record["status"] = response["status"]
                result["submitted"].append({"order_id": record["order_id"], "ticker": selected.api_ticker,
                                            "quantity": selected.quantity, "status": record["status"]})
                result["status"] = "submitted"
            except Exception as exc:
                denied = isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in {400, 403}
                record["phase"] = "denied" if denied else "unknown"
                result["violations"].append("broker denied order; manual review required before retry" if denied else
                                              "submission outcome unknown; manual broker reconciliation required")
            _save(path, journal)
    except Exception as exc:
        # Do not return broker error bodies, request headers or credentials.
        result["status"] = "blocked"
        result["failure_stage"] = stage
        result["error_type"] = type(exc).__name__
        safe = str(exc) if isinstance(exc, ExecutionBlocked) else "broker/state operation failed; inspect local state and broker before retrying"
        result["violations"].append(safe)
    return result
