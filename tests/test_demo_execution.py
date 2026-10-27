"""Offline broker fakes exercise durable demo execution and fail-closed paths."""

import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from ai_trader.broker import TRADING212_DEMO_URL, TRADING212_LIVE_URL
from ai_trader.demo_execution import execute_demo_plan
from ai_trader.rebalance import PlannedOrder, RebalancePlan
from ai_trader.risk import RiskLimits

NOW = datetime(2026, 10, 27, 12, tzinfo=UTC)
LIMITS = RiskLimits(0.15, 10, 0.20, 50, False, False, False, 3, 24)


class FakeBroker:
    base_url = TRADING212_DEMO_URL

    def __init__(self) -> None:
        self.summary: dict[str, Any] = {"id": 123, "currency": "EUR", "totalValue": 1000,
                                        "cash": {"availableToTrade": 1000, "reservedForOrders": 0}}
        self.held: list[dict[str, Any]] = []
        self.metadata = [{"ticker": "AAA_US_EQ", "type": "STOCK", "currencyCode": "USD"},
                         {"ticker": "BBB_US_EQ", "type": "ETF", "currencyCode": "USD"}]
        self.pending: list[dict[str, Any]] = []
        self.history: list[dict[str, Any]] = []
        self.flows: list[dict[str, Any]] = []
        self.details: dict[str, dict[str, Any]] = {}
        self.sent: list[tuple[str, float, float]] = []
        self.failure: Exception | None = None
        self.intent_check: Path | None = None

    def account_summary(self) -> dict[str, Any]:
        return deepcopy(self.summary)

    def positions(self) -> list[dict[str, Any]]:
        return deepcopy(self.held)

    def instruments(self) -> list[dict[str, Any]]:
        return deepcopy(self.metadata)

    def pending_orders(self) -> list[dict[str, Any]]:
        return deepcopy(self.pending)

    def order_history(self) -> list[dict[str, Any]]:
        return deepcopy(self.history)

    def transactions(self) -> list[dict[str, Any]]:
        return deepcopy(self.flows)

    def order(self, order_id: str) -> dict[str, Any]:
        if order_id not in self.details:
            raise status_error(404)
        return deepcopy(self.details[order_id])

    def place_limit_order(self, ticker: str, quantity: float, limit_price: float) -> dict[str, Any]:
        if self.intent_check is not None:
            intent = json.loads(self.intent_check.read_text())["orders"][-1]
            assert intent["phase"] == "intent"
            assert intent["ticker"] == ticker
        self.sent.append((ticker, quantity, limit_price))
        if self.failure is not None:
            raise self.failure
        order_id = str(len(self.sent))
        response = {"id": order_id, "instrument": {"ticker": ticker},
                    "side": "BUY" if quantity > 0 else "SELL", "quantity": abs(quantity), "status": "NEW"}
        self.details[order_id] = response
        return deepcopy(response)


def status_error(code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://demo.trading212.com/api/v0/equity/orders/limit")
    response = httpx.Response(code, request=request)
    return httpx.HTTPStatusError("private broker body must not be returned", request=request, response=response)


def plan(*orders: tuple[str, float]) -> RebalancePlan:
    return RebalancePlan(tuple(PlannedOrder(ticker.split("_")[0], ticker, qty, abs(qty) * 100)
                               for ticker, qty in (orders or (("AAA_US_EQ", 0.2),))), ())


def run(broker: FakeBroker, path: Path, selected: RebalancePlan | None = None, **kwargs: Any) -> dict[str, Any]:
    options: dict[str, Any] = dict(limits=LIMITS, state_dir=path, cycle_id="2026-10",
                                   price_limits={"AAA_US_EQ": 100.0, "BBB_US_EQ": 100.0},
                                   fx_native_to_eur=1.0, now=NOW, execute=True,
                                   quote_times_utc={"AAA_US_EQ": NOW, "BBB_US_EQ": NOW, "EURUSD=X": NOW})
    options.update(kwargs)
    return execute_demo_plan(broker, selected or plan(), **options)


def holding(qty: float = 1, available: float = 1, value: float = 100) -> dict[str, Any]:
    return {"instrument": {"ticker": "AAA_US_EQ", "currency": "USD"}, "quantity": qty,
            "quantityAvailableForTrading": available, "currentPrice": 100,
            "walletImpact": {"currency": "EUR", "currentValue": value}}


def test_dry_run_records_baseline_without_post(tmp_path: Path) -> None:
    broker = FakeBroker()
    result = run(broker, tmp_path, execute=False)
    assert result["status"] == "dry-run"
    assert broker.sent == []
    assert (tmp_path / "risk.json").exists()
    assert result["planned"]["time_validity"] == "DAY"


def test_deleted_order_journal_cannot_enable_duplicate_submission(tmp_path: Path) -> None:
    broker = FakeBroker()
    assert run(broker, tmp_path)["status"] == "submitted"
    (tmp_path / "orders.json").unlink()
    assert run(broker, tmp_path)["status"] == "blocked"
    assert len(broker.sent) == 1


def test_durable_intent_precedes_single_submission(tmp_path: Path) -> None:
    broker = FakeBroker()
    broker.intent_check = tmp_path / "orders.json"
    result = run(broker, tmp_path, plan(("AAA_US_EQ", .2), ("BBB_US_EQ", .2)))
    assert result["status"] == "submitted"
    assert len(broker.sent) == 1
    assert json.loads((tmp_path / "orders.json").read_text())["orders"][0]["phase"] == "accepted"


def test_sells_precede_buys(tmp_path: Path) -> None:
    broker = FakeBroker()
    broker.held = [holding()]
    run(broker, tmp_path, plan(("BBB_US_EQ", .2), ("AAA_US_EQ", -.1)))
    assert broker.sent == [("AAA_US_EQ", -.1, 100)]


@pytest.mark.parametrize("code", [400, 403])
def test_known_denial_requires_manual_review_before_later_run(tmp_path: Path, code: int) -> None:
    broker = FakeBroker()
    broker.failure = status_error(code)
    assert run(broker, tmp_path)["status"] == "blocked"
    broker.failure = None
    assert run(broker, tmp_path)["status"] == "blocked"
    assert len(broker.sent) == 1


def test_final_quote_age_check_blocks_stale_regular_session_order(tmp_path: Path) -> None:
    broker = FakeBroker()
    at = NOW.replace(hour=15)
    old = at - timedelta(minutes=16)
    result = run(broker, tmp_path, now=at, quote_times_utc={"AAA_US_EQ": old, "EURUSD=X": old})
    assert result["status"] == "blocked"
    assert broker.sent == []


def test_missing_execution_quote_times_never_reaches_post(tmp_path: Path) -> None:
    broker = FakeBroker()
    assert run(broker, tmp_path, quote_times_utc=None)["status"] == "blocked"
    assert broker.sent == []


def test_off_hours_day_limit_can_queue_with_recent_completed_quote(tmp_path: Path) -> None:
    broker = FakeBroker()
    old = NOW - timedelta(hours=16)
    assert run(broker, tmp_path, quote_times_utc={"AAA_US_EQ": old, "EURUSD=X": old})["status"] == "submitted"


@pytest.mark.parametrize("failure", [httpx.ReadTimeout("private-token"), status_error(500), status_error(429)])
def test_unknown_post_outcome_is_sticky_and_never_retried(tmp_path: Path, failure: Exception) -> None:
    broker = FakeBroker()
    broker.failure = failure
    first = run(broker, tmp_path)
    broker.failure = None
    second = run(broker, tmp_path)
    assert first["status"] == second["status"] == "blocked"
    assert len(broker.sent) == 1
    assert "private-token" not in str(first)
    assert "private broker body" not in str(first)


def test_crashed_intent_blocks_later_send(tmp_path: Path) -> None:
    broker = FakeBroker()
    run(broker, tmp_path)
    target = tmp_path / "orders.json"
    journal = json.loads(target.read_text())
    journal["orders"][0]["phase"] = "intent"
    target.write_text(json.dumps(journal))
    assert run(broker, tmp_path)["status"] == "blocked"
    assert len(broker.sent) == 1


@pytest.mark.parametrize("status", ["NEW", "PARTIALLY_FILLED", "CANCELLING", "UNKNOWN"])
def test_nonterminal_previous_order_blocks(tmp_path: Path, status: str) -> None:
    broker = FakeBroker()
    run(broker, tmp_path)
    broker.details["1"]["status"] = status
    assert run(broker, tmp_path)["status"] == "blocked"
    assert len(broker.sent) == 1


@pytest.mark.parametrize("status", ["FILLED", "CANCELLED", "REJECTED", "EXPIRED"])
def test_terminal_consumes_cycle_ticker_and_prevents_duplicate(tmp_path: Path, status: str) -> None:
    broker = FakeBroker()
    run(broker, tmp_path)
    broker.details["1"]["status"] = status
    result = run(broker, tmp_path)
    assert result["status"] == "completed"
    assert result["reconciled"][0]["status"] == status
    assert len(broker.sent) == 1


def test_pending_only_404_requires_history_not_assumed_fill(tmp_path: Path) -> None:
    broker = FakeBroker()
    run(broker, tmp_path)
    terminal = broker.details.pop("1")
    terminal["status"] = "FILLED"
    assert run(broker, tmp_path)["status"] == "blocked"
    broker.history = [{"order": terminal, "fill": None}]
    assert run(broker, tmp_path)["status"] == "completed"
    assert len(broker.sent) == 1


def test_next_cycle_can_trade_after_terminal(tmp_path: Path) -> None:
    broker = FakeBroker()
    run(broker, tmp_path)
    broker.details["1"]["status"] = "FILLED"
    assert run(broker, tmp_path, cycle_id="2026-11")["status"] == "submitted"
    assert len(broker.sent) == 2


@pytest.mark.parametrize("mutation", ["pending", "reserved", "account", "currency", "cash", "cap", "type", "fx"])
def test_account_and_order_guards(tmp_path: Path, mutation: str) -> None:
    broker = FakeBroker()
    kwargs: dict[str, Any] = {}
    if mutation == "pending":
        broker.pending = [{"id": 777}]
    elif mutation == "reserved":
        broker.summary["cash"]["reservedForOrders"] = 1
    elif mutation == "account":
        run(broker, tmp_path, execute=False)
        broker.summary["id"] = 999
    elif mutation == "currency":
        broker.summary["currency"] = "USD"
    elif mutation == "cash":
        broker.summary["cash"]["availableToTrade"] = 20  # EUR20.20 required
    elif mutation == "cap":
        broker.held = [holding(qty=1.4, available=1.4, value=140)]
    elif mutation == "type":
        broker.metadata[0]["type"] = "CFD"
    else:
        kwargs["fx_native_to_eur"] = float("nan")
    assert run(broker, tmp_path, **kwargs)["status"] == "blocked"
    assert broker.sent == []


def test_max_order_includes_one_percent_reserve(tmp_path: Path) -> None:
    broker = FakeBroker()
    assert run(broker, tmp_path, plan(("AAA_US_EQ", .5)))["status"] == "blocked"
    assert broker.sent == []


def test_sell_cannot_exceed_available_shares(tmp_path: Path) -> None:
    broker = FakeBroker()
    broker.held = [holding(available=.1)]
    assert run(broker, tmp_path, plan(("AAA_US_EQ", -.2)))["status"] == "blocked"
    assert broker.sent == []


def test_daily_loss_blocks_before_post(tmp_path: Path) -> None:
    broker = FakeBroker()
    run(broker, tmp_path, execute=False)
    broker.summary["totalValue"] = 990
    result = run(broker, tmp_path, now=NOW + timedelta(minutes=1))
    assert result["status"] == "blocked"
    assert any("daily" in violation for violation in result["violations"])
    assert broker.sent == []


def test_external_cashflow_requires_manual_rebaseline(tmp_path: Path) -> None:
    broker = FakeBroker()
    run(broker, tmp_path, execute=False)
    broker.flows = [{"reference": "deposit-1", "dateTime": (NOW + timedelta(minutes=1)).isoformat(),
                     "amount": 100, "currency": "EUR", "type": "DEPOSIT"}]
    assert run(broker, tmp_path, now=NOW + timedelta(minutes=2))["status"] == "blocked"
    assert broker.sent == []


def test_realised_sell_losses_trigger_cooldown(tmp_path: Path) -> None:
    broker = FakeBroker()
    run(broker, tmp_path, execute=False)
    for i in range(3):
        broker.history.append({"order": {"id": str(i + 1), "side": "SELL"}, "fill": {
            "id": str(i + 1), "quantity": .1, "price": 100,
            "filledAt": (NOW + timedelta(minutes=i + 1)).isoformat(),
            "walletImpact": {"currency": "EUR", "realisedProfitLoss": -1}}})
    result = run(broker, tmp_path, now=NOW + timedelta(minutes=5))
    assert result["status"] == "blocked"
    assert any("cooldown" in violation for violation in result["violations"])
    assert broker.sent == []


@pytest.mark.parametrize("mutation", ["live", "lock", "journal", "unknown_transaction", "bad_plan", "stale"])
def test_more_fail_closed_paths(tmp_path: Path, mutation: str) -> None:
    broker = FakeBroker()
    selected = plan()
    kwargs: dict[str, Any] = {}
    if mutation == "live":
        broker.base_url = TRADING212_LIVE_URL
    elif mutation == "lock":
        (tmp_path / "execution.lock").write_text("another run")
    elif mutation == "journal":
        (tmp_path / "orders.json").write_text("not json")
    elif mutation == "unknown_transaction":
        broker.flows = [{"type": "UNRECOGNISED"}]
    elif mutation == "bad_plan":
        selected = RebalancePlan((), ("upstream risk failure",))
    else:
        kwargs["price_limits"] = {"AAA_US_EQ": 120}
    assert run(broker, tmp_path, selected, **kwargs)["status"] == "blocked"
    assert broker.sent == []


def test_read_failure_does_not_expose_arbitrary_broker_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    broker = FakeBroker()

    def fail() -> dict[str, Any]:
        raise RuntimeError("API-secret-should-never-be-output")

    monkeypatch.setattr(broker, "account_summary", fail)
    result = run(broker, tmp_path)
    assert result["status"] == "blocked"
    assert "API-secret" not in str(result)
    assert broker.sent == []
