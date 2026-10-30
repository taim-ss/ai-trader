"""Demo-only runner. Plans by default; --execute --demo-only submits.

Default source is the monthly momentum baseline. With --signal FILE the targets come instead from a
validated AI analyst signal (see ai_trader.ai_signal) restricted to USD-listed watchlist stocks/ETFs.
Default virtual deployment budget EUR 200. At most one DAY limit order per run.
Later runs reconcile prior orders. No live credentials or live environment option.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from math import isfinite
from pathlib import Path
from typing import Any

import httpx
import requests
import truststore

from ai_trader.ai_demo import load_demo_signal, signal_cycle_id, sleeve_holdings, tradable_usd_map
from ai_trader.ai_signal import load_watchlist
from ai_trader.approval import ApprovedOrder, run_with_approval
from ai_trader.decision_log import log_decision
from ai_trader.demo_execution import execute_demo_plan
from ai_trader.demo_market import DemoQuotes, load_demo_market, load_demo_quotes
from ai_trader.execution_state import demo_run_lock
from ai_trader.paper_adapter import Trading212Paper, credentials_from_env
from ai_trader.rebalance import plan_rebalance, resolve_api_tickers
from ai_trader.risk import load_limits, validate_targets
from ai_trader.strategy import StrategyConfig, target_weights

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "XLE", "XLK", "XLV", "XLF", "XLI", "XLP", "XLU"]
LOG_PATH = ROOT / "logs" / "decisions.jsonl"
STATE_DIR = ROOT / "data" / "demo_execution"


def verified_holdings(positions: list[dict[str, Any]], api_tickers: dict[str, str]) -> dict[str, float]:
    """Preserve canonical identity; refuse unknown listings or duplicate positions."""
    symbols = {ticker: symbol for symbol, ticker in api_tickers.items()}
    holdings: dict[str, float] = {}
    for position in positions:
        ticker = str(position["instrument"]["ticker"])
        if ticker not in symbols:
            raise ValueError(f"{ticker}: holding outside verified USD universe")
        symbol = symbols[ticker]
        quantity = float(position["quantity"])
        if symbol in holdings or not isfinite(quantity) or quantity < 0:
            raise ValueError(f"{ticker}: duplicate position or invalid quantity")
        holdings[symbol] = quantity
    return holdings


def eligible_universe(instruments: list[dict[str, Any]]) -> tuple[list[str], dict[str, str]]:
    """Use exact available research ETFs; never substitute another fund."""
    available: list[str] = []
    mapping: dict[str, str] = {}
    for symbol in UNIVERSE:
        matches = [row for row in instruments if str(row.get("ticker", "")).split("_")[0] == symbol]
        if not matches:
            continue
        if len(matches) != 1 or matches[0].get("currencyCode") != "USD" or matches[0].get("type") != "ETF":
            raise ValueError(f"{symbol}: ambiguous, wrong-currency or non-ETF broker instrument")
        mapping.update(resolve_api_tickers([symbol], instruments, expected_currency="USD"))
        available.append(symbol)
    if not available:
        raise ValueError("none of the research ETFs is available in this demo account")
    return available, mapping


def cycle_targets(history: Any, now: datetime, budget: float, universe: list[str],
                  config: StrategyConfig) -> tuple[str, dict[str, float], str]:
    """Freeze one signal/configuration per month; quote prices remain fresh."""
    cycle = now.strftime("%Y-%m")
    path = STATE_DIR / f"signal-{cycle}.json"
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("universe") != universe or record.get("budget_eur") != budget or record.get("cap") != config.single_name_cap:
            raise ValueError("demo signal configuration changed within a monthly cycle")
        targets = record["targets"]
        if not isinstance(targets, dict) or not set(targets).issubset(universe):
            raise ValueError("invalid saved demo targets")
        return cycle, targets, str(record["signal_date"])
    signal_date = history.index[-1].date()
    targets = target_weights(history, signal_date, config)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    record = {"cycle": cycle, "signal_date": signal_date.isoformat(), "universe": universe,
              "budget_eur": budget, "cap": config.single_name_cap, "targets": targets}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, allow_nan=False, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return cycle, targets, signal_date.isoformat()


def run(args: argparse.Namespace) -> int:
    limits = load_limits(ROOT / "config" / "risk_limits.yaml")
    with demo_run_lock(ROOT / "data" / "demo_runner.lock"):
        truststore.inject_into_ssl()
        broker = Trading212Paper(*credentials_from_env())
        try:
            instruments = broker.instruments()
            universe, mapping = eligible_universe(instruments)
            positions = broker.positions()
            holdings = verified_holdings(positions, mapping)
            with requests.Session() as http:
                market = load_demo_market(http, universe, ROOT / "data", datetime.now(UTC))
            now = datetime.now(UTC)
            if any(stamp > now + timedelta(seconds=60) or now - stamp > timedelta(days=4)
                   for stamp in market.quote_times_utc.values()):
                raise ValueError("quote age failed the pre-submission check")
            config = StrategyConfig(top_n=3, single_name_cap=limits.max_position_weight)
            cycle, targets, signal_date = cycle_targets(market.history, now, args.budget_eur, universe, config)
            if validate_targets(targets, limits):
                raise ValueError("saved demo target weights violate risk limits")
            plan = plan_rebalance(targets, market.quote_prices_usd, market.fx_usd_to_eur,
                                  holdings, args.budget_eur, mapping, limits)
            price_limits = {order.api_ticker: round(market.quote_prices_usd[order.symbol]
                            * (1.01 if order.quantity > 0 else .99), 2) for order in plan.orders}
            result = execute_demo_plan(broker, plan, limits=limits, state_dir=STATE_DIR,
                        cycle_id=cycle, price_limits=price_limits,
                        fx_native_to_eur=market.fx_usd_to_eur, execute=args.execute,
                        quote_times_utc={**{mapping[s]: market.quote_times_utc[s] for s in universe},
                                         "EURUSD=X": market.quote_times_utc["EURUSD=X"]})
            result.update(environment="demo", signal_date=signal_date, virtual_budget_eur=args.budget_eur,
                          targets=targets, eligible_universe=universe,
                          unavailable_symbols=[s for s in UNIVERSE if s not in universe])
            action = "blocked" if result["violations"] else "executed" if result["submitted"] else "dry-run" if not args.execute else "hold"
            log_decision(LOG_PATH, action=action, mode="paper", targets=targets,
                         violations=result["violations"], rationale=json.dumps(result, allow_nan=False, sort_keys=True))
            print(json.dumps(result, allow_nan=False, sort_keys=True))
            return 2 if result["violations"] else 0
        finally:
            broker.close()


def run_signal(args: argparse.Namespace) -> int:
    """Plan, and with --execute submit one order, from a validated AI analyst signal. Demo only."""
    limits = load_limits(ROOT / "config" / "risk_limits.yaml")
    watchlist = load_watchlist(Path(args.watchlist))
    try:
        signal = load_demo_signal(Path(args.signal), watchlist, limits,
                                  today=datetime.now(UTC).date(), sleeve_eur=args.budget_eur)
    except ValueError as error:  # validation text names fields only; no secrets or broker data
        print(json.dumps({"environment": "demo", "status": "blocked", "reason": "signal rejected",
                          "violations": str(error).split("; ")}), file=sys.stderr)
        return 2
    cycle = signal_cycle_id(signal)
    with demo_run_lock(ROOT / "data" / "demo_runner.lock"):
        truststore.inject_into_ssl()
        broker = Trading212Paper(*credentials_from_env())
        try:
            tradable = tradable_usd_map(watchlist, broker.instruments())
            unsupported = sorted(set(signal.targets) - set(tradable))
            if unsupported:
                result: dict[str, Any] = {"status": "blocked", "submitted": [], "reconciled": [], "violations": [
                    f"{symbol}: no unique USD-listed Trading 212 instrument; the demo executor cannot trade it"
                    for symbol in unsupported]}
            else:
                holdings, outside = sleeve_holdings(broker.positions(), tradable)
                symbols = sorted(set(signal.targets) | set(holdings))
                if symbols:
                    with requests.Session() as http:
                        quotes = load_demo_quotes(http, symbols, ROOT / "data", datetime.now(UTC))
                else:  # all-cash signal and nothing held in the sleeve: no quotes needed, no orders possible
                    quotes = DemoQuotes({}, 1.0, {})
                now = datetime.now(UTC)
                if any(stamp > now + timedelta(seconds=60) or now - stamp > timedelta(days=4)
                       for stamp in quotes.quote_times_utc.values()):
                    raise ValueError("quote age failed the pre-submission check")
                mapping = {symbol: tradable[symbol] for symbol in symbols}
                plan = plan_rebalance(signal.targets, quotes.prices_usd, quotes.fx_usd_to_eur,
                                      holdings, args.budget_eur, mapping, limits)
                price_limits = {order.api_ticker: round(quotes.prices_usd[order.symbol]
                                * (1.01 if order.quantity > 0 else .99), 2) for order in plan.orders}
                quote_times = {**{mapping[symbol]: quotes.quote_times_utc[symbol] for symbol in symbols},
                               **({"EURUSD=X": quotes.quote_times_utc["EURUSD=X"]} if symbols else {})}

                def attempt(execute: bool, approved: ApprovedOrder | None = None) -> dict[str, Any]:
                    return execute_demo_plan(
                        broker, plan, limits=limits, state_dir=STATE_DIR, cycle_id=cycle,
                        price_limits=price_limits, fx_native_to_eur=quotes.fx_usd_to_eur, execute=execute,
                        quote_times_utc=quote_times, approved=approved, halt_file=ROOT / "HALT")

                if args.approve:  # a human types "yes" for the exact order shown; nothing else can send it
                    result = run_with_approval(
                        lambda: attempt(False), lambda approved: attempt(True, approved), input,
                        symbols={order.api_ticker: order.symbol for order in plan.orders},
                        fx_native_to_eur=quotes.fx_usd_to_eur)
                else:
                    result = attempt(args.execute)
                result["outside_sleeve_tickers"] = outside
                if result["status"] == "completed" and plan.orders:
                    result["warning"] = ("every planned ticker already reached a terminal order state in this cycle; "
                                         "an expired or cancelled order is not retried until the signal changes")
            result.update(environment="demo", source="ai-signal", signal_date=signal.as_of.isoformat(),
                          virtual_budget_eur=args.budget_eur, targets=signal.targets, reasons=signal.reasons)
            armed = args.execute or (args.approve and result.get("status") != "cancelled")
            action = ("blocked" if result["violations"] else "executed" if result["submitted"]
                      else "cancelled" if result.get("status") == "cancelled"
                      else "dry-run" if not armed else "hold")
            log_decision(LOG_PATH, action=action, mode="paper", targets=signal.targets,
                         violations=result["violations"], rationale=json.dumps(result, allow_nan=False, sort_keys=True))
            print(json.dumps(result, allow_nan=False, sort_keys=True))  # before anything else can fail
            if armed and not result["violations"]:
                try:
                    _log_forecasts_once(signal, cycle)
                except OSError:
                    print(json.dumps({"warning": "forecast logging failed; rerun to retry", "cycle": cycle}),
                          file=sys.stderr)
            return 2 if result["violations"] else 0
        finally:
            broker.close()


def _log_forecasts_once(signal: Any, cycle: str) -> None:
    """Record the signal's scoreable forecasts once per cycle. The marker is written last, so a crash
    mid-way can only duplicate forecasts on a rerun, never lose them."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    marker = STATE_DIR / f"forecasts-{cycle}.logged"
    if marker.exists():
        return
    for forecast in signal.forecasts:
        log_decision(LOG_PATH, action="forecast", mode="paper", targets=signal.targets, violations=[],
                     rationale=f"AI signal {cycle}", resolution_date=forecast.resolution_date,
                     probability=forecast.probability, event=forecast.event)
    marker.write_text("forecasts logged\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal", help="validated AI analyst signal JSON (demo sleeve) to plan from")
    parser.add_argument("--watchlist", default=str(ROOT / "config" / "private_watchlist.yaml"),
                        help="private watchlist used with --signal")
    parser.add_argument("--execute", action="store_true", help="submit at most one demo DAY limit order")
    parser.add_argument("--approve", action="store_true",
                        help="with --signal: show the exact order and send it only after you type yes (needs a terminal)")
    parser.add_argument("--demo-only", action="store_true", help="acknowledge virtual-money execution")
    parser.add_argument("--budget-eur", type=float, default=200.0, help="virtual deployment budget (default EUR 200)")
    args = parser.parse_args()
    if (args.execute or args.approve) and not args.demo_only:
        print("blocked: persistent daily-loss and order controls require explicit --demo-only for submission", file=sys.stderr)
        return 2
    if args.approve and (args.execute or not args.signal):
        print("blocked: --approve works only with --signal and cannot be combined with --execute", file=sys.stderr)
        return 2
    if args.approve and not sys.stdin.isatty():
        print("blocked: approval needs a person typing at a terminal; scripts and agents cannot answer for you", file=sys.stderr)
        return 2
    if not isfinite(args.budget_eur) or args.budget_eur <= 0:
        parser.error("budget must be finite and positive")
    try:
        return run_signal(args) if args.signal else run(args)
    except httpx.HTTPStatusError as error:
        print(json.dumps({"environment": "demo", "status": "blocked", "http_status": error.response.status_code,
            "reason": "broker permission or HTTP failure; no automatic POST retry"}), file=sys.stderr)
        return 2
    except (ValueError, RuntimeError, OSError, KeyError, httpx.RequestError) as error:
        print(json.dumps({"environment": "demo", "status": "blocked", "error_type": type(error).__name__,
                          "reason": "data/state validation failed; inspect demo logs and configuration"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
