"""LIVE (real money) Trading 212 runner. Dry-run by default; --approve sends ONE order after you type "yes live".

It refuses before any network call unless config/private_live.yaml opens the gate. Only the owner
edits that file (enabled, not_before, every checklist item). There is no --execute: a live order
exists only as an order you were shown and typed the confirmation for, in a real terminal.
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

from ai_trader.ai_demo import load_live_signal, signal_cycle_id, sleeve_holdings, tradable_usd_map
from ai_trader.ai_signal import live_gate_blockers, load_live_config, load_watchlist
from ai_trader.approval import LIVE_VENUE, ApprovedOrder, run_with_approval
from ai_trader.decision_log import log_decision
from ai_trader.demo_execution import execute_live_plan
from ai_trader.demo_market import DemoQuotes, load_demo_quotes
from ai_trader.execution_state import demo_run_lock
from ai_trader.paper_adapter import Trading212Live, credentials_from_env
from ai_trader.rebalance import plan_rebalance
from ai_trader.risk import load_limits

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "data" / "live_execution"
LOG_PATH = ROOT / "logs" / "live_decisions.jsonl"
CONFIRM_PHRASE = "yes live"


def blocked(reason: str, **extra: Any) -> int:
    print(json.dumps({"environment": "live", "status": "blocked", "reason": reason, **extra}, sort_keys=True),
          file=sys.stderr)
    return 2


def run(args: argparse.Namespace) -> int:
    config = load_live_config(Path(args.live_config))
    blockers = live_gate_blockers(config, datetime.now(UTC).date())
    if blockers:  # nothing is read from or sent to the broker while the gate is closed
        return blocked("live gate is closed", blockers=blockers)
    if args.approve and not sys.stdin.isatty():
        return blocked("approval needs a person typing at a terminal; scripts and agents cannot answer for you")
    sleeve_eur = float(config["sleeve_eur"])
    budget = sleeve_eur if args.budget_eur is None else args.budget_eur
    if not isfinite(budget) or budget <= 0 or budget > sleeve_eur:
        return blocked("budget must be positive and at most the sleeve_eur in the live config")
    limits = load_limits(ROOT / "config" / "risk_limits.yaml")
    watchlist = load_watchlist(Path(args.watchlist))
    try:
        signal = load_live_signal(Path(args.signal), watchlist, limits, today=datetime.now(UTC).date(), sleeve_eur=budget)
    except ValueError as error:
        return blocked("signal rejected", violations=str(error).split("; "))
    cycle = "live-" + signal_cycle_id(signal)
    with demo_run_lock(ROOT / "data" / "live_runner.lock"):
        truststore.inject_into_ssl()
        broker = Trading212Live(*credentials_from_env(live=True))
        try:
            tradable = tradable_usd_map(watchlist, broker.instruments())
            unsupported = sorted(set(signal.targets) - set(tradable))
            if unsupported:
                result: dict[str, Any] = {"status": "blocked", "submitted": [], "reconciled": [], "violations": [
                    f"{symbol}: no unique USD-listed Trading 212 instrument; this executor cannot trade it"
                    for symbol in unsupported]}
            else:
                holdings, outside = sleeve_holdings(broker.positions(), tradable)
                symbols = sorted(set(signal.targets) | set(holdings))
                if symbols:
                    with requests.Session() as http:
                        quotes = load_demo_quotes(http, symbols, ROOT / "data", datetime.now(UTC))
                else:
                    quotes = DemoQuotes({}, 1.0, {})
                now = datetime.now(UTC)
                if any(stamp > now + timedelta(seconds=60) or now - stamp > timedelta(days=4)
                       for stamp in quotes.quote_times_utc.values()):
                    raise ValueError("quote age failed the pre-submission check")
                mapping = {symbol: tradable[symbol] for symbol in symbols}
                plan = plan_rebalance(signal.targets, quotes.prices_usd, quotes.fx_usd_to_eur,
                                      holdings, budget, mapping, limits)
                price_limits = {order.api_ticker: round(quotes.prices_usd[order.symbol]
                                * (1.01 if order.quantity > 0 else .99), 2) for order in plan.orders}
                quote_times = {**{mapping[symbol]: quotes.quote_times_utc[symbol] for symbol in symbols},
                               **({"EURUSD=X": quotes.quote_times_utc["EURUSD=X"]} if symbols else {})}

                def attempt(execute: bool, approved: ApprovedOrder | None = None) -> dict[str, Any]:
                    return execute_live_plan(
                        broker, plan, live_blockers=blockers, limits=limits, state_dir=STATE_DIR, cycle_id=cycle,
                        price_limits=price_limits, fx_native_to_eur=quotes.fx_usd_to_eur, execute=execute,
                        quote_times_utc=quote_times, approved=approved)

                if args.approve:
                    result = run_with_approval(
                        lambda: attempt(False), lambda approved: attempt(True, approved), input,
                        symbols={order.api_ticker: order.symbol for order in plan.orders},
                        fx_native_to_eur=quotes.fx_usd_to_eur, confirm_phrase=CONFIRM_PHRASE, venue=LIVE_VENUE)
                else:
                    result = attempt(False)
                result["outside_sleeve_tickers"] = outside
            result.update(environment="live", source="ai-signal", signal_date=signal.as_of.isoformat(),
                          budget_eur=budget, targets=signal.targets, reasons=signal.reasons)
            action = ("blocked" if result["violations"] else "executed" if result["submitted"]
                      else "cancelled" if result.get("status") == "cancelled" else "dry-run")
            log_decision(LOG_PATH, action=action, mode="live", targets=signal.targets,
                         violations=result["violations"], rationale=json.dumps(result, allow_nan=False, sort_keys=True))
            print(json.dumps(result, allow_nan=False, sort_keys=True))
            return 2 if result["violations"] else 0
        finally:
            broker.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal", required=True, help="live-sleeve analyst signal JSON (sleeve must be 'live')")
    parser.add_argument("--live-config", default=str(ROOT / "config" / "private_live.yaml"))
    parser.add_argument("--watchlist", default=str(ROOT / "config" / "private_watchlist.yaml"))
    parser.add_argument("--approve", action="store_true",
                        help='show the exact order and send ONE only after you type "yes live" (needs a terminal)')
    parser.add_argument("--budget-eur", type=float, default=None, help="deploy at most this much (default: sleeve_eur)")
    args = parser.parse_args()
    try:
        return run(args)
    except httpx.HTTPStatusError as error:
        return blocked("broker permission or HTTP failure; no automatic POST retry",
                       http_status=error.response.status_code)
    except (ValueError, RuntimeError, OSError, KeyError, httpx.RequestError) as error:
        return blocked("data/state validation failed; inspect live logs and configuration",
                       error_type=type(error).__name__)


if __name__ == "__main__":
    raise SystemExit(main())
