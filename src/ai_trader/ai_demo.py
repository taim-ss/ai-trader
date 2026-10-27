"""Glue between validated AI signals and the demo executor. Pure functions, no network.

Only watchlist entries that map to exactly one USD-listed Trading 212 stock/ETF are executable.
EUR-listed core/gold ETFs are refused rather than silently dropped, so the analyst sees why.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

from ai_trader.ai_signal import AiSignal, WatchEntry, parse_signal
from ai_trader.risk import RiskLimits

EXECUTABLE_TYPES = frozenset({"STOCK", "ETF"})


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite JSON constant {name} is not allowed")


def load_demo_signal(path: Path, watchlist: Mapping[str, WatchEntry], limits: RiskLimits, *,
                     today: date, sleeve_eur: float) -> AiSignal:
    """Read and validate a signal file for the demo sleeve. Live or live-gate signals are refused."""
    # utf-8-sig: Windows PowerShell 5.1 writes a BOM; tolerate it so scheduled writers do not break runs.
    raw = json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=_reject_constant)
    if not isinstance(raw, dict):
        raise ValueError("signal file must contain a JSON object")
    signal = parse_signal(raw, watchlist, limits, today=today, sleeve_eur=sleeve_eur)
    if signal.sleeve != "demo" or signal.run == "live-gate":
        raise ValueError("the demo runner only accepts demo-sleeve signals")
    return signal


def signal_cycle_id(signal: AiSignal) -> str:
    """Date, run and a digest of the targets, so a corrected signal never inherits an old cycle's 'done' state."""
    digest = hashlib.sha256(json.dumps(signal.targets, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    return f"ai-{signal.as_of.isoformat()}-{signal.run}-{digest}"


def tradable_usd_map(watchlist: Mapping[str, WatchEntry], instruments: list[dict[str, Any]]) -> dict[str, str]:
    """Watchlist symbol -> Trading 212 ticker, only for a unique USD stock/ETF instrument."""
    by_ticker: dict[str, list[dict[str, Any]]] = {}
    for item in instruments:
        by_ticker.setdefault(str(item.get("ticker", "")), []).append(item)
    mapping: dict[str, str] = {}
    for symbol, entry in watchlist.items():
        rows = by_ticker.get(entry.t212_ticker, [])
        if (len(rows) == 1 and rows[0].get("currencyCode") == "USD" and entry.currency == "USD"
                and rows[0].get("type") in EXECUTABLE_TYPES):
            mapping[symbol] = entry.t212_ticker
    return mapping


def sleeve_holdings(positions: list[dict[str, Any]], tradable: Mapping[str, str]) -> tuple[dict[str, float], list[str]]:
    """Quantities for executable watchlist symbols, plus broker tickers left outside the sleeve."""
    symbols = {ticker: symbol for symbol, ticker in tradable.items()}
    holdings: dict[str, float] = {}
    outside: list[str] = []
    for position in positions:
        ticker = str(position["instrument"]["ticker"])
        quantity = float(position["quantity"])
        if quantity < 0 or quantity != quantity or quantity in (float("inf"), float("-inf")):
            raise ValueError(f"{ticker}: invalid position quantity")
        if ticker not in symbols:
            outside.append(ticker)
            continue
        symbol = symbols[ticker]
        if symbol in holdings:
            raise ValueError(f"{ticker}: duplicate broker position")
        holdings[symbol] = quantity
    return holdings, sorted(outside)
