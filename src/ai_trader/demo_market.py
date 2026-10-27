"""Validated Yahoo research data for a demo-only order plan.

Adjusted closes are signals, raw closes are historical valuations, and current
regular-market quotes are separate order-reference estimates. Yahoo is not an
execution-quality quote feed. Every run fetches fresh metadata; cached snapshots
are audit evidence only and are never used to make a stale quote look current.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from math import isfinite
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import pandas as pd

from ai_trader.data import CHART_URL, USER_AGENT, HttpGet

MAX_QUOTE_AGE = timedelta(days=4)
ACQUISITION_CLOCK_TOLERANCE = timedelta(seconds=60)
MIN_HISTORY_ROWS = 253
FX_SYMBOL = "EURUSD=X"
NEW_YORK = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class DemoMarket:
    history: pd.DataFrame
    prices_usd: dict[str, float]
    fx_usd_to_eur: float
    signal_date: date
    quote_times_utc: dict[str, datetime]
    quote_prices_usd: dict[str, float]


def _positive_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
        raise ValueError(f"{label}: price or number must be finite and positive")
    return float(value)


def _fetch(http: HttpGet, symbol: str, now: datetime) -> dict[str, Any]:
    start = now.astimezone(NEW_YORK).date() - timedelta(days=800)
    period_start = datetime(start.year, start.month, start.day, tzinfo=UTC)
    response = http.get(
        CHART_URL.format(symbol=symbol),
        params={"period1": int(period_start.timestamp()), "period2": int(now.timestamp()),
                "interval": "1d", "events": "div,splits", "includeAdjustedClose": "true"},
        headers={"User-Agent": USER_AGENT}, timeout=20.0,
    )
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise ValueError(f"{symbol}: invalid chart payload")
    chart = body.get("chart")
    if not isinstance(chart, dict) or chart.get("error"):
        raise ValueError(f"{symbol}: chart error")
    results = chart.get("result")
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
        raise ValueError(f"{symbol}: expected one unambiguous chart result")
    return cast(dict[str, Any], results[0])


def _quote(result: dict[str, Any], symbol: str, now: datetime) -> tuple[float, datetime]:
    metadata = result.get("meta")
    if not isinstance(metadata, dict) or metadata.get("symbol") != symbol:
        raise ValueError(f"{symbol}: returned symbol does not match requested identity")
    if metadata.get("currency") != "USD":
        raise ValueError(f"{symbol}: quote currency must be USD, including EURUSD quote units")
    price = _positive_number(metadata.get("regularMarketPrice"), f"{symbol} quote price")
    raw_timestamp = _positive_number(metadata.get("regularMarketTime"), f"{symbol} quote timestamp")
    try:
        timestamp = datetime.fromtimestamp(raw_timestamp, tz=UTC)
    except (OverflowError, OSError, ValueError) as exc:
        raise ValueError(f"{symbol}: invalid quote timestamp") from exc
    if timestamp > now + ACQUISITION_CLOCK_TOLERANCE or now - timestamp > MAX_QUOTE_AGE:
        raise ValueError(f"{symbol}: quote timestamp is beyond the 60-second acquisition tolerance or more than four calendar days old")
    return price, timestamp


def _closed_history(result: dict[str, Any], symbol: str, now: datetime) -> tuple[pd.Series[float], pd.Series[float]]:
    try:
        timestamps = result["timestamp"]
        raw = result["indicators"]["quote"][0]["close"]
        adjusted = result["indicators"]["adjclose"][0]["adjclose"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"{symbol}: raw and adjusted daily price arrays are required") from exc
    if not all(isinstance(values, list) for values in (timestamps, raw, adjusted)):
        raise ValueError(f"{symbol}: invalid daily price arrays")
    if not len(timestamps) == len(raw) == len(adjusted):
        raise ValueError(f"{symbol}: timestamp and price lengths differ")
    today = now.astimezone(NEW_YORK).date()
    rows: list[tuple[date, float, float]] = []
    for stamp, raw_price, adjusted_price in zip(timestamps, raw, adjusted, strict=True):
        timestamp = _positive_number(stamp, f"{symbol} daily timestamp")
        try:
            observed = datetime.fromtimestamp(timestamp, tz=UTC)
        except (OverflowError, OSError, ValueError) as exc:
            raise ValueError(f"{symbol}: invalid daily timestamp") from exc
        day = observed.astimezone(NEW_YORK).date()
        if observed > now:
            raise ValueError(f"{symbol}: future daily timestamp")
        # Conservative even after today's close: today's bars are never signals.
        if day >= today:
            continue
        rows.append((day, _positive_number(raw_price, f"{symbol} raw price"),
                     _positive_number(adjusted_price, f"{symbol} adjusted price")))
    if not rows or len({row[0] for row in rows}) != len(rows):
        raise ValueError(f"{symbol}: empty history or duplicate daily sessions")
    dates = pd.DatetimeIndex([row[0] for row in rows])
    raw_series = pd.Series([row[1] for row in rows], index=dates, name=symbol).sort_index()
    adjusted_series = pd.Series([row[2] for row in rows], index=dates, name=symbol).sort_index()
    if today - adjusted_series.index[-1].date() > timedelta(days=4):
        raise ValueError(f"{symbol}: completed daily price history is stale")
    return adjusted_series, raw_series


def load_demo_market(http: HttpGet, universe: list[str], cache_dir: Path, now: datetime) -> DemoMarket:
    """Fetch each symbol once and return 253+ aligned, completed daily sessions.

    ``prices_usd`` are raw closes on the shared signal date. For bounded order
    reference estimates use ``quote_prices_usd`` and inspect ``quote_times_utc``.
    Quotes may be at most 60 seconds after collection-start ``now`` to tolerate
    acquisition latency and clock skew. Callers must recheck against wall clock
    immediately before any order submission.
    No input from this function authorizes or submits an order.
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must carry an explicit timezone")
    now = now.astimezone(UTC)
    if not universe or len(set(universe)) != len(universe):
        raise ValueError("universe must contain unique symbols")
    if any(not isinstance(symbol, str) or not re.fullmatch(r"[A-Z][A-Z0-9.\-]*", symbol) for symbol in universe):
        raise ValueError("universe must contain explicit uppercase US symbols")

    adjusted: dict[str, pd.Series[float]] = {}
    raw_closes: dict[str, pd.Series[float]] = {}
    quotes: dict[str, float] = {}
    timestamps: dict[str, datetime] = {}
    snapshots: dict[str, dict[str, Any]] = {}
    for symbol in [*universe, FX_SYMBOL]:
        result = _fetch(http, symbol, now)
        quote, timestamp = _quote(result, symbol, now)
        snapshots[symbol] = result
        timestamps[symbol] = timestamp
        if symbol == FX_SYMBOL:
            fx = 1.0 / quote
        else:
            quotes[symbol] = quote
            adjusted[symbol], raw_closes[symbol] = _closed_history(result, symbol, now)

    history = pd.DataFrame(adjusted).sort_index().dropna(how="any")
    if len(history) < MIN_HISTORY_ROWS:
        raise ValueError("need at least 253 aligned completed daily sessions; refusing a false all-cash signal")
    signal_date = history.index[-1].date()
    prices = {symbol: float(raw_closes[symbol].loc[history.index[-1]]) for symbol in universe}

    _write_snapshot(cache_dir, now, snapshots)
    return DemoMarket(history, prices, fx, signal_date, timestamps, quotes)


def _write_snapshot(cache_dir: Path, now: datetime, snapshots: dict[str, dict[str, Any]]) -> None:
    directory = cache_dir / "demo_market"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{now.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex[:8]}.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump({"retrieved_at_utc": now.isoformat(), "results": snapshots}, handle, allow_nan=False)


@dataclass(frozen=True)
class DemoQuotes:
    prices_usd: dict[str, float]
    fx_usd_to_eur: float
    quote_times_utc: dict[str, datetime]


def load_demo_quotes(http: HttpGet, symbols: list[str], cache_dir: Path, now: datetime) -> DemoQuotes:
    """Fetch only current USD quotes plus EURUSD for AI-signal orders.

    No price history is required, so recent listings are allowed. The same identity, currency,
    positivity and four-day staleness checks as ``load_demo_market`` apply. Callers must
    recheck quote ages against the wall clock immediately before any submission.
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must carry an explicit timezone")
    now = now.astimezone(UTC)
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("symbols must contain unique entries")
    if any(not isinstance(symbol, str) or not re.fullmatch(r"[A-Z][A-Z0-9.\-]*", symbol) for symbol in symbols):
        raise ValueError("symbols must be explicit uppercase US symbols")
    prices: dict[str, float] = {}
    timestamps: dict[str, datetime] = {}
    snapshots: dict[str, dict[str, Any]] = {}
    fx = 0.0
    for symbol in [*symbols, FX_SYMBOL]:
        result = _fetch(http, symbol, now)
        quote, timestamp = _quote(result, symbol, now)
        snapshots[symbol] = result
        timestamps[symbol] = timestamp
        if symbol == FX_SYMBOL:
            fx = 1.0 / quote
        else:
            prices[symbol] = quote
    _write_snapshot(cache_dir, now, snapshots)
    return DemoQuotes(prices, fx, timestamps)
