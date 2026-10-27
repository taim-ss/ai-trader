"""Daily adjusted-close prices from Yahoo's public chart endpoint, cached as CSV.

Uses the operating system trust store (via truststore) for TLS. Verification stays on.
Data licence and terms: Yahoo data is for personal research; not for redistribution.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
USER_AGENT = "Mozilla/5.0 (ai-trader research; contact owner)"


class HttpGet(Protocol):
    def get(self, url: str, *, params: dict[str, Any], headers: dict[str, str], timeout: float) -> Any: ...


def _to_epoch(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp())


def parse_chart(payload: dict[str, Any], forward_fill_gaps: bool = False) -> pd.Series:
    """Turn a Yahoo chart JSON payload into a date-indexed adjusted-close Series."""
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise ValueError(f"Yahoo returned an error: {chart['error']}")
    results = chart.get("result") or []
    if not results:
        raise ValueError("no chart result for this symbol")
    result = results[0]
    timestamps = list(result.get("timestamp") or [])
    adjclose = list(result["indicators"]["adjclose"][0]["adjclose"])
    if len(timestamps) != len(adjclose):
        raise ValueError("timestamp and price lengths differ")
    # Yahoo leaves today's bar empty until the session closes; drop only trailing gaps.
    while adjclose and adjclose[-1] is None:
        adjclose.pop()
        timestamps.pop()
    dates = [datetime.fromtimestamp(ts, tz=UTC).date() for ts in timestamps]
    series = pd.Series(adjclose, index=pd.DatetimeIndex(dates), dtype=float, name=result["meta"]["symbol"])
    series = series[~series.index.duplicated(keep="last")].sort_index()
    if series.isna().any():
        if not forward_fill_gaps:
            raise ValueError("missing adjusted closes inside the series; refusing to fill them silently")
        # Only for FX rates, where closed-market days carry no quote (the last rate still holds).
        series = series.ffill()
    return series


def fetch_symbol(http: HttpGet, symbol: str, start: date, end: date, forward_fill_gaps: bool = False) -> pd.Series:
    params = {
        "period1": _to_epoch(start),
        "period2": _to_epoch(end),
        "interval": "1d",
        "events": "div,splits",
        "includeAdjustedClose": "true",
    }
    response = http.get(
        CHART_URL.format(symbol=symbol),
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=20.0,
    )
    response.raise_for_status()
    return parse_chart(response.json(), forward_fill_gaps=forward_fill_gaps)


def load_prices(
    http: HttpGet,
    symbols: Iterable[str],
    start: date,
    end: date,
    cache_dir: Path,
    forward_fill: frozenset[str] = frozenset(),
) -> pd.DataFrame:
    """Return an aligned price table. Cached per symbol; a cache file is never overwritten silently."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.Series] = {}
    for symbol in symbols:
        path = cache_dir / f"{symbol}_{start.isoformat()}_{end.isoformat()}.csv"
        if path.exists():
            series = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        else:
            series = fetch_symbol(http, symbol, start, end, symbol in forward_fill)
            series.to_csv(path, header=[symbol])
        frames[symbol] = series
    table = pd.DataFrame(frames).sort_index()
    table.index = pd.DatetimeIndex(table.index).as_unit("ns")
    return table.dropna(how="any")
