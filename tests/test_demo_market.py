from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from ai_trader.demo_market import load_demo_market


NOW = datetime(2026, 10, 27, 16, tzinfo=UTC)


def payload(symbol: str, days: int = 300) -> dict[str, Any]:
    dates = pd.bdate_range(end=NOW.astimezone(ZoneInfo("America/New_York")).date(), periods=days)
    stamps = [int(day.to_pydatetime().replace(hour=9, minute=30, tzinfo=ZoneInfo("America/New_York")).timestamp()) for day in dates]
    raw = [float(100 + index) for index in range(days)]
    return {"chart": {"error": None, "result": [{
        "meta": {"symbol": symbol, "currency": "USD", "regularMarketPrice": 999.0 if symbol != "EURUSD=X" else 1.25,
                 "regularMarketTime": int((NOW - timedelta(minutes=1)).timestamp())},
        "timestamp": stamps,
        "indicators": {"quote": [{"close": raw}], "adjclose": [{"adjclose": [value / 2 for value in raw]}]},
    }]}}


class FakeResponse:
    def __init__(self, body: dict[str, Any]) -> None:
        self.body = body

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return deepcopy(self.body)


class FakeHttp:
    def __init__(self) -> None:
        self.payloads = {symbol: payload(symbol) for symbol in ["SPY", "TLT", "EURUSD=X"]}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, *, params: dict[str, Any], headers: dict[str, str], timeout: float) -> FakeResponse:
        symbol = url.rsplit("/", 1)[-1]
        self.calls.append((symbol, params))
        return FakeResponse(self.payloads[symbol])


def meta(http: FakeHttp, symbol: str = "SPY") -> dict[str, Any]:
    return http.payloads[symbol]["chart"]["result"][0]["meta"]


def test_market_separates_adjusted_signal_raw_close_and_current_quote(tmp_path: Path) -> None:
    http = FakeHttp()
    market = load_demo_market(http, ["SPY", "TLT"], tmp_path, NOW)
    assert market.signal_date.isoformat() == "2026-10-26"
    assert len(market.history) == 299
    assert market.history.loc[pd.Timestamp(market.signal_date), "SPY"] == 199.0
    assert market.prices_usd["SPY"] == 398.0
    assert market.quote_prices_usd["SPY"] == 999.0
    assert market.fx_usd_to_eur == .8
    assert market.quote_times_utc["SPY"] == NOW - timedelta(minutes=1)
    assert [symbol for symbol, _ in http.calls] == ["SPY", "TLT", "EURUSD=X"]
    assert len(list((tmp_path / "demo_market").glob("*.json"))) == 1
    assert http.calls[0][1]["period1"] < int((NOW - timedelta(days=799)).timestamp())


@pytest.mark.parametrize("field,value", [("symbol", "SPYG"), ("currency", "GBp"), ("currency", "EUR"), ("regularMarketPrice", float("nan")), ("regularMarketPrice", 0), ("regularMarketTime", float("inf"))])
def test_invalid_quote_identity_currency_and_numbers_refused(tmp_path: Path, field: str, value: Any) -> None:
    http = FakeHttp()
    meta(http)[field] = value
    with pytest.raises(ValueError):
        load_demo_market(http, ["SPY"], tmp_path, NOW)


@pytest.mark.parametrize("delta", [timedelta(days=4, seconds=1), timedelta(seconds=-120)])
def test_stale_or_future_quotes_refused(tmp_path: Path, delta: timedelta) -> None:
    http = FakeHttp()
    meta(http)["regularMarketTime"] = int((NOW - delta).timestamp())
    with pytest.raises(ValueError, match="quote"):
        load_demo_market(http, ["SPY"], tmp_path, NOW)


def test_quote_within_acquisition_clock_tolerance_accepted(tmp_path: Path) -> None:
    http = FakeHttp()
    meta(http)["regularMarketTime"] = int((NOW + timedelta(seconds=30)).timestamp())
    market = load_demo_market(http, ["SPY"], tmp_path, NOW)
    assert market.quote_times_utc["SPY"] == NOW + timedelta(seconds=30)


def test_four_calendar_day_quote_boundary_is_allowed(tmp_path: Path) -> None:
    http = FakeHttp()
    for symbol in http.payloads:
        meta(http, symbol)["regularMarketTime"] = int((NOW - timedelta(days=4)).timestamp())
    market = load_demo_market(http, ["SPY"], tmp_path, NOW)
    assert market.quote_times_utc["SPY"] == NOW - timedelta(days=4)


def test_naive_now_insufficient_history_and_duplicate_symbols_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="timezone"):
        load_demo_market(FakeHttp(), ["SPY"], tmp_path, NOW.replace(tzinfo=None))
    http = FakeHttp()
    http.payloads["SPY"] = payload("SPY", days=253)
    with pytest.raises(ValueError, match="253"):
        load_demo_market(http, ["SPY"], tmp_path, NOW)
    with pytest.raises(ValueError, match="unique"):
        load_demo_market(FakeHttp(), ["SPY", "SPY"], tmp_path, NOW)


@pytest.mark.parametrize("kind,value", [("quote", None), ("quote", float("nan")), ("quote", -1), ("adjclose", 0)])
def test_invalid_completed_daily_bars_refused(tmp_path: Path, kind: str, value: Any) -> None:
    http = FakeHttp()
    bars = http.payloads["SPY"]["chart"]["result"][0]["indicators"][kind][0]
    bars["close" if kind == "quote" else "adjclose"][-2] = value
    with pytest.raises(ValueError, match="price"):
        load_demo_market(http, ["SPY"], tmp_path, NOW)


def test_open_today_bar_is_discarded_even_when_non_null(tmp_path: Path) -> None:
    http = FakeHttp()
    bars = http.payloads["SPY"]["chart"]["result"][0]["indicators"]
    bars["quote"][0]["close"][-1] = None
    bars["adjclose"][0]["adjclose"][-1] = None
    market = load_demo_market(http, ["SPY"], tmp_path, NOW)
    assert market.signal_date < NOW.date()


def test_fx_identity_currency_and_rate_checked(tmp_path: Path) -> None:
    http = FakeHttp()
    meta(http, "EURUSD=X")["currency"] = "EUR"
    with pytest.raises(ValueError, match="currency"):
        load_demo_market(http, ["SPY"], tmp_path, NOW)
