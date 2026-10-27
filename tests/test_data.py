from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from ai_trader.data import load_prices, parse_chart


def chart_payload(symbol: str, closes: list[float], start_ts: int = 1704067200) -> dict[str, Any]:
    timestamps = [start_ts + i * 86400 for i in range(len(closes))]
    return {
        "chart": {
            "result": [
                {
                    "meta": {"symbol": symbol},
                    "timestamp": timestamps,
                    "indicators": {"adjclose": [{"adjclose": closes}]},
                }
            ],
            "error": None,
        }
    }


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeHttp:
    def __init__(self, payloads: dict[str, dict[str, Any]]) -> None:
        self.payloads = payloads
        self.calls: list[str] = []

    def get(self, url: str, *, params: dict[str, Any], headers: dict[str, str], timeout: float) -> FakeResponse:
        symbol = url.rsplit("/", 1)[-1]
        self.calls.append(symbol)
        return FakeResponse(self.payloads[symbol])


def test_parse_chart_returns_dated_series() -> None:
    series = parse_chart(chart_payload("SPY", [100.0, 101.5, 102.0]))
    assert series.name == "SPY"
    assert list(series) == [100.0, 101.5, 102.0]
    assert series.index.is_monotonic_increasing


def test_parse_chart_rejects_error_and_empty_payloads() -> None:
    with pytest.raises(ValueError, match="error"):
        parse_chart({"chart": {"result": None, "error": {"code": "Not Found"}}})
    with pytest.raises(ValueError, match="no chart result"):
        parse_chart({"chart": {"result": [], "error": None}})


def test_parse_chart_refuses_missing_prices() -> None:
    with pytest.raises(ValueError, match="missing"):
        parse_chart(chart_payload("SPY", [100.0, float("nan")]))


def test_load_prices_caches_and_aligns(tmp_path: Path) -> None:
    http = FakeHttp(
        {
            "SPY": chart_payload("SPY", [1.0, 2.0, 3.0]),
            "TLT": chart_payload("TLT", [5.0, 6.0, 7.0]),
        }
    )
    start, end = date(2024, 1, 1), date(2024, 1, 10)
    table = load_prices(http, ["SPY", "TLT"], start, end, tmp_path)
    assert list(table.columns) == ["SPY", "TLT"]
    assert len(table) == 3
    assert http.calls == ["SPY", "TLT"]

    again = load_prices(http, ["SPY", "TLT"], start, end, tmp_path)
    pd.testing.assert_frame_equal(table, again)
    assert http.calls == ["SPY", "TLT"]  # served from cache, no new requests


def test_parse_chart_drops_trailing_open_bar() -> None:
    series = parse_chart(chart_payload("SPY", [100.0, 101.0, None]))
    assert list(series) == [100.0, 101.0]


def test_parse_chart_still_refuses_interior_gaps() -> None:
    with pytest.raises(ValueError, match="inside"):
        parse_chart(chart_payload("SPY", [100.0, None, 102.0]))


def test_forward_fill_is_opt_in_for_fx() -> None:
    series = parse_chart(chart_payload("EURUSD=X", [1.1, None, 1.2]), forward_fill_gaps=True)
    assert list(series) == [1.1, 1.1, 1.2]
