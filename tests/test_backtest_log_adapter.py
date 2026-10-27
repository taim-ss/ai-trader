import base64
import json
from datetime import date
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from ai_trader.backtest import CostModel, metrics, month_end_dates, run_backtest
from ai_trader.broker import TRADING212_DEMO_URL, TRADING212_LIVE_URL
from ai_trader.decision_log import log_decision
from ai_trader.paper_adapter import Trading212Paper, credentials_from_env
from ai_trader.strategy import StrategyConfig


def make_prices(days: int = 900, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=days)
    data = {}
    for name, drift in [("A", 0.0008), ("B", 0.0003), ("C", -0.0004)]:
        steps = rng.normal(drift, 0.01, days)
        data[name] = 100 * np.exp(np.cumsum(steps))
    return pd.DataFrame(data, index=idx)


CONFIG = StrategyConfig(top_n=2, single_name_cap=0.6)


def test_backtest_never_uses_future_prices() -> None:
    prices = make_prices()
    cut = prices.index[500]
    curve = run_backtest(prices, CONFIG, CostModel())

    tampered = prices.copy()
    tampered.loc[tampered.index > cut] *= 50.0
    curve_tampered = run_backtest(tampered, CONFIG, CostModel())

    before = curve.loc[:cut]
    after = curve_tampered.loc[:cut]
    pd.testing.assert_series_equal(before, after, check_names=False)


def test_costs_reduce_final_equity() -> None:
    prices = make_prices()
    free = run_backtest(prices, CONFIG, CostModel(slippage_bps=0, fx_bps=0))
    costly = run_backtest(prices, CONFIG, CostModel(slippage_bps=50, fx_bps=50))
    assert costly.iloc[-1] < free.iloc[-1]


def test_backtest_rejects_nan_prices() -> None:
    prices = make_prices(days=50)
    prices.iloc[10, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        run_backtest(prices, CONFIG, CostModel())


def test_holdings_drift_without_free_daily_rebalancing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ai_trader.backtest.target_weights", lambda *args: {"A": 0.5})
    prices = pd.DataFrame({"A": [100, 100, 100, 200, 100]},
                          index=pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01",
                                                "2024-02-02", "2024-02-05"]))
    equity = run_backtest(prices, CONFIG, CostModel(slippage_bps=0, fx_bps=0))
    assert equity.iloc[-2] == pytest.approx(1500)
    assert equity.iloc[-1] == pytest.approx(1000)


def test_signal_executes_after_next_session_return(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[pd.Timestamp] = []

    def signal(history: pd.DataFrame, *args: object) -> dict[str, float]:
        observed.append(history.index[-1])
        return {"A": 1.0}

    monkeypatch.setattr("ai_trader.backtest.target_weights", signal)
    prices = pd.DataFrame({"A": [100, 100, 200, 400]},
                          index=pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02"]))
    equity = run_backtest(prices, CONFIG, CostModel(slippage_bps=0, fx_bps=0))
    assert observed == [pd.Timestamp("2024-01-31")]
    assert equity.iloc[2] == pytest.approx(1000)  # no exposure to pre-fill gap
    assert equity.iloc[3] == pytest.approx(2000)


def test_costs_charged_on_actual_traded_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ai_trader.backtest.target_weights", lambda *args: {"A": 1.0})
    prices = pd.DataFrame({"A": [100, 100, 100]},
                          index=pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01"]))
    equity = run_backtest(prices, CONFIG, CostModel(slippage_bps=100, fx_bps=0))
    assert equity.iloc[-1] == pytest.approx(1000 / 1.01)
    assert equity.attrs["turnover"] == pytest.approx((1000 / 1.01) / 1000)


def test_rebalance_turnover_uses_drifted_holdings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ai_trader.backtest.target_weights", lambda *args: {"A": 0.5})
    prices = pd.DataFrame({"A": [100, 100, 100, 200, 200]},
                          index=pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01",
                                                "2024-02-29", "2024-03-01"]))
    equity = run_backtest(prices, CONFIG, CostModel(slippage_bps=0, fx_bps=0))
    assert equity.attrs["turnover"] == pytest.approx(0.5 + 250 / 1500)


def test_month_end_dates_pick_last_trading_day_per_month() -> None:
    idx = pd.bdate_range("2024-01-01", "2024-03-31")
    ends = month_end_dates(idx)
    assert [d.strftime("%Y-%m-%d") for d in ends] == ["2024-01-31", "2024-02-29", "2024-03-29"]


def test_metrics_on_known_curve() -> None:
    idx = pd.bdate_range("2024-01-01", periods=253)
    equity = pd.Series(np.linspace(100.0, 200.0, 253), index=idx)
    result = metrics(equity)
    assert result["cagr"] > 0
    assert result["max_drawdown"] == pytest.approx(0.0, abs=1e-12)
    assert result["sharpe"] > 0


def test_metrics_drawdown_is_negative_fraction() -> None:
    idx = pd.bdate_range("2024-01-01", periods=4)
    equity = pd.Series([100.0, 120.0, 90.0, 110.0], index=idx)
    assert metrics(equity)["max_drawdown"] == pytest.approx(90 / 120 - 1)


def test_decision_log_appends_json_lines(tmp_path: Path) -> None:
    target = tmp_path / "logs" / "decisions.jsonl"
    first = log_decision(
        target,
        action="rebalance",
        mode="paper",
        targets={"A": 0.5},
        violations=[],
        rationale="momentum top-1",
        resolution_date=date(2026, 1, 31),
        probability=0.6,
    )
    log_decision(target, action="hold", mode="paper", targets={}, violations=["x"], rationale="r")
    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["forecast_id"] == first["forecast_id"]
    assert json.loads(lines[0])["resolution_date"] == "2026-01-31"


def test_decision_log_rejects_bad_probability(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        log_decision(
            tmp_path / "d.jsonl",
            action="x",
            mode="paper",
            targets={},
            violations=[],
            rationale="r",
            probability=1.5,
        )


def test_adapter_refuses_live_url() -> None:
    with pytest.raises(ValueError, match="demo"):
        Trading212Paper("k", "s", base_url=TRADING212_LIVE_URL)


def test_adapter_sends_basic_auth_and_reads_summary() -> None:
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["req"] = request
        return httpx.Response(200, json={"cash": 1.0})

    client = Trading212Paper("key-x", "secret-y", transport=httpx.MockTransport(handler))
    assert client.account_summary() == {"cash": 1.0}
    request = seen["req"]
    expected = base64.b64encode(b"key-x:secret-y").decode()
    assert request.headers["Authorization"] == f"Basic {expected}"
    assert str(request.url) == f"{TRADING212_DEMO_URL}/equity/account/summary"


def test_market_order_payload_and_validation() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"id": 1, "status": "NEW"})

    client = Trading212Paper("k", "s", transport=httpx.MockTransport(handler))
    result = client.place_market_order("AAPL_US_EQ", -0.5)
    assert result["id"] == 1
    assert bodies[0] == {"ticker": "AAPL_US_EQ", "quantity": -0.5, "extendedHours": False}
    with pytest.raises(ValueError):
        client.place_market_order("aapl_us_eq", 1.0)
    with pytest.raises(ValueError):
        client.place_market_order("AAPL_US_EQ", 0)


def test_http_errors_are_raised_not_swallowed() -> None:
    client = Trading212Paper(
        "k", "s", transport=httpx.MockTransport(lambda r: httpx.Response(401))
    )
    with pytest.raises(httpx.HTTPStatusError):
        client.account_summary()


def test_credentials_must_come_from_environment() -> None:
    with pytest.raises(RuntimeError, match="environment"):
        credentials_from_env({})
    assert credentials_from_env({"TRADING212_DEMO_API_KEY": "a", "TRADING212_DEMO_API_SECRET": "b"}) == (
        "a",
        "b",
    )
