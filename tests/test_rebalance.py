from pathlib import Path

import pytest

from ai_trader.risk import RiskLimits, load_limits
from ai_trader.rebalance import plan_rebalance, resolve_api_tickers

LIMITS = load_limits(Path(__file__).resolve().parents[1] / "config" / "risk_limits.yaml")
TICKERS = {"SPY": "SPY_US_EQ", "TLT": "TLT_US_EQ"}


def test_resolve_api_tickers_maps_unique_matches() -> None:
    instruments = [
        {"ticker": "SPY_US_EQ", "shortName": "SPY"},
        {"ticker": "TLT_US_EQ", "shortName": "TLT"},
        {"ticker": "SPYD_US_EQ", "shortName": "SPYD"},
    ]
    assert resolve_api_tickers(["SPY", "TLT"], instruments) == TICKERS


def test_resolve_api_tickers_refuses_missing_or_ambiguous() -> None:
    instruments = [
        {"ticker": "ABC_US_EQ"},
        {"ticker": "ABC_DE_EQ"},
    ]
    with pytest.raises(ValueError, match="found 2"):
        resolve_api_tickers(["ABC"], instruments)
    with pytest.raises(ValueError, match="found 0"):
        resolve_api_tickers(["XYZ"], instruments)


def test_buys_targets_and_sells_unwanted_holdings() -> None:
    relaxed = RiskLimits(**{**LIMITS.__dict__, "max_position_weight": 1.0, "max_order_eur": 1000.0})
    plan = plan_rebalance(
        targets={"SPY": 0.5},
        prices_usd={"SPY": 100.0, "TLT": 50.0},
        fx_usd_to_account=1.0,
        holdings_qty={"TLT": 2.0},
        portfolio_value=1000.0,
        api_tickers=TICKERS,
        limits=relaxed,
    )
    by_symbol = {o.symbol: o for o in plan.orders}
    assert by_symbol["SPY"].quantity == pytest.approx(5.0)
    assert by_symbol["TLT"].quantity == pytest.approx(-2.0)
    assert plan.violations == ()


def test_fx_conversion_changes_share_counts() -> None:
    plan = plan_rebalance(
        targets={"SPY": 0.5},
        prices_usd={"SPY": 100.0},
        fx_usd_to_account=0.5,  # account in a currency worth half a USD
        holdings_qty={},
        portfolio_value=1000.0,
        api_tickers={"SPY": "SPY_US_EQ"},
        limits=LIMITS,
    )
    assert plan.orders[0].quantity == pytest.approx(10.0)


def test_tiny_trades_are_skipped() -> None:
    plan = plan_rebalance(
        targets={"SPY": 0.5},
        prices_usd={"SPY": 100.0},
        fx_usd_to_account=1.0,
        holdings_qty={"SPY": 4.9999},
        portfolio_value=1000.0,
        api_tickers={"SPY": "SPY_US_EQ"},
        limits=LIMITS,
    )
    assert plan.orders == ()


def test_oversize_order_is_reported_not_resized() -> None:
    tight = RiskLimits(**{**LIMITS.__dict__, "max_order_eur": 20.0, "max_position_weight": 1.0})
    plan = plan_rebalance(
        targets={"SPY": 0.5},
        prices_usd={"SPY": 100.0},
        fx_usd_to_account=1.0,
        holdings_qty={},
        portfolio_value=1000.0,
        api_tickers={"SPY": "SPY_US_EQ"},
        limits=tight,
    )
    assert plan.orders[0].quantity == pytest.approx(5.0)  # not resized
    assert any("exceeds" in v for v in plan.violations)


def test_weight_cap_violation_is_reported() -> None:
    plan = plan_rebalance(
        targets={"SPY": 0.9},
        prices_usd={"SPY": 100.0},
        fx_usd_to_account=1.0,
        holdings_qty={},
        portfolio_value=1000.0,
        api_tickers={"SPY": "SPY_US_EQ"},
        limits=LIMITS,
    )
    assert any("exceeds cap" in v for v in plan.violations)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"portfolio_value": 0.0},
        {"fx_usd_to_account": 0.0},
    ],
)
def test_planner_rejects_bad_inputs(kwargs: dict[str, float]) -> None:
    base: dict[str, object] = {
        "targets": {"SPY": 0.5},
        "prices_usd": {"SPY": 100.0},
        "fx_usd_to_account": 1.0,
        "holdings_qty": {},
        "portfolio_value": 1000.0,
        "api_tickers": {"SPY": "SPY_US_EQ"},
        "limits": LIMITS,
    }
    with pytest.raises(ValueError):
        plan_rebalance(**{**base, **kwargs})  # type: ignore[arg-type]


def test_missing_price_is_refused() -> None:
    with pytest.raises(ValueError, match="missing"):
        plan_rebalance(
            targets={"SPY": 0.5},
            prices_usd={},
            fx_usd_to_account=1.0,
            holdings_qty={},
            portfolio_value=1000.0,
            api_tickers={"SPY": "SPY_US_EQ"},
            limits=LIMITS,
        )


def test_usd_order_cap_is_measured_in_eur() -> None:
    plan = plan_rebalance({"SPY": 0.15}, {"SPY": 100.0}, 1.0, {}, 400.0,
                          {"SPY": "SPY_US_EQ"}, LIMITS, fx_account_to_eur=0.8)
    assert plan.orders[0].value_account == pytest.approx(60.0)
    assert plan.violations == ()
    plan = plan_rebalance({"SPY": 0.15}, {"SPY": 100.0}, 1.0, {}, 300.0,
                          {"SPY": "SPY_US_EQ"}, LIMITS, fx_account_to_eur=1.2)
    assert any("exceeds" in message for message in plan.violations)


def test_sell_rounding_never_sells_more_than_owned() -> None:
    owned = 0.123456
    plan = plan_rebalance({}, {"SPY": 100.0}, 1.0, {"SPY": owned}, 100.0,
                          {"SPY": "SPY_US_EQ"}, LIMITS)
    assert plan.orders[0].quantity == pytest.approx(-0.1234)
    assert abs(plan.orders[0].quantity) <= owned


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", ["targets", "prices_usd", "holdings_qty", "portfolio_value",
                                    "fx_usd_to_account", "fx_account_to_eur", "min_value"])
def test_planner_rejects_nonfinite_values(field: str, value: float) -> None:
    inputs: dict[str, object] = {
        "targets": {"SPY": 0.15}, "prices_usd": {"SPY": 100.0},
        "holdings_qty": {}, "portfolio_value": 100.0, "fx_usd_to_account": 1.0,
        "api_tickers": {"SPY": "SPY_US_EQ"}, "limits": LIMITS,
    }
    inputs[field] = {"SPY": value} if field in ("targets", "prices_usd", "holdings_qty") else value
    with pytest.raises(ValueError):
        plan_rebalance(**inputs)  # type: ignore[arg-type]


def test_currency_mismatch_and_missing_currency_are_refused() -> None:
    for instrument in ({"ticker": "SPY_US_EQ", "currencyCode": "EUR"},
                       {"ticker": "SPY_US_EQ"}):
        with pytest.raises(ValueError, match="currency"):
            resolve_api_tickers(["SPY"], [instrument], expected_currency="USD")
    assert resolve_api_tickers(["SPY"], [{"ticker": "SPY_US_EQ", "currencyCode": "USD"}],
                               expected_currency="USD") == {"SPY": "SPY_US_EQ"}
