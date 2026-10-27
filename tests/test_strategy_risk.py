from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from ai_trader.risk import (
    RiskLimits,
    kill_switch_engaged,
    load_limits,
    validate_order,
    validate_targets,
)
from ai_trader.strategy import StrategyConfig, target_weights

LIMITS = RiskLimits(
    max_position_weight=0.15,
    max_daily_loss_eur=10.0,
    max_drawdown_kill_switch=0.20,
    max_order_eur=50.0,
    allow_leverage=False,
    allow_shorting=False,
    allow_options=False,
    loss_cooldown_trades=3,
    cooldown_hours=24,
)


def make_prices(days: int = 400) -> pd.DataFrame:
    idx = pd.bdate_range("2023-01-02", periods=days)
    up = pd.Series(range(days), index=idx, dtype=float) * 0.5 + 100
    down = pd.Series(range(days), index=idx, dtype=float) * -0.2 + 200
    flat = pd.Series([100.0] * days, index=idx)
    return pd.DataFrame({"UP": up, "DOWN": down, "FLAT": flat})


def test_momentum_picks_only_positive_scores() -> None:
    prices = make_prices()
    as_of = prices.index[-1].date()
    weights = target_weights(prices, as_of, StrategyConfig(top_n=3))
    assert set(weights) == {"UP"}
    assert weights["UP"] == pytest.approx(0.15)  # capped, rest stays in cash


def test_strategy_never_uses_future_prices() -> None:
    prices = make_prices()
    as_of = prices.index[200].date()
    before = target_weights(prices, as_of, StrategyConfig(top_n=2))

    tampered = prices.copy()
    tampered.loc[tampered.index > pd.Timestamp(as_of), "DOWN"] = 1_000_000.0
    tampered.loc[tampered.index > pd.Timestamp(as_of), "UP"] = 0.01
    after = target_weights(tampered, as_of, StrategyConfig(top_n=2))

    assert before == after


def test_core_weight_is_kept_and_satellite_budget_shrinks() -> None:
    prices = make_prices()
    config = StrategyConfig(top_n=1, single_name_cap=0.9, core_ticker="FLAT", core_weight=0.4)
    weights = target_weights(prices, prices.index[-1].date(), config)
    assert weights["FLAT"] == pytest.approx(0.4)
    assert weights["UP"] == pytest.approx(0.6)


def test_insufficient_history_means_all_cash() -> None:
    prices = make_prices(days=50)
    assert target_weights(prices, prices.index[-1].date(), StrategyConfig()) == {}


@pytest.mark.parametrize(
    "kwargs",
    [{"top_n": 0}, {"single_name_cap": 0}, {"core_weight": 1.0}, {"core_ticker": "X"}],
)
def test_strategy_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        StrategyConfig(**kwargs)  # type: ignore[arg-type]


def test_validate_targets_flags_cap_leverage_and_shorts() -> None:
    violations = validate_targets({"A": 0.3, "B": -0.1, "C": 0.9}, LIMITS)
    text = " | ".join(violations)
    assert "A: weight 30.0% exceeds cap 15.0%" in text
    assert "B: negative weight" in text
    assert "leverage is disabled" in text


def test_validate_targets_passes_clean_targets() -> None:
    assert validate_targets({"A": 0.15, "B": 0.15}, LIMITS) == []


def test_validate_order_rejects_oversize_and_nonpositive() -> None:
    assert validate_order(60, LIMITS)
    assert validate_order(0, LIMITS)
    assert validate_order(50, LIMITS) == []


def test_kill_switch_triggers_at_threshold() -> None:
    assert kill_switch_engaged(0.20, LIMITS)
    assert not kill_switch_engaged(0.19, LIMITS)
    with pytest.raises(ValueError):
        kill_switch_engaged(-0.1, LIMITS)


def test_load_limits_reads_repo_config() -> None:
    root = Path(__file__).resolve().parents[1]
    limits = load_limits(root / "config" / "risk_limits.yaml")
    assert limits.max_position_weight == 0.15
    assert limits.allow_leverage is False


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_risk_inputs_cannot_pass(value: float) -> None:
    assert validate_targets({"A": value}, LIMITS)
    assert validate_order(value, LIMITS)
    with pytest.raises(ValueError):
        kill_switch_engaged(value, LIMITS)


@pytest.mark.parametrize("change", [
    {"allow_shorting": "false"}, {"allow_leverage": 0}, {"allow_options": 1},
    {"max_order_eur": float("nan")}, {"max_daily_loss_eur": float("inf")},
    {"max_position_weight": 1.1}, {"max_drawdown_kill_switch": 0},
    {"max_order_eur": True}, {"loss_cooldown_trades": 2.5}, {"cooldown_hours": 0},
])
def test_risk_limits_reject_bad_runtime_types_and_ranges(change: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RiskLimits(**{**LIMITS.__dict__, **change})


def test_yaml_string_false_cannot_enable_shorting(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "config" / "risk_limits.yaml").read_text(encoding="utf-8")
    path = tmp_path / "limits.yaml"
    path.write_text(text.replace("allow_shorting: false", 'allow_shorting: "false"'), encoding="utf-8")
    with pytest.raises(ValueError, match="boolean"):
        load_limits(path)


def test_short_permission_does_not_bypass_position_or_gross_exposure_caps() -> None:
    limits = RiskLimits(**{**LIMITS.__dict__, "allow_shorting": True})
    assert any("cap" in message for message in validate_targets({"A": -0.2}, limits))
    weights = {f"A{i}": 0.15 if i % 2 else -0.15 for i in range(8)}
    assert any("leverage" in message for message in validate_targets(weights, limits))
