from datetime import date, datetime

import pandas as pd
import pytest

from ai_trader.arena import (
    Criteria,
    Observation,
    assert_no_future_data,
    bootstrap_mean_ci,
    completed_closes,
    entries_still_forward,
    horizon_result,
    percentile_rank,
    random_net_returns,
    rule_flags,
    scoreboard,
    validate_weights,
    verdict,
)

DAYS = pd.bdate_range("2026-01-05", periods=30)


def table() -> pd.DataFrame:
    steps = range(len(DAYS))
    return pd.DataFrame({
        "SPY": [100 * 1.001 ** i for i in steps],
        "UP": [100 * 1.01 ** i for i in steps],
        "FLAT": [50.0 for _ in steps],
        "DOWN": [100 * 0.99 ** i for i in steps],
    }, index=DAYS)


def test_horizon_result_math_and_costs() -> None:
    result = horizon_result(table(), DAYS[5].date(), {"UP": 0.15}, "SPY", 5)
    assert result is not None
    assert result.base_date == DAYS[5].date() and result.end_date == DAYS[10].date()
    gross = 0.15 * (1.01 ** 5 - 1)
    assert result.gross_return == pytest.approx(gross)
    assert result.net_return == pytest.approx(gross - 0.15 * 0.005)
    assert result.excess_matched == pytest.approx(gross - 0.15 * (1.001 ** 5 - 1))
    assert result.excess_raw == pytest.approx(result.net_return - ((1.001 ** 5 - 1) - 0.005))


def test_unfinished_horizon_is_never_scored_partially() -> None:
    assert horizon_result(table(), DAYS[27].date(), {"UP": 0.15}, "SPY", 5) is None


def test_weekend_round_date_uses_previous_close_and_later_prices_cannot_change_the_base() -> None:
    saturday = date(2026, 1, 10)
    first = horizon_result(table(), saturday, {"UP": 0.1}, "SPY", 5)
    assert first is not None and first.base_date == DAYS[4].date()
    changed = table()
    changed.loc[DAYS[20]:, "UP"] = 1.0  # later than the horizon end: must not matter
    second = horizon_result(changed, saturday, {"UP": 0.1}, "SPY", 5)
    assert second is not None and second.gross_return == first.gross_return


def test_cash_has_zero_gross_and_matched_excess_ignores_under_investment() -> None:
    cash = horizon_result(table(), DAYS[2].date(), {}, "SPY", 5)
    assert cash is not None and cash.net_return == 0 and cash.excess_matched == 0
    assert cash.excess_raw < 0  # raw comparison punishes cash when the index rises


def test_missing_symbol_or_bad_close_is_an_error_not_a_zero() -> None:
    with pytest.raises(ValueError, match="no price column"):
        horizon_result(table(), DAYS[2].date(), {"NOPE": 0.1}, "SPY", 5)
    broken = table()
    broken.loc[DAYS[8], "UP"] = float("nan")
    with pytest.raises(ValueError, match="missing"):
        horizon_result(broken, DAYS[3].date(), {"UP": 0.1}, "SPY", 5)


@pytest.mark.parametrize("weights", [{"UP": -0.1}, {"ZZZ": 0.1}, {"UP": float("nan")}, {"UP": True},
                                      {"UP": 0.6, "DOWN": 0.5}])
def test_invalid_weights_rejected(weights: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        validate_weights(weights, ["UP", "DOWN"])


def test_validate_drops_zero_weights_and_rule_flags_report_cap_and_count() -> None:
    clean = validate_weights({"UP": 0.2, "DOWN": 0.0}, ["UP", "DOWN"])
    assert clean == {"UP": 0.2}
    flags = rule_flags({"A": 0.2, "B": 0.1, "C": 0.1}, 0.15, 2)
    assert any("A" in f and "cap" in f for f in flags) and any("3 positions" in f for f in flags)
    assert rule_flags({"A": 0.15}, 0.15, 5) == []


def test_future_data_guard() -> None:
    assert_no_future_data(table().loc[: DAYS[10]], DAYS[10].date())
    with pytest.raises(ValueError, match="after the round date"):
        assert_no_future_data(table(), DAYS[10].date())


def utc(day: str, hour: int, minute: int = 0) -> datetime:
    return datetime.fromisoformat(f"{day}T{hour:02d}:{minute:02d}:00+00:00")


def test_unfinished_session_bar_is_dropped_but_a_settled_close_is_kept() -> None:
    frame = table().loc[: DAYS[10]]  # last bar is 2026-01-19
    during = completed_closes(frame, utc("2026-01-19", 14, 44))  # 09:44 in New York
    assert during.index.max() == DAYS[9]
    assert completed_closes(frame, utc("2026-01-19", 21, 45)).index.max() == DAYS[10]  # 16:45 in New York
    assert completed_closes(frame, utc("2026-01-20", 9, 0)).index.max() == DAYS[10]  # next morning
    with pytest.raises(ValueError):
        completed_closes(frame, datetime(2026, 1, 19, 14, 44))


def test_entries_close_when_the_next_session_opens_and_skip_weekends() -> None:
    friday = date(2026, 1, 9)
    assert entries_still_forward(friday, utc("2026-01-10", 12))  # Saturday
    assert entries_still_forward(friday, utc("2026-01-12", 14, 29))  # Monday 09:29 New York
    assert not entries_still_forward(friday, utc("2026-01-12", 14, 30))  # Monday 09:30 New York
    with pytest.raises(ValueError):
        entries_still_forward(friday, datetime(2026, 1, 10))


def test_random_portfolios_are_seeded_and_empty_until_resolved() -> None:
    kwargs = dict(count=50, picks=2, weight=0.15, seed=7)
    first = random_net_returns(table(), DAYS[3].date(), ["UP", "FLAT", "DOWN"], 5, **kwargs)
    again = random_net_returns(table(), DAYS[3].date(), ["UP", "FLAT", "DOWN"], 5, **kwargs)
    assert first == again and len(first) == 50
    assert random_net_returns(table(), DAYS[28].date(), ["UP", "FLAT", "DOWN"], 5, **kwargs) == []
    with pytest.raises(ValueError):
        random_net_returns(table(), DAYS[3].date(), ["UP"], 5, **kwargs)


def test_percentile_rank() -> None:
    assert percentile_rank(0.5, [0.1, 0.2, 0.3, 0.4]) == 1.0
    assert percentile_rank(0.25, [0.1, 0.2, 0.3, 0.4]) == 0.5
    assert percentile_rank(0.3, [0.3, 0.3]) == 0.5
    with pytest.raises(ValueError):
        percentile_rank(0.1, [])


def test_bootstrap_is_deterministic_and_needs_two_points() -> None:
    values = [0.02, 0.01, 0.03, -0.01, 0.02]
    first = bootstrap_mean_ci(values, 0.8, seed=1)
    assert first is not None and first == bootstrap_mean_ci(values, 0.8, seed=1)
    assert first[0] <= sum(values) / len(values) <= first[1]
    assert bootstrap_mean_ci([0.1], 0.8) is None
    with pytest.raises(ValueError):
        bootstrap_mean_ci(values, 1.0)


def obs(name: str, round_id: str, excess: float, pct: float | None = 0.8) -> Observation:
    return Observation(name, round_id, 5, excess, excess, excess, pct)


CRITERIA = Criteria(min_rounds=4, horizon=5, ci_level=0.8, min_random_percentile=0.6)


def test_verdict_needs_enough_rounds_then_passes_or_fails() -> None:
    early = scoreboard([obs("a", "r1", 0.02), obs("a", "r2", 0.01)], CRITERIA)
    assert verdict(early[0], CRITERIA)[0] == "NO EVIDENCE YET"
    good = scoreboard([obs("a", f"r{i}", 0.02 + i / 1000) for i in range(5)], CRITERIA)
    assert verdict(good[0], CRITERIA) == ("PASS", [])
    mixed = scoreboard([obs("b", f"r{i}", x) for i, x in enumerate([0.02, -0.03, 0.01, -0.02, 0.0])], CRITERIA)
    name, reasons = verdict(mixed[0], CRITERIA)
    assert name == "FAIL" and reasons
    lucky = scoreboard([obs("c", f"r{i}", 0.02 + i / 1000, 0.4) for i in range(5)], CRITERIA)
    assert "random picks" in " ".join(verdict(lucky[0], CRITERIA)[1])


def test_scoreboard_filters_to_the_criteria_horizon() -> None:
    twenty = Observation("a", "r1", 20, 0.5, 0.5, 0.5, 0.9)
    assert scoreboard([twenty], CRITERIA) == []
