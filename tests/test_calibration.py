from datetime import date
from pathlib import Path
import importlib.util

import pytest

from ai_trader.calibration import BinaryForecast, score_forecasts
from ai_trader.decision_log import log_decision


def forecast(identifier: str = "f1", probability: float = .7, day: date = date(2026, 1, 10)) -> BinaryForecast:
    return BinaryForecast(identifier, "SPY adjusted close is higher on the resolution day than at issue", probability, day)


def test_brier_scores_verified_outcomes_only_after_resolution_day() -> None:
    forecasts = [forecast(), forecast("f2", .2), forecast("today", .9, date(2026, 1, 11))]
    result = score_forecasts(forecasts, {"f1": True, "f2": False, "today": True}, date(2026, 1, 11))
    assert result.pending == 1
    assert result.unresolved == 0
    assert len(result.scores) == 2
    assert result.mean_brier == pytest.approx((.09 + .04) / 2)


def test_missing_outcome_is_unresolved_not_assumed_a_miss() -> None:
    result = score_forecasts([forecast()], {}, date(2026, 1, 11))
    assert result.unresolved == 1
    assert result.scores == ()
    assert result.mean_brier is None


@pytest.mark.parametrize("probability", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_invalid_probability_rejected(probability: float) -> None:
    with pytest.raises(ValueError):
        forecast(probability=probability)


def test_duplicate_ids_and_non_boolean_or_unknown_outcomes_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        score_forecasts([forecast(), forecast()], {}, date(2026, 1, 11))
    with pytest.raises(ValueError, match="booleans"):
        score_forecasts([forecast()], {"f1": 1}, date(2026, 1, 11))  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="unknown"):
        score_forecasts([forecast()], {"other": True}, date(2026, 1, 11))


def load_dashboard_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_dashboard.py"
    spec = importlib.util.spec_from_file_location("build_dashboard", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dashboard_does_not_treat_plain_decisions_as_forecasts() -> None:
    dashboard = load_dashboard_module()
    result = dashboard.calibration_from_records([
        {"forecast_id": "decision-only", "probability": .6, "resolution_date": "2026-01-10"},
    ], {}, date(2026, 1, 11))
    assert result.total == 0


def test_logged_explicit_forecast_is_scoreable_end_to_end(tmp_path: Path) -> None:
    dashboard = load_dashboard_module()
    path = tmp_path / "decisions.jsonl"
    record = log_decision(
        path, action="hold", mode="paper", targets={}, violations=[], rationale="test only",
        event="SPY adjusted close on 2026-01-10 exceeds its issue-day close", probability=.7,
        resolution_date=date(2026, 1, 10),
    )
    loaded = dashboard.load_decisions(path)
    result = dashboard.calibration_from_records(loaded, {record["forecast_id"]: True}, date(2026, 1, 11))
    assert result.total == 1
    assert result.mean_brier == pytest.approx(.09)


@pytest.mark.parametrize("extra", [
    {"event": ""}, {"event": "explicit proposition"},
    {"event": "explicit proposition", "probability": .5},
    {"event": "explicit proposition", "resolution_date": date(2026, 1, 10)},
])
def test_incomplete_explicit_forecasts_are_rejected(tmp_path: Path, extra: dict) -> None:
    with pytest.raises(ValueError):
        log_decision(tmp_path / "d.jsonl", action="hold", mode="paper", targets={}, violations=[], rationale="test", **extra)


def test_public_dashboard_omits_private_details_and_private_output_escapes_html(tmp_path: Path) -> None:
    dashboard = load_dashboard_module()
    records = [{"forecast_id": "PRIVATE-ID", "rationale": "<script>secret()</script>", "targets": {"PRIVATE": .5}, "action": "dry-run"}]
    summary = score_forecasts([], {}, date(2026, 1, 11))
    public = dashboard.render_dashboard(tmp_path, records, summary, private=False)
    assert "PRIVATE" not in public
    assert "secret()" not in public
    private = dashboard.render_dashboard(tmp_path, records, summary, private=True)
    assert "&lt;script&gt;secret()&lt;/script&gt;" in private
    assert "<script>" not in private
