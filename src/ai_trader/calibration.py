"""Score explicitly defined binary forecasts against supplied, dated outcomes.

Decision IDs alone are not forecasts. This module neither invents probabilities
nor infers outcomes from price data; callers must supply an event and its result.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from math import isfinite


@dataclass(frozen=True)
class BinaryForecast:
    forecast_id: str
    event: str
    probability: float
    resolution_date: date

    def __post_init__(self) -> None:
        if not isinstance(self.forecast_id, str) or not self.forecast_id.strip():
            raise ValueError("forecast_id must be a non-empty string")
        if not isinstance(self.event, str) or not self.event.strip():
            raise ValueError("event must explicitly define the binary proposition")
        if (
            isinstance(self.probability, bool)
            or not isinstance(self.probability, (int, float))
            or not isfinite(self.probability)
            or not 0 <= self.probability <= 1
        ):
            raise ValueError("probability must be finite and between zero and one")
        if type(self.resolution_date) is not date:
            raise ValueError("resolution_date must be a date")


@dataclass(frozen=True)
class ForecastScore:
    forecast_id: str
    probability: float
    outcome: bool
    brier: float


@dataclass(frozen=True)
class CalibrationSummary:
    total: int
    pending: int
    unresolved: int
    scores: tuple[ForecastScore, ...]

    @property
    def mean_brier(self) -> float | None:
        """Mean squared probability error; zero is best. None means no evidence."""
        return sum(score.brier for score in self.scores) / len(self.scores) if self.scores else None


def score_forecasts(
    forecasts: Iterable[BinaryForecast],
    outcomes: Mapping[str, bool],
    as_of: date,
) -> CalibrationSummary:
    """Score only forecasts whose resolution day has fully passed.

    Outcomes are authoritative user/data-pipeline inputs, not model judgements.
    Unknown outcome IDs are rejected to catch mismatched or stale files.
    """
    items = tuple(forecasts)
    ids = [forecast.forecast_id for forecast in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate forecast IDs")
    if set(outcomes) - set(ids):
        raise ValueError("outcomes reference unknown forecast IDs")
    if any(type(value) is not bool for value in outcomes.values()):
        raise ValueError("outcomes must contain explicit booleans")

    pending = unresolved = 0
    scores: list[ForecastScore] = []
    for forecast in items:
        if forecast.resolution_date >= as_of:
            pending += 1
        elif forecast.forecast_id not in outcomes:
            unresolved += 1
        else:
            outcome = outcomes[forecast.forecast_id]
            error = (forecast.probability - int(outcome)) ** 2
            scores.append(ForecastScore(forecast.forecast_id, forecast.probability, outcome, error))
    return CalibrationSummary(len(items), pending, unresolved, tuple(scores))
