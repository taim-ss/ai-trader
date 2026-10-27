"""User profile intake. Validates goals, horizon, risk, budget and reserve at the boundary."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal, cast

Horizon = Literal["1m", "3m", "1y", "5y+"]
RiskTolerance = Literal["low", "medium", "high"]
Mode = Literal["advise", "paper", "live-with-approval", "live-auto"]

ALLOWED_HORIZONS: frozenset[str] = frozenset({"1m", "3m", "1y", "5y+"})
ALLOWED_RISK: frozenset[str] = frozenset({"low", "medium", "high"})


@dataclass(frozen=True)
class UserProfile:
    goal: str
    horizon: Horizon
    risk_tolerance: RiskTolerance
    budget_eur: float
    reserve_months: float
    mode: Mode = "advise"


def parse_profile(raw: dict[str, object]) -> UserProfile:
    """Validate a raw mapping (from YAML/JSON) and return a typed profile.

    Raises ValueError with a readable message for any invalid field.
    """
    goal = raw.get("goal")
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("goal must be a non-empty string")

    horizon = raw.get("horizon")
    if horizon not in ALLOWED_HORIZONS:
        raise ValueError(f"horizon must be one of {sorted(ALLOWED_HORIZONS)}")

    risk = raw.get("risk_tolerance")
    if risk not in ALLOWED_RISK:
        raise ValueError(f"risk_tolerance must be one of {sorted(ALLOWED_RISK)}")

    budget = raw.get("budget_eur")
    if not isinstance(budget, (int, float)) or isinstance(budget, bool) or not isfinite(budget) or budget <= 0:
        raise ValueError("budget_eur must be a positive number")

    reserve = raw.get("reserve_months", 0)
    if not isinstance(reserve, (int, float)) or isinstance(reserve, bool) or not isfinite(reserve) or reserve < 0:
        raise ValueError("reserve_months must be zero or a positive number")

    mode = raw.get("mode", "advise")
    if mode not in ("advise", "paper", "live-with-approval", "live-auto"):
        raise ValueError("mode must be advise, paper, live-with-approval or live-auto")

    return UserProfile(
        goal=goal.strip(),
        horizon=cast(Horizon, horizon),
        risk_tolerance=cast(RiskTolerance, risk),
        budget_eur=float(budget),
        reserve_months=float(reserve),
        mode=mode,
    )
