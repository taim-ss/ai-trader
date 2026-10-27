"""Risk limit model and hard checks. The model output cannot override these."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RiskLimits:
    max_position_weight: float
    max_daily_loss_eur: float
    max_drawdown_kill_switch: float
    max_order_eur: float
    allow_leverage: bool
    allow_shorting: bool
    allow_options: bool
    loss_cooldown_trades: int
    cooldown_hours: int

    def __post_init__(self) -> None:
        for name in ("allow_leverage", "allow_shorting", "allow_options"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for name in ("max_position_weight", "max_daily_loss_eur",
                     "max_drawdown_kill_switch", "max_order_eur"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if self.max_position_weight > 1 or self.max_drawdown_kill_switch > 1:
            raise ValueError("position and drawdown limits must be at most 1")
        for name in ("loss_cooldown_trades", "cooldown_hours"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")


def load_limits(path: Path) -> RiskLimits:
    """Read and validate the limits file. Fails loudly on missing or wrong-typed keys."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at top level")
    return RiskLimits(**raw)


def validate_targets(weights: Mapping[str, float], limits: RiskLimits) -> list[str]:
    """Return violation messages. An empty list means the targets pass."""
    violations: list[str] = []
    for ticker, weight in weights.items():
        if isinstance(weight, bool) or not isfinite(weight):
            violations.append(f"{ticker}: weight must be finite")
            continue
        if weight < 0 and not limits.allow_shorting:
            violations.append(f"{ticker}: negative weight, shorting is disabled")
        if abs(weight) > limits.max_position_weight + 1e-12:
            violations.append(
                f"{ticker}: weight {weight:.1%} exceeds cap {limits.max_position_weight:.1%}"
            )
    total = sum(abs(weight) for weight in weights.values() if isfinite(weight))
    if not isfinite(total):
        violations.append("total weight must be finite")
    if total > 1.0 + 1e-9 and not limits.allow_leverage:
        violations.append(f"total weight {total:.1%} exceeds 100%; leverage is disabled")
    return violations


def validate_order(order_eur: float, limits: RiskLimits) -> list[str]:
    """Reject orders above the per-order cap. Never resize an order to fit."""
    if isinstance(order_eur, bool) or not isfinite(order_eur):
        return ["order value must be finite"]
    if order_eur <= 0:
        return ["order value must be positive"]
    if order_eur > limits.max_order_eur:
        return [f"order €{order_eur:.2f} exceeds the €{limits.max_order_eur:.2f} cap"]
    return []


def kill_switch_engaged(drawdown: float, limits: RiskLimits) -> bool:
    """True when drawdown (as a positive fraction, e.g. 0.25) breaches the kill switch."""
    if isinstance(drawdown, bool) or not isfinite(drawdown) or drawdown < 0:
        raise ValueError("drawdown must be given as a positive fraction")
    return drawdown >= limits.max_drawdown_kill_switch
