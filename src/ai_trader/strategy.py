"""Rules-based target weights: 12-1 month momentum, top-N equal weight, optional core.

Every function here takes prices up to and including `as_of` only. Changing later
prices must never change the output (tested in tests/test_strategy.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

TRADING_DAYS_PER_YEAR = 252
LOOKBACK_DAYS = 252  # about 12 months
SKIP_DAYS = 21  # skip the most recent month (standard 12-1 construction)


@dataclass(frozen=True)
class StrategyConfig:
    top_n: int = 3
    single_name_cap: float = 0.15
    core_ticker: str | None = None
    core_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.top_n < 1:
            raise ValueError("top_n must be at least 1")
        if not 0 < self.single_name_cap <= 1:
            raise ValueError("single_name_cap must be in (0, 1]")
        if not 0 <= self.core_weight < 1:
            raise ValueError("core_weight must be in [0, 1)")
        if (self.core_ticker is None) != (self.core_weight == 0):
            raise ValueError("core_ticker and core_weight must be set together")


def momentum_scores(prices: pd.DataFrame, as_of: date) -> pd.Series:
    """12-1 month total return per ticker, using only rows dated on or before `as_of`."""
    history = prices.loc[: pd.Timestamp(as_of)]
    if len(history) <= LOOKBACK_DAYS:
        return pd.Series(dtype=float)
    recent = history.iloc[-(SKIP_DAYS + 1)]
    past = history.iloc[-(LOOKBACK_DAYS + 1)]
    return (recent / past - 1.0).dropna()


def target_weights(prices: pd.DataFrame, as_of: date, config: StrategyConfig) -> dict[str, float]:
    """Return target weights summing to at most 1. Missing signal means cash, not leverage."""
    scores = momentum_scores(prices, as_of)
    winners = scores[scores > 0].sort_values(ascending=False).head(config.top_n)

    weights: dict[str, float] = {}
    satellite_budget = 1.0 - config.core_weight
    if config.core_ticker is not None:
        weights[config.core_ticker] = config.core_weight

    if not winners.empty:
        per_name = min(satellite_budget / len(winners), config.single_name_cap)
        for ticker in winners.index:
            if ticker == config.core_ticker:
                continue
            weights[str(ticker)] = per_name
    return {t: w for t, w in weights.items() if w > 0}
