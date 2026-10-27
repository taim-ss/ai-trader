"""Monthly signals at close t, executed at the next observed session's close.

Adjusted closes approximate total returns, cash earns 0%, and costs are a flat
basis-point charge on actual traded value. Holdings drift between rebalances.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ai_trader.strategy import StrategyConfig, target_weights

TRADING_DAYS = 252


@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 0.0  # Trading 212 stock commission is reported as 0
    slippage_bps: float = 10.0  # assumption, not measured
    fx_bps: float = 15.0  # 0.15% when the asset currency differs from the account

    @property
    def total_bps(self) -> float:
        return self.commission_bps + self.slippage_bps + self.fx_bps


def month_end_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Last trading day of each calendar month present in the index."""
    series = pd.Series(index, index=index)
    return list(series.groupby(index.to_period("M")).last())


def run_backtest(
    prices: pd.DataFrame,
    config: StrategyConfig,
    costs: CostModel,
    start_value: float = 1000.0,
) -> pd.Series:
    """Return equity with fixed holdings between next-session-close executions.

    This is a close-only execution approximation, not a next-open fill model.
    A final partial month never creates a trade inside the observed sample.
    """
    if prices.isna().any().any():
        raise ValueError("prices contain NaN; clean the data before backtesting")
    if prices.empty or not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("prices must be finite and positive")
    if not prices.index.is_monotonic_increasing or not prices.index.is_unique:
        raise ValueError("price dates must be unique and increasing")
    if not np.isfinite(start_value) or start_value <= 0:
        raise ValueError("start value must be finite and positive")
    cost_rate = costs.total_bps / 10_000.0
    if any(not np.isfinite(c) or c < 0 for c in
           (costs.commission_bps, costs.slippage_bps, costs.fx_bps)) or cost_rate >= 1:
        raise ValueError("costs must be finite, nonnegative, and total less than 100%")
    daily_returns = prices.pct_change().fillna(0.0).to_numpy()
    column_of = {str(name): i for i, name in enumerate(prices.columns)}
    rebalance = set(month_end_dates(pd.DatetimeIndex(prices.index)))
    dates = list(prices.index)

    cash = start_value
    current: dict[str, float] = {}
    pending: dict[str, float] | None = None
    turnover = 0.0
    executions = 0
    curve: list[tuple[pd.Timestamp, float]] = []
    for position, day in enumerate(dates):
        if position > 0 and current:
            current = {t: value * (1.0 + float(daily_returns[position, column_of[t]]))
                       for t, value in current.items()}
        equity = cash + sum(current.values())
        if pending is not None:
            if (any(not np.isfinite(w) or w < 0 for w in pending.values())
                    or sum(pending.values()) > 1 + 1e-12):
                raise ValueError("backtest supports long-only, unleveraged target weights")
            # Solve post-fee equity = pre-fee equity - fee(actual traded value).
            before = equity
            low, high = 0.0, before
            symbols = set(pending) | set(current)
            for _ in range(60):
                candidate = (low + high) / 2
                traded_value = sum(abs(pending.get(t, 0.0) * candidate - current.get(t, 0.0))
                                   for t in symbols)
                if candidate + cost_rate * traded_value > before:
                    high = candidate
                else:
                    low = candidate
            equity = (low + high) / 2
            traded_value = sum(abs(pending.get(t, 0.0) * equity - current.get(t, 0.0))
                               for t in symbols)
            turnover += traded_value / before
            executions += 1
            current = {t: weight * equity for t, weight in pending.items()}
            cash = equity - sum(current.values())
            pending = None
        if day in rebalance and position < len(dates) - 1:
            pending = target_weights(prices.loc[:day], day.date(), config)
        curve.append((day, equity))
    result = pd.Series([v for _, v in curve], index=[d for d, _ in curve], name="equity")
    result.attrs.update(turnover=turnover, execution_count=executions)
    return result


def metrics(equity: pd.Series, turnover_note: str = "") -> dict[str, float]:
    """CAGR, max drawdown, Sharpe and Sortino (rf = 0), from a daily equity curve."""
    if len(equity) < 3:
        raise ValueError("need at least three observations")
    daily = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    running_max = equity.cummax()
    max_dd = float((equity / running_max - 1.0).min())
    std = float(daily.std(ddof=0))
    sharpe = float(np.sqrt(TRADING_DAYS) * daily.mean() / std) if std > 0 else 0.0
    downside = float(np.sqrt((np.minimum(daily, 0.0) ** 2).mean()))
    sortino = float(np.sqrt(TRADING_DAYS) * daily.mean() / downside) if downside > 0 else 0.0
    return {
        "cagr": float(cagr),
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "sortino": sortino,
        "years": years,
        "daily_hit_rate": float((daily > 0).mean()),
        "annual_turnover": float(equity.attrs.get("turnover", np.nan)) / years if years > 0 else 0.0,
    }
