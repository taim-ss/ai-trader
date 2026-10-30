"""Forward test arena: contestants commit target weights before a round, prices decide later.

An LLM analyst cannot be backtested on past dates (the model may already know the outcome), so
contestants are scored only on closes that did not exist when they committed. Baselines (cash,
index, equal weight, random picks) run in the same rounds. Pass criteria are fixed before any
result exists. This module is pure: it takes price tables and weights, never fetches data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from math import isfinite
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROUND_TRIP_COST_BPS = 50.0  # 25 bps per side (10 slippage + 15 FX), same as the backtest CostModel
WEIGHT_TOLERANCE = 1e-9


def validate_weights(weights: Mapping[str, float], universe: Sequence[str]) -> dict[str, float]:
    """Return clean long-only weights summing to at most 1. Anything else is rejected, never repaired."""
    if not isinstance(weights, Mapping):
        raise ValueError("weights must be a mapping of symbol to fraction")
    allowed = set(universe)
    clean: dict[str, float] = {}
    for symbol, weight in weights.items():
        if symbol not in allowed:
            raise ValueError(f"{symbol}: not in the round universe")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not isfinite(weight):
            raise ValueError(f"{symbol}: weight must be a finite number")
        if weight < 0:
            raise ValueError(f"{symbol}: negative weight (no shorting)")
        if weight > 0:
            clean[symbol] = float(weight)
    if sum(clean.values()) > 1.0 + WEIGHT_TOLERANCE:
        raise ValueError("total weight above 100% (no leverage)")
    return clean


def rule_flags(weights: Mapping[str, float], max_weight: float, max_positions: int) -> list[str]:
    """Flags (not errors): would the real bot's risk rules have allowed this entry?"""
    flags = [f"{s}: {w:.0%} above the {max_weight:.0%} cap" for s, w in weights.items() if w > max_weight + 1e-12]
    if len(weights) > max_positions:
        flags.append(f"{len(weights)} positions above the limit of {max_positions}")
    return flags


NEW_YORK = ZoneInfo("America/New_York")
SESSION_OPEN = time(9, 30)
CLOSE_SETTLED = time(16, 30)  # Yahoo bars are provisional until a little after the 16:00 close


def completed_closes(prices: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """Drop today's bar while the New York session is still open or the close is not settled."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone aware")
    local = now.astimezone(NEW_YORK)
    if len(prices) and prices.index.max().date() == local.date() and local.time() < CLOSE_SETTLED:
        return prices.iloc[:-1]
    return prices


def entries_still_forward(as_of: date, now: datetime) -> bool:
    """True until the next session after `as_of` opens (holidays ignored: conservative, never late)."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone aware")
    following = as_of + timedelta(days=1)
    while following.weekday() >= 5:
        following += timedelta(days=1)
    return now < datetime.combine(following, SESSION_OPEN, tzinfo=NEW_YORK)


def assert_no_future_data(prices: pd.DataFrame, as_of: date) -> None:
    """A round may only be opened or entered while no close after `as_of` exists."""
    if len(prices) and prices.index.max() > pd.Timestamp(as_of):
        raise ValueError("prices already contain closes after the round date; entries would not be forward")


@dataclass(frozen=True)
class HorizonResult:
    horizon: int
    base_date: date
    end_date: date
    gross_return: float
    net_return: float
    exposure: float
    benchmark_return: float
    excess_raw: float  # net return minus the benchmark held fully (after its own entry cost)
    excess_matched: float  # gross return minus exposure * benchmark: skill, not just being under-invested


def _closes(prices: pd.DataFrame, symbol: str, base_pos: int, end_pos: int) -> float:
    if symbol not in prices.columns:
        raise ValueError(f"{symbol}: no price column")
    start, end = float(prices[symbol].iloc[base_pos]), float(prices[symbol].iloc[end_pos])
    if not (isfinite(start) and isfinite(end)) or start <= 0 or end <= 0:
        raise ValueError(f"{symbol}: missing or non-positive close in the window")
    return end / start - 1.0


def horizon_result(prices: pd.DataFrame, as_of: date, weights: Mapping[str, float], benchmark: str,
                   horizon: int, cost_bps: float = ROUND_TRIP_COST_BPS) -> HorizonResult | None:
    """Score weights over `horizon` sessions after the last close on or before `as_of`.

    Returns None while the horizon has not fully elapsed (never a partial score).
    """
    if horizon < 1:
        raise ValueError("horizon must be at least one session")
    base_pos = int(prices.index.searchsorted(pd.Timestamp(as_of), side="right")) - 1
    if base_pos < 0:
        raise ValueError("no close on or before the round date")
    end_pos = base_pos + horizon
    if end_pos >= len(prices):
        return None
    gross = sum(w * _closes(prices, s, base_pos, end_pos) for s, w in weights.items())
    exposure = float(sum(weights.values()))
    bench = _closes(prices, benchmark, base_pos, end_pos)
    cost = cost_bps / 1e4
    return HorizonResult(
        horizon=horizon, base_date=prices.index[base_pos].date(), end_date=prices.index[end_pos].date(),
        gross_return=gross, net_return=gross - exposure * cost, exposure=exposure, benchmark_return=bench,
        excess_raw=(gross - exposure * cost) - (bench - cost), excess_matched=gross - exposure * bench,
    )


def random_net_returns(prices: pd.DataFrame, as_of: date, symbols: Sequence[str], horizon: int, *,
                       count: int, picks: int, weight: float, seed: int,
                       cost_bps: float = ROUND_TRIP_COST_BPS) -> list[float]:
    """Net returns of `count` random equal-weight portfolios: the luck distribution for this round."""
    rng = np.random.default_rng(seed)
    pool = sorted(symbols)
    if len(pool) < picks:
        raise ValueError("not enough symbols for the requested number of picks")
    out: list[float] = []
    for _ in range(count):
        chosen = rng.choice(len(pool), size=picks, replace=False)
        result = horizon_result(prices, as_of, {pool[int(i)]: weight for i in chosen}, pool[0], horizon, cost_bps)
        if result is None:
            return []
        out.append(result.net_return)
    return out


def percentile_rank(value: float, sample: Sequence[float]) -> float:
    """Share of the random sample this value beats (ties count half). 0.5 means no better than luck."""
    if not sample:
        raise ValueError("empty random sample")
    below = sum(1 for x in sample if x < value)
    equal = sum(1 for x in sample if x == value)
    return (below + 0.5 * equal) / len(sample)


def bootstrap_mean_ci(values: Sequence[float], level: float, seed: int = 0,
                      resamples: int = 2000) -> tuple[float, float] | None:
    """Percentile bootstrap interval for the mean. None with fewer than two observations."""
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    data = np.asarray(values, dtype=float)
    means = rng.choice(data, size=(resamples, len(data)), replace=True).mean(axis=1)
    tail = (1 - level) / 2
    return float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))


@dataclass(frozen=True)
class Observation:
    contestant: str
    round_id: str
    horizon: int
    net_return: float
    excess_raw: float
    excess_matched: float
    random_percentile: float | None


@dataclass(frozen=True)
class BoardRow:
    contestant: str
    horizon: int
    rounds: int
    mean_net: float
    mean_excess_matched: float
    beat_rate: float  # share of rounds with excess_matched > 0
    ci_low: float | None
    ci_high: float | None
    mean_random_percentile: float | None


@dataclass(frozen=True)
class Criteria:
    """Fixed before the first result exists. Changing it later must be logged as a new version."""

    min_rounds: int
    horizon: int
    ci_level: float
    min_random_percentile: float


def scoreboard(observations: Sequence[Observation], criteria: Criteria, seed: int = 0) -> list[BoardRow]:
    rows: list[BoardRow] = []
    names = sorted({o.contestant for o in observations if o.horizon == criteria.horizon})
    for name in names:
        mine = [o for o in observations if o.contestant == name and o.horizon == criteria.horizon]
        excess = [o.excess_matched for o in mine]
        pct = [o.random_percentile for o in mine if o.random_percentile is not None]
        interval = bootstrap_mean_ci(excess, criteria.ci_level, seed)
        rows.append(BoardRow(
            contestant=name, horizon=criteria.horizon, rounds=len(mine),
            mean_net=sum(o.net_return for o in mine) / len(mine),
            mean_excess_matched=sum(excess) / len(excess),
            beat_rate=sum(1 for x in excess if x > 0) / len(excess),
            ci_low=interval[0] if interval else None, ci_high=interval[1] if interval else None,
            mean_random_percentile=sum(pct) / len(pct) if pct else None,
        ))
    return rows


def verdict(row: BoardRow, criteria: Criteria) -> tuple[str, list[str]]:
    """NO EVIDENCE YET / PASS / FAIL against the pre-set criteria, with the reasons."""
    if row.rounds < criteria.min_rounds:
        return "NO EVIDENCE YET", [f"{row.rounds} of {criteria.min_rounds} required rounds"]
    reasons: list[str] = []
    if row.mean_excess_matched <= 0:
        reasons.append("mean exposure-matched excess return is not above zero")
    if row.ci_low is None or row.ci_low <= 0:
        reasons.append(f"{criteria.ci_level:.0%} interval for the mean excess includes zero")
    if row.mean_random_percentile is None or row.mean_random_percentile < criteria.min_random_percentile:
        reasons.append(f"average rank against random picks is below {criteria.min_random_percentile:.0%}")
    return ("FAIL", reasons) if reasons else ("PASS", [])
