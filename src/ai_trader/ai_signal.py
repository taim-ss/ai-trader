"""Validate AI analyst signals and evaluate the live-sleeve gate. Pure functions, no network.

The analyst proposes target weights; this module only checks them against the private watchlist
and the hard risk limits. A rejected signal is reported, never repaired. Nothing here places an order.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from math import isfinite
from pathlib import Path
from typing import Any, TypeGuard

import yaml

from ai_trader.risk import RiskLimits, validate_targets

# Flip only after a live adapter with persistent loss/drawdown latches and reconciliation exists and is tested.
# True since 2026-10-29: the human-approved live path exists (execute_live_plan, scripts/live_trade.py).
# It changes nothing by itself: the gate still needs enabled + every checklist item set by the owner.
LIVE_EXECUTION_IMPLEMENTED = True

WATCHLIST_SECTIONS = ("core", "tier1", "tier1b", "tier2", "tier3")
RUNS = frozenset({"routine", "event", "live-gate"})
SLEEVES = frozenset({"demo", "live"})
ACTIONS = frozenset({"BUY", "HOLD", "TRIM", "WAIT", "NO"})
MAX_SIGNAL_AGE_DAYS = 3
MAX_POSITIONS = 5


@dataclass(frozen=True)
class WatchEntry:
    symbol: str
    t212_ticker: str
    currency: str
    section: str
    max_eur: float | None


@dataclass(frozen=True)
class Forecast:
    event: str
    probability: float
    resolution_date: date


@dataclass(frozen=True)
class AiSignal:
    as_of: date
    run: str
    sleeve: str
    targets: dict[str, float]
    reasons: dict[str, str]
    forecasts: tuple[Forecast, ...] = ()


def _is_number(value: Any) -> TypeGuard[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return isfinite(value)
    except OverflowError:  # an absurdly large JSON integer
        return False


def _as_date(value: Any) -> date | None:
    if type(value) is date:
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


def load_watchlist(path: Path) -> dict[str, WatchEntry]:
    """Read the private watchlist. Only entries with a Trading 212 ticker and sleeve_ok are tradable."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at top level")
    entries: dict[str, WatchEntry] = {}
    tickers: set[str] = set()
    for section in WATCHLIST_SECTIONS:
        for row in raw.get(section, []):
            if not isinstance(row, dict):
                raise ValueError(f"{section}: every entry must be a mapping")
            symbol, ticker = row.get("symbol"), row.get("t212_ticker")
            if not isinstance(symbol, str) or not isinstance(ticker, str) or not symbol or not ticker:
                raise ValueError(f"{section}: entry needs symbol and t212_ticker")
            if symbol in entries or ticker in tickers:
                raise ValueError(f"{symbol}: duplicate watchlist symbol or t212_ticker")
            tickers.add(ticker)
            if row.get("sleeve_ok", True) is not True:
                continue
            cap = row.get("max_eur")
            if cap is not None and (not _is_number(cap) or cap <= 0):
                raise ValueError(f"{symbol}: max_eur must be a positive number")
            entries[symbol] = WatchEntry(symbol, ticker, str(row.get("currency", "")), section,
                                         float(cap) if cap is not None else None)
    return entries


def parse_signal(raw: Mapping[str, Any], watchlist: Mapping[str, WatchEntry], limits: RiskLimits, *,
                 today: date, sleeve_eur: float, max_positions: int = MAX_POSITIONS) -> AiSignal:
    """Validate an analyst signal. Raises ValueError listing every problem found."""
    problems: list[str] = []
    as_of = _as_date(raw.get("as_of"))
    if as_of is None:
        problems.append("as_of must be an ISO date")
    elif as_of > today:
        problems.append("as_of is in the future")
    elif today - as_of > timedelta(days=MAX_SIGNAL_AGE_DAYS):
        problems.append(f"signal is older than {MAX_SIGNAL_AGE_DAYS} days")
    run, sleeve = raw.get("run"), raw.get("sleeve")
    if not isinstance(run, str) or run not in RUNS:
        problems.append(f"run must be one of {sorted(RUNS)}")
    if not isinstance(sleeve, str) or sleeve not in SLEEVES:
        problems.append(f"sleeve must be one of {sorted(SLEEVES)}")
    if not _is_number(sleeve_eur) or sleeve_eur <= 0:
        problems.append("sleeve_eur must be a positive number")

    targets_raw, reasons_raw = raw.get("targets"), raw.get("reasons", {})
    targets: dict[str, float] = {}
    reasons: dict[str, str] = {}
    if not isinstance(targets_raw, dict):  # an empty mapping is a valid all-cash signal
        problems.append("targets must be a mapping")
    else:
        for symbol, weight in targets_raw.items():
            if symbol not in watchlist:
                problems.append(f"{symbol}: not in the tradable watchlist")
            elif not _is_number(weight) or weight < 0:
                problems.append(f"{symbol}: weight must be a finite number, zero or above")
            elif weight > 0:
                targets[symbol] = float(weight)
        problems.extend(validate_targets(targets, limits))
        if len(targets) > max_positions:
            problems.append(f"{len(targets)} positions exceed the limit of {max_positions}")
        if _is_number(sleeve_eur) and sleeve_eur > 0:
            for symbol, weight in targets.items():
                cap = watchlist[symbol].max_eur if symbol in watchlist else None
                if cap is not None and weight * sleeve_eur > cap + 1e-9:
                    problems.append(f"{symbol}: EUR {weight * sleeve_eur:.2f} exceeds the EUR {cap:.2f} name cap")
        if not isinstance(reasons_raw, dict):
            problems.append("reasons must be a mapping")
        else:
            for symbol in targets:
                reason = reasons_raw.get(symbol)
                if not isinstance(reason, str) or not reason.strip():
                    problems.append(f"{symbol}: a non-zero weight needs a written reason")
                else:
                    reasons[symbol] = reason.strip()

    verdicts = raw.get("verdicts") or {}
    if not isinstance(verdicts, dict):
        problems.append("verdicts must be a mapping")
        verdicts = {}
    for symbol, verdict in verdicts.items():
        action = verdict.get("action") if isinstance(verdict, dict) else None
        if symbol not in watchlist or not isinstance(action, str) or action not in ACTIONS:
            problems.append(f"{symbol}: verdict needs a watchlist symbol and an action in {sorted(ACTIONS)}")
    forecasts: list[Forecast] = []
    forecasts_raw = raw.get("forecasts") or []
    if not isinstance(forecasts_raw, list):
        problems.append("forecasts must be a list")
        forecasts_raw = []
    for forecast in forecasts_raw:
        due = _as_date(forecast.get("resolution_date")) if isinstance(forecast, dict) else None
        prob = forecast.get("probability") if isinstance(forecast, dict) else None
        event = forecast.get("event") if isinstance(forecast, dict) else None
        if due is None or not _is_number(prob) or not 0 <= prob <= 1 or not isinstance(event, str) or not event.strip():
            problems.append("forecast needs an event, a probability in [0, 1] and an ISO resolution_date")
        else:
            forecasts.append(Forecast(event.strip(), float(prob), due))
    if problems:
        raise ValueError("; ".join(problems))
    assert as_of is not None and isinstance(run, str) and isinstance(sleeve, str)
    return AiSignal(as_of, run, sleeve, targets, reasons, tuple(forecasts))


def load_live_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at top level")
    return raw


def live_gate_blockers(config: Mapping[str, Any], today: date) -> list[str]:
    """Every reason the live sleeve may not trade. An empty list means the gate is open."""
    blockers: list[str] = []
    if config.get("enabled") is not True:
        blockers.append("live sleeve is disabled in config/private_live.yaml")
    not_before = _as_date(config.get("not_before"))
    if not_before is None:
        blockers.append("not_before must be an ISO date")
    elif today < not_before:
        blockers.append(f"too early: not before {not_before.isoformat()}")
    if config.get("mode") != "live-with-approval":
        blockers.append("only live-with-approval is permitted; live-auto stays off")
    sleeve = config.get("sleeve_eur")
    if not _is_number(sleeve) or sleeve <= 0:
        blockers.append("sleeve_eur must be a positive number")
    checklist = config.get("checklist_required_true")
    if not isinstance(checklist, dict) or not checklist:
        blockers.append("checklist is missing")
    else:
        blockers.extend(f"checklist not done: {name}" for name, done in checklist.items() if done is not True)
    if not LIVE_EXECUTION_IMPLEMENTED:
        blockers.append("no live order adapter with persistent loss latches exists yet")
    return blockers
