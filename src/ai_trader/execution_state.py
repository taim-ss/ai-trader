"""Persistent demo risk observations. Call under demo_run_lock for the whole run.

Limits use observed EUR equity, UTC days and broker-reported realized fills. This
is not continuous monitoring. A deposit/withdrawal after initialization requires
manual investigation and rebaselining; it never erases a loss automatically.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from math import isfinite
from pathlib import Path
from typing import Any

from ai_trader.risk import RiskLimits


def _timestamp(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("risk timestamps must be timezone-aware datetimes")
    return value.astimezone(UTC)


def _number(value: object, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError("risk amounts must be finite numbers")
    if nonnegative and value < 0:
        raise ValueError("equity must be nonnegative")
    return float(value)


def _identifier(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value):
        raise ValueError("risk identifiers must be 1-128 safe identifier characters")
    return value


@dataclass(frozen=True)
class RealizedFill:
    fill_id: str
    executed_at: datetime
    pnl_eur: float

    def __post_init__(self) -> None:
        _identifier(self.fill_id)
        _timestamp(self.executed_at)
        _number(self.pnl_eur)


@dataclass(frozen=True)
class CashFlow:
    flow_id: str
    occurred_at: datetime
    amount_eur: float

    def __post_init__(self) -> None:
        _identifier(self.flow_id)
        _timestamp(self.occurred_at)
        _number(self.amount_eur)


@dataclass
class _State:
    account_id: str
    initialized_at: str
    last_observed_at: str
    trading_day: str
    day_open_equity_eur: float
    last_equity_eur: float
    peak_equity_eur: float
    daily_halted: bool = False
    drawdown_halted: bool = False
    cashflow_halted: bool = False
    fills: dict[str, tuple[str, float]] = field(default_factory=dict)
    cashflows: dict[str, tuple[str, float]] = field(default_factory=dict)


def _parse_time(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("invalid risk timestamp")
    return _timestamp(datetime.fromisoformat(value))


def _read_events(value: object) -> dict[str, tuple[str, float]]:
    if not isinstance(value, dict):
        raise ValueError("invalid risk events")
    result: dict[str, tuple[str, float]] = {}
    for key, event in value.items():
        _identifier(key)
        if not isinstance(event, list) or len(event) != 2:
            raise ValueError("invalid risk event")
        result[key] = (_parse_time(event[0]).isoformat(), _number(event[1]))
    return result


def _state_from_json(raw: Any) -> _State:
    if not isinstance(raw, dict) or raw.get("version") != 1 or raw.get("currency") != "EUR":
        raise ValueError("unsupported or malformed risk ledger")
    for name in ("daily_halted", "drawdown_halted", "cashflow_halted"):
        if type(raw.get(name)) is not bool:
            raise ValueError("invalid risk halt flags")
    initialized = _parse_time(raw.get("initialized_at"))
    last = _parse_time(raw.get("last_observed_at"))
    if initialized > last or raw.get("trading_day") != last.date().isoformat():
        raise ValueError("inconsistent risk ledger dates")
    peak = _number(raw.get("peak_equity_eur"), nonnegative=True)
    equity = _number(raw.get("last_equity_eur"), nonnegative=True)
    if peak <= 0 or peak < equity:
        raise ValueError("inconsistent risk equity history")
    fills = _read_events(raw.get("fills"))
    cashflows = _read_events(raw.get("cashflows"))
    if any(not initialized <= _parse_time(event[0]) <= last
           for event in [*fills.values(), *cashflows.values()]):
        raise ValueError("risk event outside observation period")
    return _State(
        account_id=_identifier(raw.get("account_id")),
        initialized_at=initialized.isoformat(), last_observed_at=last.isoformat(),
        trading_day=last.date().isoformat(),
        day_open_equity_eur=_number(raw.get("day_open_equity_eur"), nonnegative=True),
        last_equity_eur=equity, peak_equity_eur=peak,
        daily_halted=raw["daily_halted"], drawdown_halted=raw["drawdown_halted"],
        cashflow_halted=raw["cashflow_halted"], fills=fills, cashflows=cashflows,
    )


def _write_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class RiskLedger:
    """Durable risk state bound to one EUR demo account.

    A companion marker detects deletion of the JSON after initialization. Deleting
    both files cannot be detected locally; keep them with the private order log.
    There is deliberately no automatic reset or rebaseline method.
    """

    def __init__(self, path: Path, limits: RiskLimits) -> None:
        self.path = path
        self.limits = limits
        self._marker = path.with_name(f"{path.name}.initialized")
        self._state = self._load()

    def _load(self) -> _State | None:
        if self.path.exists() != self._marker.exists():
            raise ValueError("risk ledger or initialization marker missing; manual recovery required")
        if not self.path.exists():
            return None
        try:
            return _state_from_json(json.loads(self.path.read_text(encoding="utf-8")))
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError("malformed risk ledger; manual recovery required") from error

    @property
    def initialized_at(self) -> datetime | None:
        return _parse_time(self._state.initialized_at) if self._state else None

    def observe(
        self, *, account_id: str, currency: str, equity_eur: float, now: datetime,
        fills: Sequence[RealizedFill] = (), cashflows: Sequence[CashFlow] = (),
    ) -> list[str]:
        account_id = _identifier(account_id)
        if currency != "EUR":
            raise ValueError("persistent demo risk ledger supports EUR accounts only")
        equity = _number(equity_eur, nonnegative=True)
        observed_at = _timestamp(now)
        # Reload so deletion, corruption or updates cannot be bypassed by a cached instance.
        state = self._load()
        if state is None:
            if self._state is not None:
                raise ValueError("previously observed risk state disappeared")
            if equity <= 0:
                raise ValueError("initial risk baseline requires positive equity")
            state = _State(account_id, observed_at.isoformat(), observed_at.isoformat(),
                           observed_at.date().isoformat(), equity, equity, equity)
        if state.account_id != account_id:
            raise ValueError("risk ledger belongs to a different account")
        if observed_at < _parse_time(state.last_observed_at):
            raise ValueError("risk observation time moved backwards")
        baseline = _parse_time(state.initialized_at)
        # Validate the entire input before changing anything durable.
        additions: list[tuple[dict[str, tuple[str, float]], str, tuple[str, float]]] = []
        for events, destination, is_flow in ((fills, state.fills, False), (cashflows, state.cashflows, True)):
            for event in events:
                if is_flow:
                    if not isinstance(event, CashFlow):
                        raise ValueError("cashflows must contain CashFlow records")
                    event_id, event_at, amount = event.flow_id, event.occurred_at, event.amount_eur
                else:
                    if not isinstance(event, RealizedFill):
                        raise ValueError("fills must contain RealizedFill records")
                    event_id, event_at, amount = event.fill_id, event.executed_at, event.pnl_eur
                timestamp = _timestamp(event_at)
                if timestamp > observed_at:
                    raise ValueError("risk event is in the future")
                if timestamp <= baseline:
                    continue
                entry = (timestamp.isoformat(), _number(amount))
                if event_id in destination and destination[event_id] != entry:
                    raise ValueError("conflicting duplicate risk event")
                additions.append((destination, event_id, entry))
        for destination, event_id, entry in additions:
            if event_id in destination and destination[event_id] != entry:
                raise ValueError("conflicting duplicate risk event")
            destination[event_id] = entry
        if observed_at.date().isoformat() != state.trading_day:
            state.trading_day = observed_at.date().isoformat()
            state.day_open_equity_eur = state.last_equity_eur
            state.daily_halted = False
        state.daily_halted |= state.day_open_equity_eur - equity >= self.limits.max_daily_loss_eur
        state.peak_equity_eur = max(state.peak_equity_eur, equity)
        drawdown = (state.peak_equity_eur - equity) / state.peak_equity_eur
        state.drawdown_halted |= drawdown + 1e-12 >= self.limits.max_drawdown_kill_switch
        state.cashflow_halted |= bool(state.cashflows)
        consecutive_losses = 0
        cooldown_until: datetime | None = None
        for _, (timestamp_text, pnl) in sorted(state.fills.items(), key=lambda item: (item[1][0], item[0])):
            consecutive_losses = consecutive_losses + 1 if pnl < 0 else 0
            if consecutive_losses >= self.limits.loss_cooldown_trades:
                candidate = _parse_time(timestamp_text) + timedelta(hours=self.limits.cooldown_hours)
                cooldown_until = max(cooldown_until, candidate) if cooldown_until else candidate
        state.last_equity_eur = equity
        state.last_observed_at = observed_at.isoformat()
        if not self._marker.exists():
            self._marker.parent.mkdir(parents=True, exist_ok=True)
            with self._marker.open("x", encoding="utf-8") as handle:
                handle.write("Risk ledger initialized; missing JSON requires manual recovery.\n")
                handle.flush()
                os.fsync(handle.fileno())
        _write_atomic(self.path, {**state.__dict__, "version": 1, "currency": "EUR"})
        self._state = state
        violations = []
        if state.daily_halted:
            violations.append("daily EUR loss limit reached; halted for this UTC day")
        if state.drawdown_halted:
            violations.append("drawdown kill switch latched; manual recovery required")
        if state.cashflow_halted:
            violations.append("external cashflow detected; manual rebaseline required")
        if cooldown_until and observed_at < cooldown_until:
            violations.append(f"realized-loss cooldown active until {cooldown_until.isoformat()}")
        return violations


@contextmanager
def demo_run_lock(path: Path) -> Iterator[None]:
    """Exclusive fail-fast lock. A lock left by a crash needs manual investigation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise RuntimeError("demo run lock exists; another run or manual recovery is required") from error
    try:
        with handle:
            handle.write(f"pid={os.getpid()}\n")
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        path.unlink()
