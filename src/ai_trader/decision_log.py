"""Append-only JSON Lines decision log. Every decision gets a forecast ID for calibration."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, date, datetime
from math import isfinite
from pathlib import Path
from typing import Any


def new_forecast_id() -> str:
    return uuid.uuid4().hex


def log_decision(
    path: Path,
    *,
    action: str,
    mode: str,
    targets: dict[str, float],
    violations: list[str],
    rationale: str,
    resolution_date: date | None = None,
    probability: float | None = None,
    event: str | None = None,
) -> dict[str, Any]:
    """Append a decision, optionally defining a complete scoreable binary forecast."""
    if probability is not None and (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not isfinite(probability)
        or not 0 <= probability <= 1
    ):
        raise ValueError("probability must be between 0 and 1")
    if event is not None and (
        not isinstance(event, str) or not event.strip()
        or probability is None or type(resolution_date) is not date
    ):
        raise ValueError("an explicit event requires non-empty text, probability and resolution_date")
    record: dict[str, Any] = {
        "forecast_id": new_forecast_id(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "mode": mode,
        "action": action,
        "targets": targets,
        "violations": violations,
        "rationale": rationale,
        "resolution_date": resolution_date.isoformat() if resolution_date else None,
        "probability": probability,
        "event": event.strip() if event is not None else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return record
