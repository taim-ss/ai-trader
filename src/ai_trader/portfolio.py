"""Holdings import (CSV first). Validates rows and computes weights."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO
from math import isfinite


@dataclass(frozen=True)
class Holding:
    ticker: str
    quantity: float
    value_eur: float


@dataclass(frozen=True)
class Portfolio:
    holdings: tuple[Holding, ...]

    @property
    def total_eur(self) -> float:
        return sum(h.value_eur for h in self.holdings)

    def weights(self) -> dict[str, float]:
        total = self.total_eur
        if total <= 0:
            raise ValueError("portfolio has no value")
        return {h.ticker: h.value_eur / total for h in self.holdings}


REQUIRED_COLUMNS: tuple[str, ...] = ("ticker", "quantity", "value_eur")


def parse_holdings_csv(text: str) -> Portfolio:
    """Parse a holdings CSV with columns ticker, quantity, value_eur.

    Rejects missing columns, duplicate tickers, non-numeric or negative values.
    Cash is not a holding: pass only instruments.
    """
    reader = csv.DictReader(StringIO(text))
    if reader.fieldnames is None or any(c not in reader.fieldnames for c in REQUIRED_COLUMNS):
        raise ValueError(f"CSV must contain columns: {', '.join(REQUIRED_COLUMNS)}")

    seen: set[str] = set()
    rows: list[Holding] = []
    for line_no, row in enumerate(reader, start=2):
        ticker = (row["ticker"] or "").strip().upper()
        if not ticker:
            raise ValueError(f"line {line_no}: empty ticker")
        if ticker in seen:
            raise ValueError(f"line {line_no}: duplicate ticker {ticker}")
        seen.add(ticker)
        try:
            quantity = float(row["quantity"])
            value = float(row["value_eur"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"line {line_no}: quantity and value_eur must be numbers") from exc
        if not isfinite(quantity) or not isfinite(value):
            raise ValueError(f"line {line_no}: quantity and value_eur must be finite")
        if quantity < 0 or value < 0:
            raise ValueError(f"line {line_no}: negative values are not allowed (no shorts)")
        rows.append(Holding(ticker=ticker, quantity=quantity, value_eur=value))

    if not rows:
        raise ValueError("CSV has no holdings")
    return Portfolio(holdings=tuple(rows))
