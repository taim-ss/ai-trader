"""Portfolio diagnosis: concentration, cap breaches and missing diversification."""

from __future__ import annotations

from dataclasses import dataclass, field

from ai_trader.portfolio import Portfolio


@dataclass(frozen=True)
class Diagnosis:
    holdings_count: int
    largest_weight: float
    largest_ticker: str
    herfindahl: float
    effective_names: float
    names_over_cap: tuple[str, ...]
    findings: tuple[str, ...] = field(default_factory=tuple)


def diagnose(portfolio: Portfolio, single_name_cap: float) -> Diagnosis:
    """Describe concentration and flag problems. Pure function, no market data needed."""
    if not 0 < single_name_cap <= 1:
        raise ValueError("single_name_cap must be in (0, 1]")

    weights = portfolio.weights()
    largest_ticker, largest = max(weights.items(), key=lambda kv: kv[1])
    herfindahl = sum(w * w for w in weights.values())
    effective = 1.0 / herfindahl
    over_cap = tuple(sorted(t for t, w in weights.items() if w > single_name_cap))

    findings: list[str] = []
    if over_cap:
        findings.append(
            f"{len(over_cap)} name(s) above the {single_name_cap:.0%} cap: {', '.join(over_cap)}"
        )
    if len(weights) < 10:
        findings.append(f"only {len(weights)} holdings; diversification is limited")
    if effective < 5:
        findings.append(f"effective number of independent bets is {effective:.1f}")

    return Diagnosis(
        holdings_count=len(weights),
        largest_weight=largest,
        largest_ticker=largest_ticker,
        herfindahl=herfindahl,
        effective_names=effective,
        names_over_cap=over_cap,
        findings=tuple(findings),
    )
