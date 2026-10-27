"""Build an offline dashboard; default public output contains aggregates only.

Use --private to include escaped decision details in git-ignored logs/dashboard.html.
Explicit forecasts require forecast_id, event, probability and resolution_date in
logs/decisions.jsonl. Supply verified outcomes as {"forecast_id": true/false} in
logs/forecast_outcomes.json. A decision without those fields is never calibrated.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
from collections import Counter
from datetime import date
from html import escape
from pathlib import Path
from typing import Any

from ai_trader.calibration import BinaryForecast, CalibrationSummary, score_forecasts

ROOT = Path(__file__).resolve().parents[1]
KNOWN_ACTIONS = frozenset({"blocked", "dry-run", "executed", "hold", "rebalance"})


def load_decisions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"decision log line {number} is not an object")
        records.append(record)
    return records


def calibration_from_records(
    records: list[dict[str, Any]], outcomes: dict[str, bool], as_of: date,
) -> CalibrationSummary:
    forecasts: list[BinaryForecast] = []
    for record in records:
        if not record.get("event") or record.get("probability") is None or not record.get("resolution_date"):
            continue
        forecasts.append(BinaryForecast(
            forecast_id=record["forecast_id"], event=record["event"],
            probability=record["probability"], resolution_date=date.fromisoformat(record["resolution_date"]),
        ))
    return score_forecasts(forecasts, outcomes, as_of)


def table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{escape(value)}</th>" for value in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(value)}</td>" for value in row) + "</tr>" for row in rows)
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def calibration_chart(summary: CalibrationSummary) -> str:
    if not summary.scores:
        return "<p class='muted'>Calibration chart unavailable: no resolved forecasts yet.</p>"
    groups: dict[int, list[tuple[float, int]]] = {}
    for score in summary.scores:
        bucket = min(9, int(score.probability * 10))
        groups.setdefault(bucket, []).append((score.probability, int(score.outcome)))
    points: list[str] = []
    for values in groups.values():
        predicted = sum(probability for probability, _ in values) / len(values)
        observed = sum(outcome for _, outcome in values) / len(values)
        x, y = 50 + 320 * predicted, 360 - 320 * observed
        points.append(f"<circle cx='{x:.2f}' cy='{y:.2f}' r='5' fill='#176e70'><title>Mean probability {predicted:.2f}; observed {observed:.2f}; n={len(values)}</title></circle>")
    return (
        "<svg viewBox='0 0 420 410' role='img' aria-label='Calibration reliability chart' style='max-width:420px;width:100%;height:auto'>"
        "<title>Predicted probability versus observed frequency, grouped in probability deciles</title>"
        "<path d='M50 40 V360 H370' fill='none' stroke='#526578'/>"
        "<path d='M50 360 L370 40' fill='none' stroke='#a6b4be' stroke-dasharray='5 5'/>"
        "<text x='45' y='380' font-size='12'>0</text><text x='365' y='380' font-size='12'>1</text>"
        "<text x='30' y='45' font-size='12'>1</text><text x='125' y='403' font-size='12'>Predicted probability</text>"
        "<text transform='translate(15 280) rotate(-90)' font-size='12'>Observed event frequency</text>"
        + "".join(points) + "</svg><p class='muted'>Dashed line is perfect calibration. Points are probability-decile aggregates; small samples do not establish skill.</p>"
    )


def render_dashboard(root: Path, records: list[dict[str, Any]], summary: CalibrationSummary, private: bool) -> str:
    metrics_path = root / "reports" / "backtest_metrics.csv"
    results = "<p>No historical metrics file is available.</p>"
    if metrics_path.exists():
        with metrics_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        columns = ["series", "cagr", "max_drawdown", "sharpe", "sortino"]
        results = table(
            ["Historical series", "CAGR", "Max drawdown", "Sharpe", "Sortino"],
            [[str(row.get(column, "")) for column in columns] for row in rows],
        ) + "<p class='muted'>CAGR and drawdown are decimal fractions; these are historical backtests, not paper or live returns.</p>"
    chart_path = root / "reports" / "equity_curves.png"
    if chart_path.exists():
        encoded = base64.b64encode(chart_path.read_bytes()).decode("ascii")
        results += f"<img alt='Historical equity curves versus benchmarks' src='data:image/png;base64,{encoded}'>"

    counts = Counter(str(record.get("action")) if record.get("action") in KNOWN_ACTIONS else "other" for record in records)
    decisions = table(["Logged action", "Count"], [[action, str(count)] for action, count in sorted(counts.items())]) if counts else "<p>No decision record yet.</p>"
    brier = "Not available: no scored binary forecasts" if summary.mean_brier is None else f"{summary.mean_brier:.4f}"
    calibration = table(["Measure", "Value"], [
        ["Explicit binary forecasts", str(summary.total)], ["Scored", str(len(summary.scores))],
        ["Pending resolution date", str(summary.pending)], ["Due, outcome missing", str(summary.unresolved)],
        ["Mean Brier score (zero is best)", brier],
    ]) + calibration_chart(summary)
    details = ""
    if private:
        details = "<section><h2>Private decision history</h2>" + table(
            ["UTC timestamp", "Action", "Mode", "Rationale"],
            [[str(record.get(key, "")) for key in ("timestamp_utc", "action", "mode", "rationale")] for record in records[-100:]],
        ) + "</section>"
    visibility = "Private output: includes decision details" if private else "Public-safe output: no holdings, targets, rationale or decision IDs"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Trader — evidence dashboard</title><style>
:root{{color-scheme:light;font-family:system-ui,sans-serif;color:#162635;background:#f1f5f7}}
body{{max-width:1100px;margin:0 auto;padding:40px 24px}}h1{{font-size:2.5rem;letter-spacing:-.05em;margin-bottom:8px}}
h2{{font-size:1.2rem}}p{{line-height:1.6}}section{{background:white;border:1px solid #dbe3e8;border-radius:16px;padding:24px;margin-top:24px}}
.eyebrow{{color:#176e70;font-weight:700;letter-spacing:.12em;font-size:.8rem}}.muted{{color:#526578;font-size:.9rem}}
.status{{border-left:4px solid #b16816;padding-left:16px}}.table-wrap{{overflow:auto}}table{{width:100%;border-collapse:collapse;text-align:left}}
th,td{{padding:12px;border-bottom:1px solid #e5ebef;vertical-align:top}}th{{font-size:.85rem;color:#526578}}img{{width:100%;height:auto;margin-top:18px}}
@media(max-width:600px){{body{{padding:24px 12px}}h1{{font-size:2rem}}section{{padding:16px}}}}
</style></head><body><div class="eyebrow">AI TRADER / RESEARCH EVIDENCE</div>
<h1>Measure first. Trade later.</h1><p class="muted">{visibility}. Built entirely from local files.</p>
<p class="status"><strong>Phase 4 remains incomplete.</strong> No verified broker paper performance is supplied to this dashboard.
Decision logs and order-submission counts do not prove fills or profit. Live trading requires the owner's separate CHECKPOINT 4 approval.</p>
<section><h2>Historical strategy versus benchmarks</h2>{results}</section>
<section><h2>Decision activity</h2>{decisions}<p class="muted">Counts describe log records only; they are not a paper-trading track record.</p></section>
<section><h2>Forecast calibration</h2>{calibration}<p class="muted">Scoring requires an explicit event, probability, resolution date and verified binary outcome.
No forecast probabilities or outcomes are fabricated. Resolution dates must have fully passed.</p></section>{details}
<section><h2>Evidence limits</h2><p>Historical ETF universe selection may contain survivorship bias. USD research returns do not represent EUR account returns.
Assumed costs and daily-close execution do not establish live fill quality. Cash interest and taxes are not modelled. Review REPORT.md alongside this dashboard.</p>
<p class="muted">Educational project. Not financial advice. No external scripts, trackers or network requests.</p></section></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private", action="store_true", help="write private decision history under git-ignored logs/")
    args = parser.parse_args()
    records = load_decisions(ROOT / "logs" / "decisions.jsonl")
    outcomes_path = ROOT / "logs" / "forecast_outcomes.json"
    outcomes = json.loads(outcomes_path.read_text(encoding="utf-8")) if outcomes_path.exists() else {}
    if not isinstance(outcomes, dict):
        raise ValueError("forecast_outcomes.json must be an ID-to-boolean mapping")
    summary = calibration_from_records(records, outcomes, date.today())
    output = ROOT / ("logs" if args.private else "reports") / "dashboard.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_dashboard(ROOT, records, summary, args.private), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
