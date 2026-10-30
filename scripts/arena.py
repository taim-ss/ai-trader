"""Forward test arena CLI: open a round, take entries, resolve after the horizon, show the board.

  python scripts/arena.py open                       # freeze today's last close, create baseline entries
  python scripts/arena.py submit --round 2026-10-29 --contestant analyst --file entry.json
  python scripts/arena.py resolve --round 2026-10-29
  python scripts/arena.py board

Arena data lives outside the repo (private): ARENA_DIR, default ../trading-team/arena.
Entry file: {"weights": {"NVDA": 0.1}, "reasons": {"NVDA": "..."}}; weights are fractions of the sleeve.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import truststore
import yaml

from ai_trader.arena import (
    Criteria,
    Observation,
    assert_no_future_data,
    completed_closes,
    entries_still_forward,
    horizon_result,
    percentile_rank,
    random_net_returns,
    rule_flags,
    scoreboard,
    validate_weights,
    verdict,
)
from ai_trader.data import fetch_symbol
from ai_trader.strategy import StrategyConfig, target_weights

ROOT = Path(__file__).resolve().parents[1]
ARENA = Path(os.environ.get("ARENA_DIR", ROOT.parent / "trading-team" / "arena"))
HISTORY_DAYS = 450  # enough sessions for the 12-1 momentum baseline


def load_config() -> dict[str, Any]:
    raw = yaml.safe_load((ARENA / "config.yaml").read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit("arena config.yaml must be a mapping")
    return raw


def fetch_prices(symbols: list[str], as_of_end: date) -> tuple[pd.DataFrame, list[str]]:
    truststore.inject_into_ssl()
    session = requests.Session()
    series: dict[str, pd.Series] = {}
    missing: list[str] = []
    start = as_of_end - timedelta(days=HISTORY_DAYS)
    for symbol in symbols:
        try:
            series[symbol] = fetch_symbol(session, symbol, start, as_of_end + timedelta(days=2))
        except Exception as exc:  # noqa: BLE001 - one bad symbol must not stop the round
            missing.append(f"{symbol} ({type(exc).__name__})")
    frame = pd.DataFrame(series).sort_index()
    frame.index = pd.DatetimeIndex(frame.index).as_unit("ns")
    return frame, missing


def write_json(path: Path, payload: dict[str, Any], *, new: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if new and path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def round_dir(round_id: str) -> Path:
    date.fromisoformat(round_id)
    return ARENA / "rounds" / round_id


def baseline_entries(prices: pd.DataFrame, as_of: date, config: dict[str, Any]) -> dict[str, dict[str, float]]:
    stocks = [s for s in config["stocks"] if s in prices.columns]
    etfs = [s for s in config["etfs"] if s in prices.columns]
    entries: dict[str, dict[str, float]] = {"cash": {}}
    for name, symbol in (("index_spy", config["benchmark"]), ("index_world", config["world"])):
        if symbol in prices.columns:
            entries[name] = {symbol: 1.0}
    if stocks:
        entries["equal_weight_stocks"] = {s: 1.0 / len(stocks) for s in stocks}
    etf_frame = prices[etfs].dropna(how="any")
    entries["momentum_etf_rule"] = target_weights(etf_frame, as_of, StrategyConfig(
        top_n=3, single_name_cap=float(config["max_weight"])))
    return entries


def cmd_open(args: argparse.Namespace) -> int:
    config = load_config()
    symbols = sorted(set(config["stocks"]) | set(config["etfs"]) | {config["benchmark"], config["world"]})
    now = datetime.now(UTC)
    fetched, missing = fetch_prices(symbols, now.date())
    prices = completed_closes(fetched, now)  # never freeze a session that is still trading
    latest = prices.index.max().date()
    as_of = date.fromisoformat(args.as_of) if args.as_of else latest
    if as_of != latest:
        print(f"round date {as_of} is not the latest completed close {latest}; a forward round must open on it",
              file=sys.stderr)
        return 1
    if not entries_still_forward(as_of, now):
        print(f"the next session after {as_of} has already opened; wait for the next completed close", file=sys.stderr)
        return 1
    assert_no_future_data(prices, as_of)
    directory = round_dir(as_of.isoformat())
    if (directory / "package.json").exists():
        print(f"round {as_of} already open", file=sys.stderr)
        return 1
    universe = [s for s in symbols if s in prices.columns and pd.notna(prices[s].iloc[-1])]
    package = {
        "as_of": as_of.isoformat(), "opened_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "universe": universe, "unavailable": missing,
        "last_close": {s: float(prices[s].iloc[-1]) for s in universe},
        "closes_last_60": {s: [round(float(v), 4) for v in prices[s].dropna().iloc[-60:]] for s in universe},
        "benchmark": config["benchmark"], "horizons": config["horizons"], "cost_bps": config["cost_bps"],
        "max_weight": config["max_weight"], "max_positions": config["max_positions"], "criteria": config["criteria"],
    }
    write_json(directory / "package.json", package, new=True)
    for name, weights in baseline_entries(prices, as_of, config).items():
        clean = validate_weights(weights, universe)
        write_json(directory / "entries" / f"{name}.json", {
            "contestant": name, "kind": "baseline", "weights": clean, "reasons": {}, "flags": [],
            "submitted_at_utc": datetime.now(UTC).isoformat(timespec="seconds")}, new=True)
    print(f"opened round {as_of}: {len(universe)} symbols, baselines written")
    if missing:
        print("unavailable: " + ", ".join(missing))
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    directory = round_dir(args.round)
    package = json.loads((directory / "package.json").read_text(encoding="utf-8"))
    as_of = date.fromisoformat(package["as_of"])
    if not entries_still_forward(as_of, datetime.now(UTC)):
        print(f"entry refused: the session after {as_of} has already opened, so this would not be a forward entry",
              file=sys.stderr)
        return 1
    entry =json.loads(Path(args.file).read_text(encoding="utf-8"))
    weights = validate_weights(entry.get("weights", {}), package["universe"])
    flags = rule_flags(weights, float(package["max_weight"]), int(package["max_positions"]))
    write_json(directory / "entries" / f"{args.contestant}.json", {
        "contestant": args.contestant, "kind": args.kind, "weights": weights,
        "reasons": entry.get("reasons", {}), "flags": flags,
        "submitted_at_utc": datetime.now(UTC).isoformat(timespec="seconds")}, new=True)
    print(f"{args.contestant}: {len(weights)} position(s), exposure {sum(weights.values()):.0%}"
          + (f", rule flags: {'; '.join(flags)}" if flags else ""))
    return 0


def cmd_resolve(args: argparse.Namespace) -> int:
    directory = round_dir(args.round)
    package = json.loads((directory / "package.json").read_text(encoding="utf-8"))
    as_of = date.fromisoformat(package["as_of"])
    now = datetime.now(UTC)
    fetched, _ = fetch_prices(package["universe"], now.date())
    prices = completed_closes(fetched, now)
    entries =[json.loads(p.read_text(encoding="utf-8")) for p in sorted((directory / "entries").glob("*.json"))]
    stocks = [s for s in load_config()["stocks"] if s in prices.columns]
    random_cfg = load_config()["random"]
    result: dict[str, Any] = {"as_of": package["as_of"], "horizons": {}}
    for horizon in package["horizons"]:
        sample = random_net_returns(prices, as_of, stocks, int(horizon), count=int(random_cfg["count"]),
                                    picks=int(random_cfg["picks"]), weight=float(random_cfg["weight"]),
                                    seed=int(as_of.strftime("%Y%m%d")) + int(horizon),
                                    cost_bps=float(package["cost_bps"]))
        if not sample:
            continue
        rows: dict[str, Any] = {}
        for entry in entries:
            scored = horizon_result(prices, as_of, entry["weights"], package["benchmark"], int(horizon),
                                    float(package["cost_bps"]))
            if scored is not None:
                rows[entry["contestant"]] = {**scored.__dict__, "base_date": scored.base_date.isoformat(),
                                             "end_date": scored.end_date.isoformat(),
                                             "random_percentile": percentile_rank(scored.net_return, sample),
                                             "flags": entry.get("flags", [])}
        result["horizons"][str(horizon)] = {"random_median": sorted(sample)[len(sample) // 2], "entries": rows}
    write_json(directory / "result.json", result)
    print(f"round {as_of}: resolved horizons {sorted(result['horizons'])} (others not elapsed yet)")
    return 0


def cmd_board(_: argparse.Namespace) -> int:
    criteria = Criteria(**load_config()["criteria"])
    observations: list[Observation] = []
    for path in sorted((ARENA / "rounds").glob("*/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        for horizon, block in result["horizons"].items():
            for name, row in block["entries"].items():
                observations.append(Observation(name, result["as_of"], int(horizon), row["net_return"],
                                                row["excess_raw"], row["excess_matched"], row["random_percentile"]))
    rows = scoreboard(observations, criteria)
    lines = [f"# Arena scoreboard ({criteria.horizon}-session horizon, generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC)", "",
             f"Pass criteria (fixed in advance): at least {criteria.min_rounds} rounds, mean exposure-matched excess > 0, "
             f"{criteria.ci_level:.0%} bootstrap interval above 0, average rank vs random picks >= "
             f"{criteria.min_random_percentile:.0%}. Rounds overlap in time, so treat them as correlated.", "",
             "| Contestant | Rounds | Mean net | Mean excess (matched) | Beat-rate | Interval | vs random | Verdict |",
             "|---|---|---|---|---|---|---|---|"]
    for row in sorted(rows, key=lambda r: -r.mean_excess_matched):
        label, reasons = verdict(row, criteria)
        interval = f"{row.ci_low:+.2%} .. {row.ci_high:+.2%}" if row.ci_low is not None and row.ci_high is not None else "n/a"
        pct = f"{row.mean_random_percentile:.0%}" if row.mean_random_percentile is not None else "n/a"
        lines.append(f"| {row.contestant} | {row.rounds} | {row.mean_net:+.2%} | {row.mean_excess_matched:+.2%} | "
                     f"{row.beat_rate:.0%} | {interval} | {pct} | {label}{' (' + '; '.join(reasons) + ')' if reasons else ''} |")
    if not rows:
        lines.append("| (no resolved rounds yet) | | | | | | | |")
    text = "\n".join(lines) + "\n"
    (ARENA / "scoreboard.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    opener = sub.add_parser("open")
    opener.add_argument("--as-of", default="")
    opener.set_defaults(func=cmd_open)
    submitter = sub.add_parser("submit")
    submitter.add_argument("--round", required=True)
    submitter.add_argument("--contestant", required=True)
    submitter.add_argument("--file", required=True)
    submitter.add_argument("--kind", choices=["llm", "human"], default="llm")
    submitter.set_defaults(func=cmd_submit)
    resolver = sub.add_parser("resolve")
    resolver.add_argument("--round", required=True)
    resolver.set_defaults(func=cmd_resolve)
    sub.add_parser("board").set_defaults(func=cmd_board)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
