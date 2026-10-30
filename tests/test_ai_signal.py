from datetime import date
from pathlib import Path
from typing import Any

import pytest

from ai_trader.ai_signal import WatchEntry, live_gate_blockers, load_live_config, load_watchlist, parse_signal
from ai_trader.risk import RiskLimits

ROOT = Path(__file__).resolve().parents[1]
LIMITS = RiskLimits(0.15, 10.0, 0.20, 50.0, False, False, False, 3, 24)
WATCH = {
    "NVDA": WatchEntry("NVDA", "NVDA_US_EQ", "USD", "tier1", None),
    "TSM": WatchEntry("TSM", "TSM_US_EQ", "USD", "tier1", None),
    "CBRS": WatchEntry("CBRS", "CBRS_US_EQ", "USD", "tier1b", 10.0),
}
TODAY = date(2026, 10, 30)


def signal(**changes: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "as_of": "2026-10-30", "run": "routine", "sleeve": "demo",
        "targets": {"NVDA": 0.10, "TSM": 0.05},
        "reasons": {"NVDA": "record data-center demand", "TSM": "capex steady"},
    }
    base.update(changes)
    return base


def parse(raw: dict[str, Any]) -> Any:
    return parse_signal(raw, WATCH, LIMITS, today=TODAY, sleeve_eur=200.0)


def test_valid_signal_is_accepted() -> None:
    result = parse(signal())
    assert result.targets == {"NVDA": 0.10, "TSM": 0.05}
    assert result.sleeve == "demo"


@pytest.mark.parametrize("changes, text", [
    ({"targets": {"NVDA": 0.20}, "reasons": {"NVDA": "x"}}, "exceeds cap"),
    ({"targets": {"AAPL": 0.05}, "reasons": {"AAPL": "x"}}, "not in the tradable watchlist"),
    ({"targets": {"NVDA": -0.05}}, "weight must be"),
    ({"targets": {"NVDA": 0.05}, "reasons": {}}, "needs a written reason"),
    ({"targets": {"CBRS": 0.10}, "reasons": {"CBRS": "x"}}, "name cap"),
    ({"as_of": "2026-10-20"}, "older than"),
    ({"as_of": "2026-11-05"}, "future"),
    ({"sleeve": "real"}, "sleeve must be"),
    ({"verdicts": {"NVDA": {"action": "YOLO"}}}, "verdict needs"),
    ({"forecasts": [{"event": "x", "probability": 1.5, "resolution_date": "2026-11-05"}]}, "forecast needs"),
])
def test_bad_signals_are_rejected(changes: dict[str, Any], text: str) -> None:
    with pytest.raises(ValueError, match=text):
        parse(signal(**changes))


@pytest.mark.parametrize("changes", [
    {"targets": {"NVDA": 10 ** 400}}, {"verdicts": ["NVDA"]}, {"run": []}, {"sleeve": []}, {"forecasts": 5},
])
def test_malformed_shapes_raise_value_error_not_other_exceptions(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        parse(signal(**changes))


def test_all_cash_signal_is_valid_but_non_mapping_targets_are_not() -> None:
    result = parse(signal(targets={}, reasons={}))
    assert result.targets == {} and result.reasons == {}
    with pytest.raises(ValueError, match="targets must be a mapping"):
        parse(signal(targets=[]))


def test_duplicate_watchlist_ticker_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "w.yaml"
    path.write_text("tier1:\n  - {symbol: A, t212_ticker: X_US_EQ, currency: USD}\n"
                    "  - {symbol: B, t212_ticker: X_US_EQ, currency: USD}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_watchlist(path)


def test_too_many_positions_rejected() -> None:
    names = {f"S{i}": WatchEntry(f"S{i}", f"S{i}_US_EQ", "USD", "tier1", None) for i in range(6)}
    raw = signal(targets={n: 0.1 for n in names}, reasons={n: "r" for n in names})
    with pytest.raises(ValueError, match="positions exceed"):
        parse_signal(raw, names, LIMITS, today=TODAY, sleeve_eur=200.0)


def live_config(**changes: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "enabled": True, "sleeve_eur": 200.0, "not_before": date(2026, 10, 30), "mode": "live-with-approval",
        "checklist_required_true": {"deposit_confirmed_in_account": True},
    }
    base.update(changes)
    return base


def test_gate_is_closed_while_adapter_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ai_trader.ai_signal.LIVE_EXECUTION_IMPLEMENTED", False)
    assert any("no live order adapter" in b for b in live_gate_blockers(live_config(), TODAY))


def test_a_fully_owner_enabled_config_is_the_only_way_to_open_the_gate() -> None:
    assert live_gate_blockers(live_config(), TODAY) == []
    for change in ({"enabled": False}, {"mode": "live-auto"}, {"not_before": date(2099, 1, 1)},
                   {"checklist_required_true": {"explicit_written_approval": False}}, {"sleeve_eur": 0}):
        assert live_gate_blockers(live_config(**change), TODAY)


def test_gate_blocks_early_disabled_unchecked_and_auto() -> None:
    config = live_config(enabled=False, not_before=date(2026, 10, 30), mode="live-auto",
                         checklist_required_true={"a": False})
    blockers = live_gate_blockers(config, date(2026, 10, 29))
    text = " | ".join(blockers)
    for fragment in ("disabled", "too early", "live-auto", "checklist not done: a"):
        assert fragment in text


@pytest.mark.parametrize("watchlist_name,live_name", [
    ("watchlist.example.yaml", "live.example.yaml"),
    ("private_watchlist.yaml", "private_live.yaml"),
])
def test_shipped_config_files_load_and_stay_locked(watchlist_name: str, live_name: str) -> None:
    watch_path, live_path = ROOT / "config" / watchlist_name, ROOT / "config" / live_name
    if not (watch_path.exists() and live_path.exists()):
        pytest.skip("private config is git-ignored and absent in a fresh clone")
    watch = load_watchlist(watch_path)
    assert watch and "SPY" not in watch
    if watchlist_name.startswith("private"):
        assert watch["MP"].t212_ticker == "FVAC_US_EQ" and watch["CBRS"].max_eur == 10.0
    config = load_live_config(live_path)
    assert config["enabled"] is False and live_gate_blockers(config, date(2099, 1, 1))


def test_example_signal_matches_the_schema_and_example_watchlist() -> None:
    import json

    raw = json.loads((ROOT / "examples" / "signal.example.json").read_text(encoding="utf-8"))
    raw["as_of"] = TODAY.isoformat()
    raw["forecasts"][0]["resolution_date"] = "2026-11-05"
    watch = load_watchlist(ROOT / "config" / "watchlist.example.yaml")
    result = parse_signal(raw, watch, LIMITS, today=TODAY, sleeve_eur=200.0)
    assert set(result.targets) == {"MSFT", "TSM"} and result.sleeve == "demo"
