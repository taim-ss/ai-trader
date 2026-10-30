"""The live path: closed gate refuses before any read, orders need the gate AND a human-approved order."""

import importlib.util
import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from ai_trader.broker import TRADING212_DEMO_URL, TRADING212_LIVE_URL
from ai_trader.demo_execution import execute_demo_plan, execute_live_plan, execute_plan
from ai_trader.paper_adapter import Trading212Live, Trading212Paper, credentials_from_env
from tests.test_demo_execution import LIMITS, NOW, FakeBroker, plan

OPEN: list[str] = []
APPROVED = ("AAA_US_EQ", 0.2, 100.0)


def live_broker() -> FakeBroker:
    broker = FakeBroker()
    broker.base_url = TRADING212_LIVE_URL
    return broker


def go(broker: FakeBroker, path: Path, *, blockers: list[str], **kwargs: Any) -> dict[str, Any]:
    options: dict[str, Any] = dict(limits=LIMITS, state_dir=path, cycle_id="live-1", fx_native_to_eur=1.0, now=NOW,
                                   price_limits={"AAA_US_EQ": 100.0, "BBB_US_EQ": 100.0},
                                   quote_times_utc={"AAA_US_EQ": NOW, "BBB_US_EQ": NOW, "EURUSD=X": NOW})
    options.update(kwargs)
    return execute_live_plan(broker, plan(), live_blockers=blockers, **options)


def test_closed_gate_refuses_before_touching_state_or_broker(tmp_path: Path) -> None:
    broker = live_broker()
    result = go(broker, tmp_path / "state", blockers=["live sleeve is disabled in config/private_live.yaml"], execute=True,
                approved=APPROVED)
    assert result["status"] == "blocked" and "live gate is closed" in result["violations"][0]
    assert broker.sent == [] and not (tmp_path / "state").exists()


def test_missing_gate_evidence_is_treated_as_closed(tmp_path: Path) -> None:
    result = execute_plan(live_broker(), plan(), environment="live", live_blockers=None, limits=LIMITS,
                          state_dir=tmp_path, cycle_id="c", price_limits={"AAA_US_EQ": 100.0}, fx_native_to_eur=1.0,
                          now=NOW, execute=True, quote_times_utc={"AAA_US_EQ": NOW, "EURUSD=X": NOW}, approved=APPROVED)
    assert "no gate evidence" in result["violations"][0]


def test_open_gate_dry_run_plans_but_never_posts(tmp_path: Path) -> None:
    broker = live_broker()
    result = go(broker, tmp_path, blockers=OPEN)
    assert result["status"] == "dry-run" and result["planned"]["ticker"] == "AAA_US_EQ" and broker.sent == []


def test_open_gate_still_needs_an_approved_order_to_post(tmp_path: Path) -> None:
    broker = live_broker()
    result = go(broker, tmp_path, blockers=OPEN, execute=True)
    assert result["status"] == "blocked" and "human approved" in result["violations"][0] and broker.sent == []


def test_open_gate_and_matching_approval_sends_exactly_once(tmp_path: Path) -> None:
    broker = live_broker()
    result = go(broker, tmp_path, blockers=OPEN, execute=True, approved=APPROVED)
    assert result["status"] == "submitted" and broker.sent == [("AAA_US_EQ", 0.2, 100.0)]
    again = go(broker, tmp_path, blockers=OPEN, execute=True, approved=APPROVED)
    assert len(broker.sent) == 1 and again["status"] != "submitted"  # pending/accepted order blocks a second send


def test_live_order_that_differs_from_the_approved_one_is_refused(tmp_path: Path) -> None:
    broker = live_broker()
    result = go(broker, tmp_path, blockers=OPEN, execute=True, approved=("AAA_US_EQ", 0.3, 100.0))
    assert result["status"] == "blocked" and "approved" in " ".join(result["violations"]) and broker.sent == []


def test_environments_cannot_be_crossed(tmp_path: Path) -> None:
    demo = FakeBroker()
    assert "does not match" in go(demo, tmp_path / "a", blockers=OPEN)["violations"][0]
    assert "exact demo" in execute_demo_plan(live_broker(), plan(), limits=LIMITS, state_dir=tmp_path / "b",
                                             cycle_id="c", price_limits={"AAA_US_EQ": 100.0}, fx_native_to_eur=1.0,
                                             now=NOW)["violations"][0]
    assert demo.sent == []


def test_live_adapter_is_live_only_limit_only_and_uses_live_keys() -> None:
    assert Trading212Live("k", "s").base_url == TRADING212_LIVE_URL
    with pytest.raises(ValueError, match="live environment"):
        Trading212Live("k", "s", base_url=TRADING212_DEMO_URL)
    with pytest.raises(ValueError, match="demo environment"):
        Trading212Paper("k", "s", base_url=TRADING212_LIVE_URL)
    with pytest.raises(NotImplementedError):
        Trading212Live("k", "s").place_market_order("AAA_US_EQ", 1.0)
    demo_only = {"TRADING212_DEMO_API_KEY": "d", "TRADING212_DEMO_API_SECRET": "d"}
    with pytest.raises(RuntimeError, match="LIVE_API_KEY"):
        credentials_from_env(demo_only, live=True)
    assert credentials_from_env({"TRADING212_LIVE_API_KEY": "l", "TRADING212_LIVE_API_SECRET": "l"}, live=True) == ("l", "l")


def runner() -> Any:
    source = Path(__file__).resolve().parents[1] / "scripts" / "live_trade.py"
    spec = importlib.util.spec_from_file_location("live_runner_test", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_config(path: Path, **changes: Any) -> Path:
    config = {"enabled": True, "sleeve_eur": 58.0, "not_before": "2026-10-30", "mode": "live-with-approval",
              "max_orders_per_run": 1, "checklist_required_true": {"explicit_written_approval": True}}
    config.update(changes)
    path.write_text(json.dumps(config), encoding="utf-8")  # JSON is valid YAML
    return path


def forbid_broker(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_: Any, **__: Any) -> None:
        raise AssertionError("the broker must not be constructed")

    monkeypatch.setattr(module, "Trading212Live", boom)


def test_runner_refuses_with_closed_gate_and_never_builds_a_broker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                   capsys: pytest.CaptureFixture[str]) -> None:
    module = runner()
    forbid_broker(module, monkeypatch)
    config = write_config(tmp_path / "live.json", enabled=False)
    monkeypatch.setattr("sys.argv", ["live_trade.py", "--signal", "x.json", "--live-config", str(config), "--approve"])
    assert module.main() == 2
    assert "live gate is closed" in capsys.readouterr().err


def test_runner_open_gate_still_refuses_approval_without_a_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                    capsys: pytest.CaptureFixture[str]) -> None:
    module = runner()
    forbid_broker(module, monkeypatch)
    config = write_config(tmp_path / "live.json", not_before=date(2020, 1, 1).isoformat())
    monkeypatch.setattr("sys.argv", ["live_trade.py", "--signal", "x.json", "--live-config", str(config), "--approve"])
    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)
    assert module.main() == 2
    assert "person typing at a terminal" in capsys.readouterr().err


def test_shipped_live_config_is_closed_so_the_runner_refuses_today(monkeypatch: pytest.MonkeyPatch,
                                                                    capsys: pytest.CaptureFixture[str]) -> None:
    module = runner()
    if not (Path(module.ROOT) / "config" / "private_live.yaml").exists():
        pytest.skip("no private live config in this checkout")
    forbid_broker(module, monkeypatch)
    monkeypatch.setattr("sys.argv", ["live_trade.py", "--signal", "x.json"])
    assert module.main() == 2
    assert "live gate is closed" in capsys.readouterr().err
