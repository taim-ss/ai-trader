from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from ai_trader.execution_state import CashFlow, RealizedFill, RiskLedger, demo_run_lock
from ai_trader.risk import RiskLimits


NOW = datetime(2026, 10, 27, 12, tzinfo=UTC)
LIMITS = RiskLimits(0.15, 10.0, 0.20, 50.0, False, False, False, 3, 24)


def observe(ledger: RiskLedger, equity: float = 100.0, now: datetime = NOW) -> list[str]:
    return ledger.observe(account_id="demo-123", currency="EUR", equity_eur=equity, now=now)


def test_daily_halt_survives_restart_and_intraday_recovery(tmp_path: Path) -> None:
    path = tmp_path / "risk.json"
    assert observe(RiskLedger(path, LIMITS)) == []
    assert "daily" in " ".join(observe(RiskLedger(path, LIMITS), 90, NOW + timedelta(hours=1)))
    assert "daily" in " ".join(observe(RiskLedger(path, LIMITS), 101, NOW + timedelta(hours=2)))


def test_new_utc_day_uses_previous_observation_and_resets_daily_latch(tmp_path: Path) -> None:
    path = tmp_path / "risk.json"
    assert observe(RiskLedger(path, LIMITS), 100) == []
    assert observe(RiskLedger(path, LIMITS), 95, NOW + timedelta(hours=1)) == []
    assert "daily" in " ".join(observe(RiskLedger(path, LIMITS), 85, NOW + timedelta(days=1)))
    assert observe(RiskLedger(path, LIMITS), 85, NOW + timedelta(days=2)) == []


def test_drawdown_halt_survives_recovery_and_later_days(tmp_path: Path) -> None:
    path = tmp_path / "risk.json"
    observe(RiskLedger(path, LIMITS))
    assert "drawdown" in " ".join(observe(RiskLedger(path, LIMITS), 80, NOW + timedelta(hours=1)))
    assert "drawdown" in " ".join(observe(RiskLedger(path, LIMITS), 120, NOW + timedelta(days=1)))


def test_high_water_mark_is_observed_and_persistent(tmp_path: Path) -> None:
    path = tmp_path / "risk.json"
    observe(RiskLedger(path, LIMITS))
    observe(RiskLedger(path, LIMITS), 200, NOW + timedelta(hours=1))
    assert "drawdown" in " ".join(observe(RiskLedger(path, LIMITS), 160, NOW + timedelta(hours=2)))


def test_fill_deduplication_cooldown_and_expiry(tmp_path: Path) -> None:
    path = tmp_path / "risk.json"
    ledger = RiskLedger(path, LIMITS)
    observe(ledger)
    fills = [RealizedFill(str(i), NOW + timedelta(minutes=i), -1) for i in range(1, 4)]
    def with_fills(events: list[RealizedFill], at: datetime) -> list[str]:
        return RiskLedger(path, LIMITS).observe(account_id="demo-123", currency="EUR",
                                               equity_eur=100, now=at, fills=events)
    assert with_fills([fills[0], fills[0]], NOW + timedelta(minutes=10)) == []
    assert with_fills(fills[:2], NOW + timedelta(minutes=11)) == []
    assert "cooldown" in " ".join(with_fills(fills, NOW + timedelta(minutes=12)))
    assert "cooldown" in " ".join(observe(RiskLedger(path, LIMITS), 100, NOW + timedelta(hours=1)))
    assert with_fills(fills, NOW + timedelta(hours=24, minutes=3)) == []


def test_winning_trade_breaks_loss_streak_without_clearing_active_cooldown(tmp_path: Path) -> None:
    path = tmp_path / "risk.json"
    observe(RiskLedger(path, LIMITS))
    events = [RealizedFill(str(i), NOW + timedelta(minutes=i), pnl)
              for i, pnl in enumerate([-1, -1, 1, -1, -1], start=1)]
    ledger = RiskLedger(path, LIMITS)
    assert ledger.observe(account_id="demo-123", currency="EUR", equity_eur=100,
                          now=NOW + timedelta(minutes=10), fills=events) == []
    events.append(RealizedFill("6", NOW + timedelta(minutes=11), -1))
    assert "cooldown" in " ".join(ledger.observe(
        account_id="demo-123", currency="EUR", equity_eur=100,
        now=NOW + timedelta(minutes=12), fills=events))
    assert "cooldown" in " ".join(ledger.observe(
        account_id="demo-123", currency="EUR", equity_eur=100,
        now=NOW + timedelta(minutes=14),
        fills=[RealizedFill("7", NOW + timedelta(minutes=13), 10)]))


def test_historical_fills_and_cashflows_are_not_new_losses_or_deposits(tmp_path: Path) -> None:
    ledger = RiskLedger(tmp_path / "risk.json", LIMITS)
    assert ledger.initialized_at is None
    historical = [RealizedFill(str(i), NOW - timedelta(minutes=i), -10) for i in range(3)]
    assert ledger.observe(account_id="demo-123", currency="EUR", equity_eur=100, now=NOW,
                          fills=historical, cashflows=[CashFlow("d", NOW, 100)]) == []
    assert ledger.initialized_at == NOW


@pytest.mark.parametrize("amount", [-50.0, 50.0])
def test_external_cashflow_latches_and_does_not_erase_daily_losses(tmp_path: Path, amount: float) -> None:
    path = tmp_path / "risk.json"
    observe(RiskLedger(path, LIMITS))
    violation = RiskLedger(path, LIMITS).observe(
        account_id="demo-123", currency="EUR", equity_eur=90, now=NOW + timedelta(hours=1),
        cashflows=[CashFlow("external", NOW + timedelta(minutes=1), amount)])
    assert "cashflow" in " ".join(violation)
    assert "daily" in " ".join(violation)
    assert "cashflow" in " ".join(observe(RiskLedger(path, LIMITS), 150, NOW + timedelta(days=1)))


def test_account_binding_currency_and_clock_are_enforced(tmp_path: Path) -> None:
    ledger = RiskLedger(tmp_path / "risk.json", LIMITS)
    observe(ledger)
    for override in ({"account_id": "other"}, {"currency": "USD"}, {"now": NOW - timedelta(seconds=1)},
                     {"account_id": "x" * 129}, {"now": NOW.replace(tzinfo=None)}):
        args: dict[str, object] = dict(account_id="demo-123", currency="EUR", equity_eur=100, now=NOW)
        with pytest.raises(ValueError):
            ledger.observe(**{**args, **override})  # type: ignore[arg-type]


@pytest.mark.parametrize("equity", [float("nan"), float("inf"), -1, True])
def test_bad_equity_does_not_initialize_ledger(tmp_path: Path, equity: float) -> None:
    path = tmp_path / "risk.json"
    with pytest.raises(ValueError):
        observe(RiskLedger(path, LIMITS), equity)
    assert not path.exists()


def test_zero_equity_is_allowed_only_after_positive_baseline(tmp_path: Path) -> None:
    path = tmp_path / "risk.json"
    with pytest.raises(ValueError):
        observe(RiskLedger(path, LIMITS), 0)
    observe(RiskLedger(path, LIMITS))
    assert "drawdown" in " ".join(observe(RiskLedger(path, LIMITS), 0, NOW + timedelta(hours=1)))


def test_duplicate_conflict_and_future_fill_do_not_overwrite_state(tmp_path: Path) -> None:
    path = tmp_path / "risk.json"
    ledger = RiskLedger(path, LIMITS)
    observe(ledger)
    snapshot = path.read_bytes()
    for events in ([RealizedFill("f", NOW + timedelta(days=1), -1)],
                   [RealizedFill("f", NOW + timedelta(minutes=1), -1),
                    RealizedFill("f", NOW + timedelta(minutes=1), -2)]):
        with pytest.raises(ValueError):
            ledger.observe(account_id="demo-123", currency="EUR", equity_eur=100,
                           now=NOW + timedelta(hours=1), fills=events)
        assert path.read_bytes() == snapshot


def test_same_fill_time_in_different_timezone_is_a_duplicate(tmp_path: Path) -> None:
    path = tmp_path / "risk.json"
    observe(RiskLedger(path, LIMITS))
    at = NOW + timedelta(minutes=1)
    for timestamp in (at, at.astimezone(timezone(timedelta(hours=2)))):
        assert RiskLedger(path, LIMITS).observe(
            account_id="demo-123", currency="EUR", equity_eur=100,
            now=NOW + timedelta(hours=1), fills=[RealizedFill("f", timestamp, -1)]) == []


def test_corrupt_state_and_deleted_marker_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "risk.json"
    observe(RiskLedger(path, LIMITS))
    original = path.read_text(encoding="utf-8")
    for invalid in ('{', '{}', original.replace('"daily_halted": false', '"daily_halted": "false"')):
        path.write_text(invalid, encoding="utf-8")
        with pytest.raises(ValueError, match="malformed"):
            RiskLedger(path, LIMITS)
    path.write_text(original, encoding="utf-8")
    path.with_name("risk.json.initialized").unlink()
    with pytest.raises(ValueError, match="missing"):
        RiskLedger(path, LIMITS)


def test_deleted_json_detected_on_restart_and_existing_instance(tmp_path: Path) -> None:
    path = tmp_path / "risk.json"
    ledger = RiskLedger(path, LIMITS)
    observe(ledger)
    path.unlink()
    with pytest.raises(ValueError, match="missing"):
        RiskLedger(path, LIMITS)
    with pytest.raises(ValueError, match="missing"):
        observe(ledger)


def test_lock_rejects_competing_run_and_cleans_up_after_exception(tmp_path: Path) -> None:
    path = tmp_path / "demo.lock"
    with pytest.raises(ValueError, match="simulated"):
        with demo_run_lock(path):
            assert path.exists()
            with pytest.raises(RuntimeError, match="lock exists"):
                with demo_run_lock(path):
                    pytest.fail("competing run entered lock")
            raise ValueError("simulated")
    assert not path.exists()


def test_stale_lock_is_not_automatically_removed(tmp_path: Path) -> None:
    path = tmp_path / "demo.lock"
    path.write_text("pid=not-running", encoding="utf-8")
    with pytest.raises(RuntimeError, match="manual recovery"):
        with demo_run_lock(path):
            pytest.fail("stale lock was bypassed")
    assert path.read_text(encoding="utf-8") == "pid=not-running"
