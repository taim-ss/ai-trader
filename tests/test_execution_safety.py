"""Regression coverage for the safe planning CLI and imported data boundaries."""

import importlib.util
import math
from pathlib import Path

import httpx
import pytest

from ai_trader.paper_adapter import Trading212Paper
from ai_trader.portfolio import parse_holdings_csv
from ai_trader.profile import parse_profile


def load_cli():
    path = Path(__file__).resolve().parents[1] / "scripts" / "paper_rebalance.py"
    spec = importlib.util.spec_from_file_location("paper_rebalance", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_execute_is_blocked_before_market_or_account_requests(monkeypatch, capsys):
    module = load_cli()
    monkeypatch.setattr("sys.argv", ["paper_rebalance.py", "--execute"])

    def forbidden(*args, **kwargs):
        pytest.fail("submission flag must be rejected before any network or credentials")

    monkeypatch.setattr(module, "load_demo_market", forbidden)
    monkeypatch.setattr(module, "credentials_from_env", forbidden)
    assert module.main() == 2
    assert "persistent daily-loss" in capsys.readouterr().err


def test_holdings_cannot_silently_merge_different_listings():
    module = load_cli()
    positions = [{"instrument": {"ticker": "SPY_DE_EQ"}, "quantity": 1.0}]
    with pytest.raises(ValueError, match="outside verified"):
        module.verified_holdings(positions, {"SPY": "SPY_US_EQ"})
    position = {"instrument": {"ticker": "SPY_US_EQ"}, "quantity": 1.0}
    with pytest.raises(ValueError, match="duplicate"):
        module.verified_holdings([position, position], {"SPY": "SPY_US_EQ"})


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_external_values_cannot_reach_orders(bad):
    with pytest.raises(ValueError):
        parse_holdings_csv(f"ticker,quantity,value_eur\nSPY,{bad},1\n")
    with pytest.raises(ValueError):
        parse_holdings_csv(f"ticker,quantity,value_eur\nSPY,1,{bad}\n")
    for field in ("budget_eur", "reserve_months"):
        profile = dict(goal="test", horizon="3m", risk_tolerance="low",
                       budget_eur=100, reserve_months=1)
        profile[field] = bad
        with pytest.raises(ValueError):
            parse_profile(profile)

    def forbidden(request):
        pytest.fail("invalid quantity reached the network")

    broker = Trading212Paper("test", "test", transport=httpx.MockTransport(forbidden))
    try:
        with pytest.raises(ValueError):
            broker.place_market_order("SPY_US_EQ", bad)
    finally:
        broker.close()
