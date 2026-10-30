"""Runner integration through the real demo adapter with a fake HTTP transport."""
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

from ai_trader.demo_market import DemoMarket
from ai_trader.paper_adapter import Trading212Paper
from ai_trader.risk import RiskLimits


def test_full_cli_submits_demo_limit_once_and_blocks_retry_while_pending(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "scripts" / "paper_rebalance.py"
    spec = importlib.util.spec_from_file_location("runner_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "STATE_DIR", tmp_path / "data" / "demo_execution")
    monkeypatch.setattr(module, "LOG_PATH", tmp_path / "logs" / "decisions.jsonl")
    monkeypatch.setattr(module, "credentials_from_env", lambda: ("demo-test", "demo-test"))
    monkeypatch.setattr(module, "load_limits", lambda _: RiskLimits(.15, 10, .2, 50, False, False, False, 3, 24))
    monkeypatch.setattr(module.truststore, "inject_into_ssl", lambda: None)
    now = datetime.now(UTC)
    dates = pd.bdate_range(end=pd.Timestamp(now.date()) - pd.Timedelta(days=1), periods=300)
    history = pd.DataFrame({"QQQ": range(100, 400), "XLK": range(200, 500)}, index=dates, dtype=float)
    market = DemoMarket(history, {"QQQ": 100, "XLK": 100}, 1,
                        dates[-1].date(), {"QQQ": now, "XLK": now, "EURUSD=X": now}, {"QQQ": 100, "XLK": 100})
    monkeypatch.setattr(module, "load_demo_market", lambda *args: market)
    submitted = []

    def handler(request):
        assert request.url.host == "demo.trading212.com"
        path = request.url.path
        if request.method == "POST":
            assert path.endswith("/orders/limit")
            body = json.loads(request.content)
            assert body["timeValidity"] == "DAY"
            submitted.append(body)
            return httpx.Response(200, json={"id": 1, "instrument": {"ticker": body["ticker"]},
                "quantity": body["quantity"], "side": "BUY", "status": "NEW"})
        if path.endswith("/account/summary"):
            value = {"id": 1, "currency": "EUR", "totalValue": 5000,
                     "cash": {"availableToTrade": 5000, "reservedForOrders": 0}}
        elif path.endswith("/metadata/instruments"):
            value = [{"ticker": f"{ticker}_EQ", "currencyCode": "USD", "type": "ETF"} for ticker in ("QQQ", "XLK")]
        elif "/history/" in path:
            value = {"items": [], "nextPagePath": None}
        elif path.endswith("/orders/1"):
            value = {"id": 1, "instrument": {"ticker": submitted[0]["ticker"]},
                     "quantity": submitted[0]["quantity"], "side": "BUY", "status": "NEW"}
        elif path.endswith("/orders") and submitted:
            value = [{"id": 1}]
        else:
            value = []
        return httpx.Response(200, json=value)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(module, "Trading212Paper", lambda *args: Trading212Paper(*args, transport=transport))
    monkeypatch.setattr("sys.argv", ["paper_rebalance.py", "--execute", "--demo-only"])
    assert module.main() == 0
    assert len(submitted) == 1
    assert module.main() == 2
    assert len(submitted) == 1


def _runner_module():
    source = Path(__file__).resolve().parents[1] / "scripts" / "paper_rebalance.py"
    spec = importlib.util.spec_from_file_location("runner_approve_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_approve_refuses_without_a_person_at_a_terminal(monkeypatch, capsys):
    module = _runner_module()
    monkeypatch.setattr("sys.argv", ["paper_rebalance.py", "--signal", "x.json", "--approve", "--demo-only"])
    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)
    assert module.main() == 2
    assert "person typing at a terminal" in capsys.readouterr().err


def test_approve_needs_demo_only_a_signal_and_excludes_execute(monkeypatch, capsys):
    module = _runner_module()
    for argv in (["--signal", "x.json", "--approve"],
                 ["--approve", "--demo-only"],
                 ["--signal", "x.json", "--approve", "--execute", "--demo-only"]):
        monkeypatch.setattr("sys.argv", ["paper_rebalance.py", *argv])
        assert module.main() == 2
    assert "blocked" in capsys.readouterr().err
