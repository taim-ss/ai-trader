"""AI-signal demo path: helpers, quote-only market loader, and the runner with a fake broker."""
import importlib.util
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from ai_trader.ai_demo import load_demo_signal, signal_cycle_id, sleeve_holdings, tradable_usd_map
from ai_trader.ai_signal import WatchEntry
from ai_trader.demo_market import DemoQuotes, load_demo_quotes
from ai_trader.paper_adapter import Trading212Paper
from ai_trader.risk import RiskLimits
from test_demo_market import FakeHttp, NOW, payload

LIMITS = RiskLimits(.15, 10, .2, 50, False, False, False, 3, 24)
WATCH = {
    "NVDA": WatchEntry("NVDA", "NVDA_US_EQ", "USD", "tier1", None),
    "TSM": WatchEntry("TSM", "TSM_US_EQ", "USD", "tier1", None),
    "WORLD_ETF_A": WatchEntry("WORLD_ETF_A", "ALLWd_EQ", "EUR", "core", None),
}
INSTRUMENTS = [
    {"ticker": "NVDA_US_EQ", "currencyCode": "USD", "type": "STOCK"},
    {"ticker": "TSM_US_EQ", "currencyCode": "USD", "type": "STOCK"},
    {"ticker": "ALLWd_EQ", "currencyCode": "EUR", "type": "ETF"},
]


def test_only_unique_usd_instruments_are_tradable() -> None:
    assert tradable_usd_map(WATCH, INSTRUMENTS) == {"NVDA": "NVDA_US_EQ", "TSM": "TSM_US_EQ"}
    assert "NVDA" not in tradable_usd_map(WATCH, INSTRUMENTS + [INSTRUMENTS[0]])
    assert tradable_usd_map(WATCH, [{"ticker": "NVDA_US_EQ", "currencyCode": "USD", "type": "CFD"}]) == {}


def test_holdings_split_sleeve_from_outside_positions_and_reject_duplicates() -> None:
    tradable = {"NVDA": "NVDA_US_EQ"}
    positions = [{"instrument": {"ticker": "NVDA_US_EQ"}, "quantity": 1.5},
                 {"instrument": {"ticker": "QQQ_EQ"}, "quantity": 2}]
    assert sleeve_holdings(positions, tradable) == ({"NVDA": 1.5}, ["QQQ_EQ"])
    with pytest.raises(ValueError, match="duplicate"):
        sleeve_holdings(positions[:1] * 2, tradable)
    with pytest.raises(ValueError, match="invalid position quantity"):
        sleeve_holdings([{"instrument": {"ticker": "NVDA_US_EQ"}, "quantity": -1}], tradable)


def write_signal(path: Path, **changes: Any) -> Path:
    body: dict[str, Any] = {"as_of": date.today().isoformat(), "run": "routine", "sleeve": "demo",
                            "targets": {"NVDA": 0.10}, "reasons": {"NVDA": "evidence"},
                            "forecasts": [{"event": "NVDA closes higher", "probability": .55,
                                           "resolution_date": (date.today() + timedelta(days=7)).isoformat()}]}
    body.update(changes)
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


@pytest.mark.parametrize("changes,text", [({"sleeve": "live"}, "demo-sleeve"), ({"run": "live-gate"}, "demo-sleeve")])
def test_live_signals_are_refused_by_the_demo_runner(tmp_path: Path, changes: dict[str, Any], text: str) -> None:
    with pytest.raises(ValueError, match=text):
        load_demo_signal(write_signal(tmp_path / "s.json", **changes), WATCH, LIMITS, today=date.today(), sleeve_eur=200)


def test_nan_and_non_object_signal_files_are_refused(tmp_path: Path) -> None:
    bad = tmp_path / "s.json"
    bad.write_text('{"targets": {"NVDA": NaN}}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        load_demo_signal(bad, WATCH, LIMITS, today=date.today(), sleeve_eur=200)
    bad.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_demo_signal(bad, WATCH, LIMITS, today=date.today(), sleeve_eur=200)


def test_signal_file_with_powershell_bom_is_accepted(tmp_path: Path) -> None:
    path = write_signal(tmp_path / "s.json")
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    assert load_demo_signal(path, WATCH, LIMITS, today=date.today(), sleeve_eur=200).targets == {"NVDA": 0.10}


def test_cycle_id_is_stable_per_signal(tmp_path: Path) -> None:
    signal = load_demo_signal(write_signal(tmp_path / "s.json"), WATCH, LIMITS, today=date.today(), sleeve_eur=200)
    cycle = signal_cycle_id(signal)
    assert cycle.startswith(f"ai-{date.today().isoformat()}-routine-") and signal_cycle_id(signal) == cycle
    changed = load_demo_signal(write_signal(tmp_path / "t.json", targets={"NVDA": 0.05}), WATCH, LIMITS,
                               today=date.today(), sleeve_eur=200)
    assert signal_cycle_id(changed) != cycle  # a corrected signal never inherits the old cycle's done state
    assert signal.forecasts[0].probability == .55


def test_quote_loader_needs_no_history_and_writes_a_snapshot(tmp_path: Path) -> None:
    http = FakeHttp()
    http.payloads = {"CBRS": payload("CBRS", days=5), "EURUSD=X": payload("EURUSD=X", days=5)}
    quotes = load_demo_quotes(http, ["CBRS"], tmp_path, NOW)
    assert isinstance(quotes, DemoQuotes) and quotes.prices_usd == {"CBRS": 999.0}
    assert quotes.fx_usd_to_eur == .8 and set(quotes.quote_times_utc) == {"CBRS", "EURUSD=X"}
    assert len(list((tmp_path / "demo_market").glob("*.json"))) == 1


@pytest.mark.parametrize("symbols", [[], ["NVDA", "NVDA"], ["nvda"]])
def test_quote_loader_refuses_bad_symbol_lists(tmp_path: Path, symbols: list[str]) -> None:
    with pytest.raises(ValueError):
        load_demo_quotes(FakeHttp(), symbols, tmp_path, NOW)


def load_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    source = Path(__file__).resolve().parents[1] / "scripts" / "paper_rebalance.py"
    spec = importlib.util.spec_from_file_location("runner_ai_test", source)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "STATE_DIR", tmp_path / "data" / "demo_execution")
    monkeypatch.setattr(module, "LOG_PATH", tmp_path / "logs" / "decisions.jsonl")
    monkeypatch.setattr(module, "credentials_from_env", lambda: ("demo-test", "demo-test"))
    monkeypatch.setattr(module, "load_limits", lambda _: LIMITS)
    monkeypatch.setattr(module.truststore, "inject_into_ssl", lambda: None)
    now = datetime.now(UTC)
    quotes = DemoQuotes({"NVDA": 200.0, "TSM": 400.0}, 1.0, {"NVDA": now, "TSM": now, "EURUSD=X": now})
    monkeypatch.setattr(module, "load_demo_quotes", lambda *args: quotes)
    return module


def watchlist_file(tmp_path: Path) -> Path:
    path = tmp_path / "watch.yaml"
    path.write_text("tier1:\n  - {symbol: NVDA, t212_ticker: NVDA_US_EQ, currency: USD}\n"
                    "  - {symbol: TSM, t212_ticker: TSM_US_EQ, currency: USD}\n"
                    "core:\n  - {symbol: WORLD_ETF_A, t212_ticker: ALLWd_EQ, currency: EUR}\n", encoding="utf-8")
    return path


def fake_broker(monkeypatch: pytest.MonkeyPatch, module: Any, submitted: list[dict[str, Any]]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "demo.trading212.com"
        path = request.url.path
        if request.method == "POST":
            body = json.loads(request.content)
            assert body["timeValidity"] == "DAY"
            submitted.append(body)
            return httpx.Response(200, json={"id": 7, "instrument": {"ticker": body["ticker"]},
                                             "quantity": body["quantity"], "side": "BUY", "status": "NEW"})
        if path.endswith("/account/summary"):
            value: Any = {"id": 1, "currency": "EUR", "totalValue": 5000,
                          "cash": {"availableToTrade": 5000, "reservedForOrders": 0}}
        elif path.endswith("/metadata/instruments"):
            value = INSTRUMENTS
        elif "/history/" in path:
            value = {"items": [], "nextPagePath": None}
        elif path.endswith("/orders/7"):
            value = {"id": 7, "instrument": {"ticker": submitted[0]["ticker"]},
                     "quantity": submitted[0]["quantity"], "side": "BUY", "status": "NEW"}
        elif path.endswith("/orders") and submitted:
            value = [{"id": 7}]
        else:
            value = []
        return httpx.Response(200, json=value)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(module, "Trading212Paper", lambda *args: Trading212Paper(*args, transport=transport))


def test_runner_dry_run_then_one_order_then_pending_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                           capsys: pytest.CaptureFixture[str]) -> None:
    module = load_runner(tmp_path, monkeypatch)
    submitted: list[dict[str, Any]] = []
    fake_broker(monkeypatch, module, submitted)
    signal = write_signal(tmp_path / "signal.json")
    base = ["paper_rebalance.py", "--signal", str(signal), "--watchlist", str(watchlist_file(tmp_path))]
    monkeypatch.setattr("sys.argv", base)
    assert module.main() == 0 and submitted == []
    assert json.loads(capsys.readouterr().out)["planned"]["ticker"] == "NVDA_US_EQ"
    monkeypatch.setattr("sys.argv", [*base, "--execute", "--demo-only"])
    assert module.main() == 0
    assert len(submitted) == 1 and submitted[0]["ticker"] == "NVDA_US_EQ"
    assert 0 < submitted[0]["quantity"] <= 0.1 * 200 / 200.0 + 1e-9  # 10% of EUR 200 at USD 200 and fx 1
    assert module.main() == 2 and len(submitted) == 1  # pending order blocks a second submission
    lines = [json.loads(line) for line in (tmp_path / "logs" / "decisions.jsonl").read_text().splitlines()]
    assert [line["action"] for line in lines].count("forecast") == 1  # logged once despite reruns
    assert len(list((tmp_path / "data" / "demo_execution").glob("forecasts-ai-*.logged"))) == 1


def test_runner_blocks_non_usd_weights_without_quotes_or_orders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                capsys: pytest.CaptureFixture[str]) -> None:
    module = load_runner(tmp_path, monkeypatch)
    submitted: list[dict[str, Any]] = []
    fake_broker(monkeypatch, module, submitted)
    monkeypatch.setattr(module, "load_demo_quotes", lambda *a: pytest.fail("quotes must not be fetched"))
    signal = write_signal(tmp_path / "signal.json", targets={"WORLD_ETF_A": 0.10}, reasons={"WORLD_ETF_A": "core"})
    monkeypatch.setattr("sys.argv", ["paper_rebalance.py", "--signal", str(signal), "--watchlist",
                                     str(watchlist_file(tmp_path)), "--execute", "--demo-only"])
    assert module.main() == 2 and submitted == []
    assert "WORLD_ETF_A" in json.loads(capsys.readouterr().out)["violations"][0]


def test_runner_all_cash_signal_completes_without_quotes_or_orders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                   capsys: pytest.CaptureFixture[str]) -> None:
    module = load_runner(tmp_path, monkeypatch)
    submitted: list[dict[str, Any]] = []
    fake_broker(monkeypatch, module, submitted)
    monkeypatch.setattr(module, "load_demo_quotes", lambda *a: pytest.fail("quotes must not be fetched"))
    signal = write_signal(tmp_path / "signal.json", targets={}, reasons={})
    monkeypatch.setattr("sys.argv", ["paper_rebalance.py", "--signal", str(signal), "--watchlist",
                                     str(watchlist_file(tmp_path)), "--execute", "--demo-only"])
    assert module.main() == 0 and submitted == []
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "completed" and output["violations"] == [] and output["targets"] == {}


def test_runner_rejects_invalid_signal_before_any_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    module = load_runner(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "Trading212Paper", lambda *a: pytest.fail("no broker call for a bad signal"))
    signal = write_signal(tmp_path / "signal.json", targets={"NVDA": 0.5})
    monkeypatch.setattr("sys.argv", ["paper_rebalance.py", "--signal", str(signal), "--watchlist",
                                     str(watchlist_file(tmp_path))])
    assert module.main() == 2
    assert "exceeds cap" in capsys.readouterr().err
