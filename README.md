# AI Trader

A risk-gated trading agent for a **broker demo account**. An LLM analyst proposes portfolio weights as a
JSON signal; a deterministic risk engine must approve every order **before it is placed through the
Trading 212 API**. The model never talks to the broker and cannot override a limit.

> Educational project and self-directed research tool. **Not financial advice.** Demo account only:
> there is no live-trading code in this repository. Results below are a historical backtest of a simple
> baseline, not a track record.

## Why it exists

LLMs are decent at digesting news, earnings dates and macro context, and bad at discipline. This project
keeps the part that benefits from judgement (what to look at, why, with what probability) separate from the
part that must be boring and strict (how much, how often, and whether an order is allowed to exist at all).

## How an order happens

```
LLM analyst (prompt, web research)
        |  writes JSON signal: weights, reasons, verdicts, scoreable forecasts
        v
1. Signal validation        ai_signal.parse_signal
        |  schema, freshness (<= 3 days), watchlist-only symbols, 15% cap per name,
        |  per-name EUR caps, <= 5 positions, no shorting or leverage,
        |  a written reason for every non-zero weight. Rejected, never repaired.
        v
2. Instrument resolution    ai_demo.tradable_usd_map
        |  exactly one USD-listed stock/ETF in the broker's instrument list, else refused
        v
3. Order planning           rebalance.plan_rebalance
        |  target shares from fresh quotes, sells truncated toward zero,
        |  per-order EUR cap: an oversized order is rejected, not resized
        v
4. Execution gates          demo_execution.execute_demo_plan
        |  demo URL only; EUR account; available cash + 1% fee reserve; pending orders block;
        |  daily-loss halt, drawdown kill switch, loss cooldown; quote freshness (15 min in
        |  US hours); position cap vs account equity; durable order journal written BEFORE the
        |  POST; denied/unknown orders block all runs until a human reviews; no automatic retries
        v
One DAY limit order (+/-1% of the quote) to the Trading 212 demo API, at most one per run
        v
Decision log (JSON lines) -> forecast calibration (Brier score) -> next run scores the last one
```

Everything above step 1 is the analyst; everything from step 1 down is plain, tested Python.

## Results

### LLM analyst: live track record (in progress)

The analyst's calls are executed by hand on a self-directed Revolut brokerage account (small EUR amounts,
fractional shares) and scored against what actually happened. Recording started 19 Oct 2026.

| Measure | Status |
|---|---|
| Scored forecasts (27 Oct 2026) | 3 scored: 1 hit, 2 partial, 0 miss. Too few to claim skill. |
| Return vs S&P 500 | Not reported yet. It will be computed from Revolut statement exports (real fills and cash flows) over the same dates as the benchmark. |

No performance claim is made until the statement-based figure exists. A few weeks on a small account says
nothing about skill.

### Appendix: deterministic ETF baseline (not the analyst)

| Series (2013-01 to 2026-10, costs included) | CAGR | Max drawdown | Sharpe |
|---|---|---|---|
| Baseline ETF momentum strategy (cap 15%) | 3.92% | -14.74% | 0.549 |
| SPY buy and hold | 14.66% | -33.72% | 0.904 |

This backtest covers the **baseline momentum strategy only**, not the LLM analyst, so it is not the analyst's
result. The baseline does not beat the benchmark. It uses one historical path, a surviving ETF universe and
assumed costs; see [reports/REPORT.md](reports/REPORT.md) for every limitation. No paper-trading fills are
recorded yet.

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows; use bin/activate on Linux/macOS
pip install -e ".[dev]"
pytest -q && mypy --strict src
```

Demo credentials come from environment variables only (names in `.env.example`, values never in files):
`TRADING212_DEMO_API_KEY`, `TRADING212_DEMO_API_SECRET`.

```bash
cp config/watchlist.example.yaml config/private_watchlist.yaml     # git-ignored
python scripts/paper_rebalance.py --signal examples/signal.example.json   # plan only, no order
```

`--execute --demo-only` submits one demo order. The analyst prompt and signal schema are in
[docs/analyst-prompt.md](docs/analyst-prompt.md); `examples/signal.example.json` is a template (set a
current `as_of`). Risk limits live in `config/risk_limits.yaml`.

## Status and limits

- Demo account only. Live trading is **not implemented**; the live-sleeve gate
  (`ai_signal.live_gate_blockers`) stays closed in code until a live adapter with persistent loss latches exists.
- The demo executor trades USD-listed stocks and ETFs only.
- Market data is Yahoo's unofficial chart endpoint: research use, not executable quotes.
- The LLM analyst itself is not bundled; the repo contains the prompt, the schema and the safety layer.
- 250+ automated tests; `mypy --strict` clean. Design details in [ARCHITECTURE.md](ARCHITECTURE.md).

## Legal note

Self-directed tool for the author's own demo account. It does not give personalised advice to others and
is not offered as a service. If you adapt it to manage money for anyone else, check the rules that apply
in your jurisdiction (in Germany/EU, e.g. BaFin and MiFID II).

## License

MIT, see [LICENSE](LICENSE).
