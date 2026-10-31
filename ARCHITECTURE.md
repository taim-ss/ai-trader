# AI Trader: architecture

An LLM proposes, plain code disposes, a human approves. This page describes what is implemented (checked against the code and its
tests) and what is deliberately not.

## Components

| Module | Responsibility |
|---|---|
| `ai_signal.py` | Signal schema, watchlist loader, `parse_signal`, `live_gate_blockers` |
| `ai_demo.py` | Glue from a validated signal to executable instruments (unique USD-listed stock/ETF only); demo and live signal loaders |
| `risk.py` | Limit model, target and order validation, kill-switch predicate |
| `execution_state.py` | Persistent risk ledger (daily loss, drawdown latch, cooldown), run lock, atomic writes |
| `demo_execution.py` | `execute_plan`: refresh, reconcile, validate, journal intent, send one DAY limit order; wrappers `execute_demo_plan` and `execute_live_plan` |
| `approval.py` | Shows the exact order; `run_with_decider` sends only on an explicit approval (typed phrase or phone tap) |
| `phone_approval.py` | ntfy proposal with Approve/Reject buttons, HMAC-signed order-bound tokens, polling for the tap |
| `paper_adapter.py` | `Trading212Paper` (demo URL only) and `Trading212Live` (live URL only, limit orders only) |
| `demo_market.py`, `data.py` | Quotes and cached adjusted closes (Yahoo chart endpoint; research use, not executable quotes) |
| `rebalance.py`, `portfolio.py`, `profile.py`, `diagnosis.py` | Planning from weights to orders, holdings, intake, concentration diagnosis |
| `strategy.py`, `backtest.py` | Deterministic 12-1 momentum baseline and its backtest (context only) |
| `calibration.py`, `decision_log.py` | Explicit binary forecasts, Brier scoring, append-only JSON Lines log |
| `arena.py` | Forward-test scoring: horizon returns, exposure-matched excess, random portfolios, bootstrap interval, pass criteria |

## Pipeline

```
analyst signal (JSON) -> parse_signal -> instrument resolution -> plan_rebalance -> execute_plan (dry run)
   -> approval (phone tap | typed phrase) -> execute_plan (send, same order) -> journal -> reconcile next run
   -> decision log / calibration / arena
```

## Execution gate (`execute_plan`)

Runs under a file lock and in this order, failing closed with a sanitised message at any step:

1. HALT file absent (checked here and again immediately before the POST).
2. Environment matches the broker exactly (demo or live). Live additionally requires an empty list of gate blockers supplied by the
   caller, and a human-approved order for any submission.
3. Plan validity: finite numbers, unique tickers, at most four decimals, limit notional within 2% of the planned value.
4. Account snapshot: EUR account, non-negative cash, no pending orders or reserved cash.
5. Risk ledger: daily-loss halt, drawdown latch, loss cooldown, external cash flows require a manual re-baseline.
6. Journal reconciliation: earlier accepted orders are matched against the broker; `intent`, `unknown` or `denied` records
   block everything until a human reviews them.
7. Order checks: unique stock/ETF instrument, quantity limits, per-order cap including a 1% fee reserve, cash, position-weight cap.
8. Quote freshness immediately before sending (15 minutes in US regular hours, otherwise at most four days for queued DAY limits).
9. The order about to be sent must equal the approved order (ticker, quantity, limit price).
10. Durable intent record, then exactly one POST; a failure is never retried automatically.

## Approval

- **Typed:** the exact order is printed; only the configured phrase sends it, and it refuses without a real terminal.
- **Phone:** the bot publishes the order to a private ntfy topic. The two buttons POST `APPROVE|REJECT <order-id> <expiry> <hmac>` to
  a second private topic that the bot polls. The HMAC covers decision, order id, ticker, quantity, limit price and expiry.
  Wrong key, other order, swapped decision, extended expiry or an expired token never validates. Publish failure raises, so no
  proposal means no order. Polling failures are retried until expiry and never count as approval.

## Live gate

`config/private_live.yaml` (git-ignored; template in `config/live.example.yaml`) must have `enabled: true`, `mode: live-with-approval`,
a valid `not_before`, a positive `sleeve_eur`, and every checklist item set to `true` by the owner. `live-auto` is rejected by the
gate itself. The code provides the live client and gated executor; whether to open the gate is the owner's decision and is
recorded in that file, not in code.

## Forward-test arena

LLMs cannot be backtested honestly on past dates, so contestants (the analyst, challenger prompts, and automatic baselines:
cash, index, equal weight, the deterministic momentum rule) commit weights before a session opens and are scored on closes that did
not exist yet. Rules, all enforced in code:

- A round opens only on a *completed* close; entries close when the next session opens.
- Returns are net of assumed round-trip costs at 5 and 20 sessions.
- The headline metric is **exposure-matched excess return** against SPY: gross return minus exposure x SPY, so holding cash is not
  mistaken for skill.
- 500 random portfolios per round give a luck distribution; each contestant gets a percentile rank.
- Verdicts (`NO EVIDENCE YET` / `PASS` / `FAIL`) use criteria fixed in advance in the arena config: minimum rounds, mean excess above
  zero, bootstrap interval above zero, average rank against random.

## Data and limits

- Market data: Yahoo's unofficial chart endpoint; it can lag hours behind a close, so the arena and runners fail closed on stale data.
- The analyst/scanner prompt and the personal watchlist are outside this repository by design; `docs/` contains generic templates.
- Orders are DAY limit orders about 1% beyond the quote; they can fill, partly fill, or expire.
- Only USD-listed stocks and ETFs are executable today; EUR-listed instruments are refused, not silently dropped.
- No performance claim is made. See the README for status and the baseline backtest.

## Verify locally

```
python -m pytest -q tests
python -m mypy --strict src
python scripts/run_backtest.py --offline
python scripts/arena.py board
```
