# Build order & acceptance gates

Build the project **phase by phase**. Do not jump ahead. After each phase, stop, run the acceptance check, update [STATUS](STATUS.md), and wait for confirmation before continuing.

## Order

1. **[Phase 0](05-phase-0-regime.md)** — Regime module via yfinance, standalone, proven on BTC-USD with costs.
2. **[Phase 1](06-phase-1-loop.md)** — Pluggable ingestion + ledger + guards + API, baseline only.
3. **[Phase 2](07-phase-2-comparison.md)** — LLM agent + fair comparison harness.
4. **Tests** alongside each phase — override `get_db`, mock http + agent clients.
5. **README + writeup** of the comparison result with setup and env vars.

Reference: [API comparison](11-api-comparison.md) for provider limits; [STATUS](STATUS.md) for session handoff.

## Acceptance gates (summary)

| Phase | Gate |
|---|---|
| 0 | Transition matrix + stationary distribution + net-of-cost BTC-USD backtest printed; cost model measurably lowers Sharpe. |
| 1 | API starts a run; baseline trades with costs applied; trade log and PnL readable via API. No LLM yet. |
| 2 | One command runs baseline + agent on identical snapshot stream and prints the comparison table. |

## Rule

**Do not start Phase 1 until Phase 0's acceptance check passes. Confirm each phase before moving on.**
