# Phase 1 — Ingestion + paper ledger + closed loop

Goal: prove the whole sense → reason → act loop end-to-end **without** the LLM. The "decider" in this phase is the baseline regime strategy.

## `services/ingestion.py`

Pluggable provider (config / run body). **Default: `yfinance`.** Optional: `dexscreener` for memecoin runs.

### Mode: `yfinance` (default)

- Poll a configured watchlist of liquid symbols at a configurable interval (default **15s**, range 15–30s).
- Map each poll to a normalized `Snapshot`: symbol, `price`, volume if available, timestamp.
- Regime feature still comes from Phase 0 history on the benchmark symbol (e.g. BTC-USD), not necessarily the traded symbol.

### Mode: `dexscreener` (optional)

- Poll DEX Screener at the same interval.
- Primary endpoint: `GET /latest/dex/tokens/{tokenAddress}` — parse `pairs`, prioritize `pairs[0]`. See [architecture blueprint](09-architecture-blueprint.md).
- Truncate each response to:
  - `pairAddress`
  - `priceUsd`
  - `liquidity.usd`
  - `volume.h1` (or `volume.m5`)
  - `priceChange5m`, `priceChange1h` (when present)
- Apply the **liquidity floor** here (drop tokens below threshold before storing).

**Do not use DEX Screener as the source of daily bars for regime training**—that remains yfinance in Phase 0.

## `services/ledger.py`

- Paper account with starting cash (default **$1,000**), positions, and a `fills` table.
- Apply the **cost model** and **allocation cap** on every fill.
- Track **realized** and **unrealized** PnL.

## `models/`

- `Run` — includes `ingestion_provider` (`yfinance` | `dexscreener`)
- `Snapshot` — a polled metric set
- `Trade` — both **intent** and **enforced** action
- `Metric`

## `api/routes/runs.py`

- `POST /runs` — start a run; body picks `strategy = baseline | agent` and `ingestion_provider` (default `yfinance`).
- `GET /runs/{id}` — status.
- `GET /runs/{id}/trades` — trade log.
- The polling loop runs as a **background task**; it must stop cleanly.

## Decider for Phase 1

`services/baseline.py` — pure regime → position, **no LLM**. This proves the loop cheaply.

## Acceptance check (Phase 1)

- Start a run via the API (either ingestion mode).
- Watch it poll.
- See the baseline open/close simulated positions with costs applied.
- Read back the trade log and ending PnL via the API.

No LLM involved yet. **Stop and show a run before starting Phase 2.**
