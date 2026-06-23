# Scope & boundaries

## Product focus (v1)

The product is a **paper trading application** (ledger, guards, simulated fills, optional API/CLI)—not a DEX Screener integration. Market data is **pluggable**; see [API comparison](11-api-comparison.md) and [architecture](09-architecture-blueprint.md).

## In scope (v1)

- **Paper ledger** with cash, positions, fills, realized + unrealized PnL (core deliverable).
- **Regime & history:** deterministic Markov model trained on daily bars from **yfinance** (default: liquid major e.g. BTC-USD).
- **Live ingestion (choose one mode for Phase 1):**
  - **`yfinance`** — poll a small watchlist of liquid symbols (equities/crypto quotes Yahoo supports).
  - **`dexscreener`** — poll latest pair metrics for memecoin/token addresses (discovery + live state; not the training history source).
- Hardcoded risk guards (allocation cap, liquidity floor, transaction costs, position bounds).
- LLM agent (Claude) returning a strict JSON decision.
- Baseline (non-LLM) regime strategy.
- Comparison harness running both strategies on the same snapshot stream.
- Minimal FastAPI + CLI to start runs and inspect results.

## Data providers — allowed vs excluded

| Allowed (v1) | Role |
|---|---|
| **yfinance** | Phase 0 regime, daily history, default live quotes for majors |
| **DEX Screener** | Optional live memecoin/pair snapshots only |
| **TradingView Charting Library** | Post–v1 UI; datafeed served from **internal** bar DB |

| Excluded | Reason |
|---|---|
| **Binance** and similar **Asia-centric exchange** REST/WebSocket APIs | Project policy: prefer Yahoo / DEX Screener / TV stack |
| On-chain routers, wallets, chain SDKs | Simulation-only v1 |

Optional later (not v1): US vendors **Polygon.io**, **Alpha Vantage** (paid or cached)—see [API comparison](11-api-comparison.md).

## Out of scope (v1) — do not implement

- Any on-chain execution.
- Jupiter, 1inch, or any DEX router integration.
- Wallet management or private-key handling.
- `web3.py`, `solana-py`, or any chain SDK.
- Real funds of any kind.
- Binance (and comparable Asia-centric exchange) market-data clients in this codebase.

**If a task seems to require any of the above, STOP and flag it. Do not implement.**

## Secrets

- No secrets in code.
- All API keys come from environment variables only (e.g. Anthropic; optional Polygon/Alpha Vantage later).
