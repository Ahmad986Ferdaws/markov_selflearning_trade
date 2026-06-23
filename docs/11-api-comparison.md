# API comparison — paper trading & ML backtest

Research for a **paper-trading** app with no on-chain execution. Covers DEX Screener REST, TradingView Charting Library Datafeed responsibilities, and OHLCV sources suitable for this project: **yfinance**, Alpha Vantage, Polygon.io. (Binance is documented for completeness but **out of project scope** — see [scope](01-scope.md).)

## Overview

- **DEX Screener** — public REST for latest token/pair discovery and screening.
- **TradingView Charting Library** — frontend chart; **you** implement the Datafeed API and supply bars.
- **yfinance** — primary prototype source for daily history and liquid symbols (Yahoo Finance).
- **Alpha Vantage / Polygon** — optional US vendors; free tiers are rate-limited.

v1’s main challenge is a **stable, reproducible OHLCV path** for regime training and backtests—not chart widgets alone.

## DEX Screener REST API

- **Base:** `https://api.dexscreener.com`
- **Rate limits (documented):** ~60 req/min (profile/meta-style); ~300 req/min (pair/search-oriented).
- **Use here:** live pair snapshots, memecoin watchlists, liquidity for guards—not canonical daily OHLCV history.
- **Blocker for Markov training:** no revision-stable historical daily-bar archive.

### Main endpoints

- `GET /latest/dex/pairs/{chainId}/{pairId}` — latest pair state (price, liquidity, volume, changes).
- `GET /latest/dex/search?q=...` — search current listings.
- `GET /token-pairs/v1/{chainId}/{tokenAddress}` — pairs for a token.
- `GET /tokens/v1/{chainId}/{tokenAddresses}` — token lookup.
- Profile/meta/trending endpoints for discovery.

**Auth:** no API key on public endpoints in official docs.

## TradingView Charting Library Datafeed API

TradingView does **not** provide market data. Implement `onReady`, `searchSymbols`, `resolveSymbol`, `getBars`, `subscribeBars`, `unsubscribeBars` (and quotes if using the trading terminal).

**Implementation notes:**

- Callbacks should be async.
- Daily/weekly/monthly bar `time` = start of UTC day; bars in **ascending** order.
- `getBars` + `countBack` can cause repeated history requests if the backend under-delivers.
- **Best pattern:** normalize bars into an internal DB; serve TV and ML from the **same** store.

**Blocker for ML:** backend correctness (timezone, bar timestamps, revisions)—not a TV rate limit.

## OHLCV providers

### yfinance (v1 default for regime & majors)

- Unofficial Yahoo Finance wrapper; convenient `download()` / `Ticker.history()`.
- No clear official quota—treat as best-effort.
- **Blocker:** licensing/reliability vs paid vendors—not integration difficulty.

### Alpha Vantage

- Free tier often cited as **5 req/min, 25 req/day**.
- **Blocker:** free quota too small for multi-symbol training without heavy caching.

### Polygon.io

- US account-based API; free tier often ~**5 req/min** on aggregates.
- **Blocker:** free-tier throughput; better when paid.

### Binance (reference only — not used in this repo)

- Strong crypto OHLCV + websockets; documented limits (e.g. connection caps per IP).
- **Excluded** from this project by product choice (prefer Yahoo / DEX Screener / TradingView stack). See [scope](01-scope.md).

## Comparison table

| Source | Role | Auth | Practical limits | Daily-bar ML | Main blocker |
|---|---|---|---|---|---|
| DEX Screener | Live pair discovery | None in docs | 60–300 rpm by endpoint family | Poor (no OHLCV archive) | Not a history warehouse |
| TradingView Datafeed | Chart contract | Your backend | N/A (no market data) | Good if backed by internal bars | Backend correctness |
| yfinance | History + liquid symbols | None typical | Unofficial / best-effort | Good for prototypes | Reliability & licensing |
| Alpha Vantage | Keyed REST | API key | 5/min, 25/day free | Weak at free tier | Quota |
| Polygon.io | Commercial bars | API key | ~5 rpm free (aggregates) | Good when paid | Free-tier narrowness |
| Binance | *(out of scope)* | Often none for public | WS connection limits | Strong technically | Excluded by project policy |

## Recommended architecture (this project)

```text
yfinance (regime + liquid symbols)
    → normalize + cache → internal bars DB
        → paper ledger + ML backtest
        → (later) TradingView Datafeed adapter

Optional: DEX Screener for live memecoin/pair snapshots only
```

- **Phase 0:** yfinance on BTC-USD (or other liquid Yahoo symbol).
- **Phase 1:** `ingestion` provider = `yfinance` (default) or `dexscreener` (memecoin mode); see [Phase 1](06-phase-1-loop.md).
- **Post–v1 UI:** TradingView charts read from the same internal bar store—see [future work](10-future-work.md).

Optional upgrades: Polygon or paid Alpha Vantage if Yahoo becomes insufficient. Do **not** depend on Binance or other Asia-centric exchange APIs in this codebase.

## Perplexity validation (summary)

Independent research confirmed: **v1 stack is sufficient** for simulation-only paper trading. Gaps are internal (normalization, paper-fill engine, guardrails, snapshot replay comparison)—implemented in `app/`. yfinance has no official SLA; DEX Screener is live-only; Claude limits are org-specific via Anthropic Rate Limits API; TradingView requires a self-hosted datafeed backed by the same bar DB as ML.
