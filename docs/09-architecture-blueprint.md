# Architecture blueprint — Read & Write wiring

This doc pins down how the **Read** side (data ingestion) and the internal **Write** side (paper trading engine) connect. The paper ledger is the product; ingestion is pluggable.

See [API comparison](11-api-comparison.md) for rate limits and blockers.

---

## 1. The Read Connection (Data Ingestion)

### Default: yfinance

- Pull latest quotes / OHLCV for configured symbols (watchlist per run).
- Persist normalized `Snapshot` rows for replay and comparison.
- Phase 0 regime feature is computed from **yfinance daily history** on a liquid benchmark (e.g. BTC-USD), independent of memecoin polls.

### Optional: DEX Screener (memecoin mode)

- **Base URL:** `https://api.dexscreener.com`
- **Primary endpoint:** `GET /latest/dex/tokens/{tokenAddress}`
- **Method:** async `GET` in a loop, every **15–30 seconds** (`httpx.AsyncClient`).

#### JSON fields to extract (DEX mode)

Parse `pairs` — prioritize `pairs[0]`:

| Field | Purpose |
|---|---|
| `pairAddress` | Uniquely track the market |
| `priceUsd` | Current execution price |
| `liquidity.usd` | Slippage + liquidity floor |
| `volume.h1` (or `volume.m5`) | Momentum signal |

> [Liquidity floor](03-risk-guards.md) (`liquidityUsd < 50_000` → drop) applies before storage.

### Later: TradingView (UI only)

Charts consume bars from an **internal database** filled by yfinance (and optional DEX snapshots rolled into bars). TradingView does not replace ingestion—see [future work](10-future-work.md).

---

## 2. The Paper Trading Connection (Internal Write Engine)

There is **no external write API.** The app implements a local mock exchange.

### Local database state

SQLite first (see [stack](02-stack-and-layout.md)):

**Balances** — `USD_balance`, token balances per tracked symbol/address.

**Trade ledger** — timestamp, instrument id, `BUY`/`SELL`, amount, execution price, fees.

Maps to SQLAlchemy: `Run`, `Snapshot`, `Trade`, `Metric`.

### Execution logic flow

1. **Signal** — baseline or agent intent ([agent contract](04-agent-contract.md)).
2. **Price check** — latest price from the active ingestion provider (yfinance quote or DEX `priceUsd` + `liquidity.usd`).
3. **Slippage** — order size vs. liquidity (DEX mode); base fee + slippage from [risk guards](03-risk-guards.md) (0.3% fee, 0.5% baseline slippage).
4. **State update** — adjust balances; long/flat only.
5. **Log** — intent vs. enforced action.

All [risk guards](03-risk-guards.md) wrap this flow before any state change.

---

## 3. The Complete Loop

1. **Fetch** — yfinance or DEX Screener per run config.
2. **Feed** — truncated metrics + `regime_feature` from Phase 0.
3. **Receive** — baseline decision or strict-JSON agent decision.
4. **Execute** — update local ledger only.

No wallets, no keys, no web3 SDK — consistent with [scope](01-scope.md).
