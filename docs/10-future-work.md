# Future work (post-v1)

Ideas considered and **explicitly deferred** out of v1. Do not build these until the v1 acceptance gates in [build-order](08-build-order.md) have all passed.

## TradingView Charting Library — v2 visualization layer

TradingView's Charting Library is a **frontend JavaScript widget**. v1 is a headless Python service with no frontend, so it's deferred. When we add a UI, serve the Datafeed from an **internal bar DB** populated by **yfinance** (same store as ML backtests)—see [API comparison](11-api-comparison.md). Do not use Binance or Asia-centric exchange feeds in that pipeline.

### Why it doesn't fit v1

| TradingView expects | v1 produces |
|---|---|
| OHLCV bars (open/high/low/close/volume per interval) | Point-in-time `priceUsd` snapshots every 15s |
| Symbol search + resolution | Fixed set of token addresses per run |
| Real-time bar streaming | 15s REST polling |
| A browser frontend | No frontend in scope |
| Broker adapter for trade entry | Python paper ledger, no UI trade entry |

### What a v2 integration would require

Three pieces, in order of cost:

1. **Bar aggregator** — roll the existing `Snapshot` rows into 1m / 5m / 1h OHLCV bars on the fly (or on insert). New service, no change to ingestion or ledger.
2. **Datafeed HTTP adapter** — implement the [Datafeed API](https://www.tradingview.com/charting-library-docs/latest/api/modules/Datafeed/) shape in FastAPI:
   - `onReady` → static config (supported resolutions, exchanges)
   - `searchSymbols` → query tokens we've ingested
   - `resolveSymbol` → return `LibrarySymbolInfo` (24x7 session, UTC timezone, pricescale appropriate for the token)
   - `getBars` → query the aggregator; **ascending order**, **ms timestamps**, satisfy `countBack`, return `{ noData: true }` when empty
   - `subscribeBars` / `unsubscribeBars` → websocket push of the latest forming bar
3. **Broker adapter** *(only if we want clickable trading from the chart)* — implement `IBrokerTerminal` against the paper ledger, push fills back via `tradingHost.orderUpdate / executionUpdate / positionUpdate`. Read-only chart is fine without this.

### Gotchas to remember if/when we build it

- Datafeed callbacks must be **async** (the docs recommend `setTimeout(..., 0)`).
- `Bar.time` is **milliseconds** UTC; `PeriodParams.from/to` are **seconds**. Easy to mix up.
- `subscribeBars` can only update the latest bar or append a new one — never rewrite history.
- Wrong `session` / `timezone` / `pricescale` causes silently broken charts.
- Broker mode also requires **quote** methods on the datafeed.

### Read-only vs. trading-terminal

For just *visualizing* what the agent did, we only need pieces 1 and 2 plus chart marks (`getMarks` / `getTimescaleMarks`) to plot the agent's BUY/SELL events on the chart. That's the cheap path. The full Broker terminal is a much bigger lift and only worth it if a human is meant to trade alongside the agent — which isn't a v1 or v2 goal as currently scoped.

## Other deferred items

- **Postgres swap** — already designed for via the repository interface; flip when SQLite gets painful.
- **Multi-token portfolios** — v1 ledger is single-token-per-run by default; generalize when comparison harness is stable.
- **On-chain execution** — explicitly out per [scope](01-scope.md). Not a roadmap item without a separate, deliberate decision.
