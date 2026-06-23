# Stack & target layout

## Stack

- **Python 3.11+**
- **FastAPI** — async, application-factory pattern (`create_app`)
- **httpx.AsyncClient** for outbound HTTP to DEX Screener when that provider is enabled
- **yfinance** for historical and live quote pulls (Phase 0 default; Phase 1 `yfinance` mode)
- **SQLite** first (zero-setup, fast to demo), behind a repository interface so it can be swapped to Postgres later without touching business logic
- **Pydantic v2** for all request/response/config models
- **Anthropic API** (Claude) as the LLM — model + key from env
- **numpy / pandas** for regime math
- **pytest** for tests

## Conventions

- Use dependency injection for DB sessions and external clients.
- Keep route handlers thin; business logic lives in `services/`.
- Override `get_db` and mock http + agent clients in tests.
- **Ingestion** implements a small provider interface (`yfinance` | `dexscreener`); config selects mode per run.

## Target directory layout

```text
app/
|-- main.py            # create_app factory, router registration, lifespan
|-- config.py          # Settings: API keys, poll interval, ingestion provider, regime window
|-- dependencies.py    # DB session, http client, agent client providers
|-- exceptions.py      # ApiError + handlers
|-- api/routes/
|   |-- health.py
|   |-- runs.py        # start/stop a run, list trades, fetch comparison report
|-- services/
|   |-- ingestion.py   # pluggable: yfinance (default) + optional DEX Screener
|   |-- regime.py      # Markov transition matrix, walk-forward, signal  <-- PHASE 0
|   |-- agent.py       # Claude call, strict JSON parsing, retry/repair
|   |-- guards.py      # hardcoded risk rules — AGENT CANNOT OVERRIDE
|   |-- ledger.py      # paper trading: cash, positions, fills, PnL
|   |-- baseline.py    # non-LLM regime strategy (the benchmark)
|   |-- comparison.py  # run both strategies, compute metrics, build report
|-- models/            # SQLAlchemy models: Run, Trade, Snapshot, Metric
|-- schemas/           # Pydantic models, incl. the AgentDecision contract
`-- tests/
```
