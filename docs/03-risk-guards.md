# Risk guards (non-negotiable)

These live in `services/guards.py`. They are **hardcoded**. The LLM output is clamped to them after the fact — the agent never sees them as negotiable and can never widen them. Each is a config value with the defaults below.

## Guards

### 1. Max allocation cap
- No single order may exceed **20% of current cash**, regardless of the agent returning `percentage: 100`.
- Clamp the executed size, do not reject.

### 2. Liquidity floor
- Before a token is ever shown to the agent, drop it if `liquidityUsd < 50_000`.
- Illiquid tokens never reach the reasoning layer.

### 3. Transaction cost model
- Every simulated fill applies a configurable cost to the executed price:
  - default fee: **0.3%**
  - default slippage: **0.5%**
- This is **mandatory** — a backtest without costs is a lie.
- PnL must be reported **net of costs**.

### 4. Position bounds
- Ledger can never go negative cash.
- No shorts in v1 — long/flat only.
- Reject any decision that would breach this; log the rejection.

## Logging

Every clamp or rejection must log:
- the **original agent intent**, and
- the **enforced action**.

That log is part of the final writeup — it shows how often and how badly the agent tried to violate the guards.
