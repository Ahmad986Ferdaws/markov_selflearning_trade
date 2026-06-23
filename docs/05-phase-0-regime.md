# Phase 0 — Regime module (DAY-1 DELIVERABLE)

Build `services/regime.py` **first**, as a self-contained module with its own CLI entry point, provable before anything else exists.

It does double duty later: it is both the **baseline strategy** and a **feature provider** for the LLM agent.

## Why prove it here

Memecoins lack the history for reliable transition-matrix estimation. Use a liquid major with deep history so the math is trustworthy before plugging it into anything else.

## Data

- Fetch **daily** history for a liquid major (default: **BTC-USD**, ~2–3 years).
- **Source:** `yfinance` (Yahoo Finance)—primary v1 path per [API comparison](11-api-comparison.md).

## Required functions

- `define_states(returns, window=20, bull_thresh, bear_thresh)` → 3 states (**Bull / Bear / Sideways**), mutually exclusive and exhaustive.
  - `window` is **configurable** (default `20` trading days). Use a shorter window (e.g. `10`) only after validating stability on the acceptance backtest.
- `estimate_transition_matrix(states)` via MLE: `count(i→j) / count(i→any)`. Rows sum to 1.0.
- `forecast(P, current_state, n_steps)` via `np.linalg.matrix_power` (Chapman–Kolmogorov).
- `stationary_distribution(P)` by solving `πP = π`.
- `walk_forward_backtest(...)` — re-estimate `P` at each step using only past data (**no lookahead**). Return per-step regime + one-step forecast vector `{bull, bear, sideways}`.
- `regime_feature(history) -> {state, p_next}` — exposed for the agent layer.

## Costs & metrics

- Apply the transaction-cost model from the [risk guards](03-risk-guards.md) to the backtest.
- Report **net-of-cost**:
  - Annualized Sharpe
  - Max Drawdown
  - Annual Return
  - Regime mix (% of time in each state)

## Data sufficiency

- Warn if any transition-matrix cell was estimated from **< 20–30 observed transitions**.

## Acceptance check (Phase 0)

Running the module prints:
- a transition matrix,
- a stationary distribution,
- a net-of-cost backtest summary on BTC-USD.

Numbers must be plausible (Sharpe is modest, not absurd). The cost model must measurably lower Sharpe vs. a no-cost run.

**Stop here and show the output before starting Phase 1.**
