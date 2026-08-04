# 20 — Kalman pairs-trading research framework (`kalman/`)

A rigorous, reproducible Kalman-filter research harness for dynamic pairs trading, with a
local-linear-trend filter as a separate secondary module. **Simulation and paper research
only** — no profitability claim, no broker integration, no live orders. Built to the same
standard as the rest of this repo: causal by construction, costs on real two-leg notional,
honest walk-forward, and a report that says "no edge" when that is the finding.

## Model

Dynamic hedge ratio (pairs): state `x_t = [beta_t, alpha_t]'`, random walk `F = I`:

```
x_t = x_{t-1} + w_t,          w_t ~ N(0, diag(q_beta, q_alpha))
y_t = [p2_t, 1] x_t + v_t,    v_t ~ N(0, R),      y_t = p1_t
```

Retained per bar (`StepRecord`): prior state/covariance, innovation `e_t`, innovation
variance `S_t`, standardized innovation `z_t = e_t/sqrt(S_t)`, Kalman gain, posterior
state/covariance, state intervals, predictive log-likelihood, NIS. **The trading signal is
the pre-update innovation** — the posterior (which has seen `y_t`) never forms the same
bar's signal.

Trend filter (secondary): `x = [level, velocity]`, `F = [[1, dt], [0, 1]]`, `H = [1, 0]`,
with the integrated continuous-time `Q(dt)` so irregular intervals scale correctly.
Velocity crossings are a *feature to test against MA baselines*, not an assumed edge.

## Numerical contract

Joseph-form update + symmetrization + PSD guard; no matrix inversions (scalar `S`);
`S_t > 0` verified every step; non-finite inputs raise; missing observations are explicit
predict-only steps; `q_beta` and `q_alpha` are separate (different units — `Q = qI` is not
assumed). Init modes: diffuse or training-only OLS; both pair with a no-trade warm-up.

Adaptive Q: `Q_t = Q_base * clip((sigma_{t-1}/sigma_ref)^gamma, m_min, m_max)` — realized
vol strictly lagged, reference from training data only, clipped, never backfilled.

## Timing convention (enforced structurally + by test)

```
observation(t) close  ->  signal(t) from PRE-update innovation  ->  order(t)
                      ->  execution at open(t+1)                 [same-close prohibited]
```

The ledger consumes `decisions[t-1]` at bar `t`'s open — bar-t information cannot touch
bar-t holdings even if the strategy layer misbehaves.

## Accounting (the only P&L source)

Explicit two-leg cash-and-holdings ledger: shares of both legs, marked at closes, filled at
next open. Costs on actually traded notional: commission + half-spread + slippage per side,
short borrow per bar, financing on negative cash. Hedge modes: FREEZE (beta fixed for the
trade) or REHEDGE (re-target beyond a threshold) — both charge full turnover. The shortcut
`position * posterior_spread.diff() / spread_std` is banned and structurally absent (tested):
a drifting posterior beta re-prices the past and books unobtainable P&L.

Price representations: LOG (default; beta = elasticity → dollar exposures, gross normalized
to `|L1| + |L2| = gross_target`, then shares) or LEVEL (beta = share ratio → shares). Never
mixed.

## Strategy & safety

Hysteretic state machine (`exit_z < entry_z` validated), emergency stop, max holding,
cooldown, warm-up; gates for stale data, extreme beta, invalid variance, and relationship
health. Health monitoring is unit-aware and causal — NIS windows, lagged |z| CUSUM, beta
uncertainty ratio — never the raw covariance trace (mixed units, price-scale dependent).

## Walk-forward protocol

Chronological folds: **train** (qualification + likelihood calibration on positive-parameter
grids) → **validation** (choose entry/exit from a small predeclared set) → **test** (once,
untouched) → roll. Variants compared under identical data, execution, and costs — a variant
can differ only in how it produces `(z_t, beta_t)`:
static OLS · rolling OLS · EW recursive regression · fixed-Q Kalman · adaptive-Q Kalman ·
cash. Metrics are frequency-aware (CAGR, vol, Sharpe/Sortino/Calmar, maxDD, turnover,
costs, trades); a moving-block bootstrap Sharpe CI is provided. Kalman optimality is
conditional on correct specification and **does not establish a trading edge**.

## Defaults (all configurable)

SPY/QQQ daily from this repo's pinned 20-year snapshots (offline, reproducible) · $100k
capital · gross target $100k, cap 2× · entry z 2.0 / exit 0.5 / stop 4.0 · warm-up 60 ·
commission 1bp + half-spread 2bp + slippage 2bp per side · borrow 25bp/yr · next-open fills.

## Findings (honest)

- **Qualification says no:** SPY/QQQ Engle–Granger ADF p ≈ 0.21, residual half-life ≈ 194
  days — *no stationarity evidence at 5%*. The diagnostics correctly warn this classic
  "obvious" pair is a poor cointegration candidate before a single trade.
- The exact-specification test shows ~95% coverage of the filter's own uncertainty — the
  implementation is correct; any lack of edge is the strategy's problem, not the filter's.
- Walk-forward results live in the CLI report (`kalman-cli walkforward`); see the VERDICT
  line, which is written to say "no robust edge after costs" when that is what happened.

## Mandatory-test mapping (16/16)

| # | Criterion | Test |
|---|---|---|
| 1 | Known drifting-beta recovery in bounds | `test_recovers_known_drifting_beta` (incl. 2-sd coverage) |
| 2 | Constant-state convergence | `test_constant_state_converges_and_stays` |
| 3 | Batch == one-at-a-time replay | `test_batch_equals_replay`, `kalman-cli replay` |
| 4 | Future data cannot change the past | `test_future_data_cannot_change_past` |
| 5 | Signal-to-execution lag | `test_one_bar_execution_lag` |
| 6 | Covariance symmetric PSD | `test_covariance_symmetric_psd` |
| 7 | `S_t > 0` | `test_innovation_variance_positive` |
| 8 | Hand-reconciled two-leg P&L/turnover/costs | `test_hand_reconciled_two_leg_pnl`, `test_hedge_modes_differ_and_both_pay` |
| 9 | Cost monotonicity | `test_cost_monotonicity` |
| 10 | Bad data rejected | `test_bad_data_rejected`, `test_missing_observation_explicit` |
| 11 | No backfill / centered windows / future params | `test_no_backfill_or_centered_windows_in_source`, `test_causal_z_uses_only_lagged_history`, `test_warmup_forces_flat` |
| 12 | Adaptive Q lagged + bounded | `test_adaptive_q_lagged_and_bounded` |
| 13 | Identical assumptions across variants | `test_variants_share_execution_engine`, `test_cash_reference_is_flat` |
| 14 | Reproducible randomness | `test_reproducible_synthetic` |
| 15 | Report regenerates | `test_walkforward_report_regenerates_identically` |
| 16 | Passing ≠ profitability | no test asserts positive P&L (grep the suite) |

## Known limitations (stated, not hidden)

Single-pair portfolio (multi-pair limits not yet aggregated); no dividend/total-return or
corporate-action adjustment beyond yfinance auto-adjust; no market-impact model beyond
linear bps; matplotlib report plots not generated (text report + audit table only);
level-model sizing is share-based without a separate residual-dollar translation; deflated
Sharpe not implemented (block-bootstrap CI is). None of these affect the causality or
accounting contracts, which are test-enforced.

## Reproduce

```bash
pytest tests/test_kalman_core.py tests/test_kalman_ledger.py tests/test_kalman_research.py -q
kalman-cli demo && kalman-cli replay && kalman-cli trend
kalman-cli pairs                # SPY QQQ (or: kalman-cli pairs GLD TLT)
kalman-cli walkforward          # full report with VERDICT
```
