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

- **Qualification says no:** SPY/QQQ proper Engle–Granger p ≈ 0.43 on the train half
  (0.83 on fold 0) — the first release mislabeled a raw residual-ADF p ≈ 0.21 as "EG";
  the corrected statistic is *weaker* still. Residual half-life ≈ 194
  days — *no stationarity evidence at 5%*. The diagnostics correctly warn this classic
  "obvious" pair is a poor cointegration candidate before a single trade.
- The exact-specification test shows ~95% coverage of the filter's own uncertainty — the
  implementation is correct; any lack of edge is the strategy's problem, not the filter's.
- **Corrected walk-forward (post-review, 2026-09):** 15 folds / 20 years, all six variants
  under the now-truly-identical engine (β-weighted LOG sizing, symmetric signal-health
  gate, no per-fold warm-up re-blanking, predictive-likelihood calibration):

  | variant | mean test Sharpe | 90% CI (block bootstrap) | total test ret | trades | z-std |
  |---|---:|---:|---:|---:|---:|
  | static_ols | −0.01 | [−0.18, 0.57] | +3.1% | 75 | 1.32 |
  | rolling_ols | −0.77 | [−1.17, −0.42] | −18.8% | 64 | 1.37 |
  | ew_ols | −0.30 | [−0.56, 0.19] | −4.7% | 61 | 1.38 |
  | kalman_fixed | −0.08 | [−0.33, 0.35] | +0.3% | 23 | 0.55 |
  | kalman_adaptive | −0.27 | [−0.63, 0.15] | −5.6% | 37 | 0.61 |
  | cash | 0.00 | — | 0.0% | 0 | — |

  Every CI straddles or sits below zero: **no robust edge after costs**, now with interval
  evidence rather than a bare point estimate. The z-std column discloses that the Kalman
  innovations run at ~0.55 realized std (its S_t over-estimates), so its fixed z-thresholds
  trade ~3× less often than the baselines' — a scale caveat the report prints rather than
  hides. The verdict is unchanged from the first release, but it is now earned under the
  corrected methodology instead of inherited from an asymmetric one.

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

## Adversarial review round (2026-09-08) — 12 findings, all fixed

A hostile logic review after release found one CRITICAL and several MAJOR defects; every
fix carries a regression test:

1. **CRITICAL — LOG sizing ignored β's magnitude** (only its sign reached the ledger, so
   dynamic-vs-static hedge ratios were never actually compared). Now a true elasticity
   hedge: `|notional₂| = |β|·|notional₁|`, gross-normalized.
2. **Health gate was Kalman-only** — baselines faced four gates, Kalman five. Replaced in
   the walk-forward by `health_from_signal`, an identical z-based causal gate per variant.
3. **Warm-up re-blanked every fold** (first 60 of 250 test bars structurally dead). The
   state machine now counts causal history (`_seen=start`).
4. **Calibration likelihood leaked** (OLS prior fitted on the window it was scored on).
   Now init on the first fifth, scored strictly after.
5. **"EG" p-values were raw residual-ADF** (anti-conservative). Now `statsmodels.coint`,
   with the naive number kept and labeled anti-conservative.
6. Audit-table index misalignment; missed-fill `beta_entry` desync in REHEDGE; trades
   undercounted on retried fills; unreachable cooldown branch + cooldown not ticking on
   gated bars; `tr0` ignored under rolling windows; bootstrap final-block exclusion;
   bare `assert` in the ledger — all fixed.

The verdict did not change; the *evidence quality* did.

## Known limitations (stated, not hidden)

Single-pair portfolio (multi-pair limits not yet aggregated); no dividend/total-return or
corporate-action adjustment beyond yfinance auto-adjust; no market-impact model beyond
linear bps; matplotlib report plots not generated (text report + audit table);
level-model sizing is share-based without a separate residual-dollar translation; deflated
Sharpe not implemented (the block-bootstrap CI now appears in every report); Kalman z-scale
mismatch (z-std ≈ 0.55) is disclosed but not re-calibrated away. None of these affect the
causality or accounting contracts, which are test-enforced.

## Reproduce

```bash
pytest tests/test_kalman_core.py tests/test_kalman_ledger.py tests/test_kalman_research.py -q
kalman-cli demo && kalman-cli replay && kalman-cli trend
kalman-cli pairs                # SPY QQQ (or: kalman-cli pairs GLD TLT)
kalman-cli walkforward          # full report with VERDICT
```
