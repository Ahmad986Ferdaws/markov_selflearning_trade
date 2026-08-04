# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); the project is a portfolio /
research artifact rather than a released package, so versions are snapshots.

## [0.4.0] — 2026-07-28 — Kalman pairs research framework

### Added ([docs/20](docs/20-kalman-pairs.md))
- **`kalman/` package + `kalman-cli`** — a rigorous pairs-trading research harness built to
  a 14-section external spec: Joseph-form dynamic-hedge filter (pre-update-innovation
  signal, PSD-guarded, no inversions, separate β/α process noise, missing-obs explicit),
  fixed + adaptive Q (lagged vol, clipped, causal reference), dt-aware local-linear-trend
  module, training-only likelihood calibration with sensitivity surface, Engle–Granger/
  half-life pair qualification, hysteretic state machine with safety gates, causal health
  monitoring (NIS/CUSUM/β-uncertainty), and an explicit **two-leg cash-and-holdings
  ledger** (next-open fills, costs on traded notional, borrow, freeze/rehedge) — the
  `position*spread.diff()` shortcut is banned and its absence test-enforced.
- **Walk-forward** (train → predeclared validation → untouched test, 15 folds / 20y) versus
  static OLS, rolling OLS, EW regression, and cash through the identical engine.
- **All 16 mandatory acceptance criteria** from the spec map to tests (26 new; suite: 90).
  Validated at exact specification: ~95% coverage of the filter's own uncertainty.

### The finding (honest)
- SPY/QQQ fails cointegration qualification upfront (EG-ADF p ≈ 0.21–0.63, half-life ~195d).
- Walk-forward verdict: **no robust edge after costs** (best variant Sharpe 0.09, ≈0%
  total test return). The machinery is measurably correct; the edge is absent — reported
  plainly, exactly as this repo's charter demands.

## [0.3.0] — 2026-07-27 — the anatomy of zero

### The finding, dissected ([docs/19](docs/19-anatomy-of-zero.md))
- **Positive controls (in CI):** the engine detects planted first-order switch structure
  (alternation +90pp, cycles, noisy alternation) and reproduces the exact zero-edge identity
  on sticky synthetic chains — the market zeros are measurements, not malfunctions.
- **Switch-day anatomy + Gate 7:** new engine metrics (`n_switch`, `switch_recall`,
  `switch_attempts`); BTC's 90.9% = 298/298 stay days + **0/30 regime-change days**; a
  seventh honesty gate states it in every report.
- **Per-step dominance:** the causal transition row used at every one of **16,773** held-out
  decisions had self-probability ≥ **0.600** — the argmax is forced to persistence at each
  step; with 0 divergences the 95% bound on the true divergence rate is **< 0.018%**.
- **The obvious fixes fail:** a causal duration-hazard predictor fires 23 times in 16,773
  days; a second-order-memory predictor fires 20 times; edges uninformative both ways.
- **An honest positive:** the transition matrix beats persistence-equivalent probabilistic
  baselines on log-loss **28/28 runs** (+0.029 nats/day vs the stronger baseline) —
  calibration skill that never once crosses the decision threshold.
- **Independent re-derivation** (`scripts/independent_rederivation.py`, zero app imports,
  in CI): reproduces 328 predictions / 30 switches / 0 attempts / 298 hits from the raw
  snapshot alone.
- All claims adversarially verified (statistician + hostile code review + replication);
  9 wording overclaims and 8 code findings caught and applied. Tests: **64 passing**.

## [0.2.0] — 2026-07-27 — the null survives 20 years

### The finding, extended
- **Long-history study** (`scripts/long_history_study.py`, [docs/18](docs/18-long-history.md)):
  same protocol on the max Yahoo span per asset (up to 20 years — through 2008, COVID, 2022).
  Edge vs persistence **+0.0000 on 8/8 assets**, **0 divergences in 11,348 predictions**
  (combined identity receipt now **0 / 16,773**). Retires the "one ~3-year window" limitation.
  Confirms the "beats buy-hold" caveat: on uptrending long windows the baseline loses to
  buy-and-hold on 7/8 assets.

### Added
- **Fail-closed agent guards:** hard call-budget ceiling (`AGENT_MAX_CALLS`, default 64) and
  cache-only replay mode (`--cache-only` / `AGENT_CACHE_ONLY`) — a run can never fan out
  unbounded billed calls, and a public surface can never trigger a live one.
- **`diagnose-cli`:** offline narration of any run record (edge vs the persistence bar, policy
  outcomes net of costs, every honesty-gate warning). Reads only the computed JSON.
- **CI** (`.github/workflows/ci.yml`): full suite + a determinism grep asserting the
  90.9% ≡ 90.9% tie reproduces from the pinned snapshot on every push.
- **Website refinement pass:** semantic landmarks + strict heading outline, scroll-spy nav,
  keyboard-operable equity chart, AA contrast, ≥44px touch targets, 375px fixes; disclosure
  card and footer updated with the 20-year receipt and the repo Source link.
- Tests: **58 passing**.

## [0.1.0] — 2026-06-23 — portfolio snapshot

The honest evaluation harness and its verified null result.

### The finding
- **Markov regime model has no edge over persistence.** On held-out BTC-USD (n = 328), model
  hit-rate **90.9% ≡ persistence 90.9%**. Cross-asset, the edge is **+0.000 on 20/20 assets**,
  re-verified as a pointwise identity (**0 divergences / 5,425 predictions**). Net of costs both
  trading policies lose (baseline −41.1%, buy-and-hold −47.7%).

### Added
- **Phase 0 — regime module:** Markov transition matrix, z-score regime labels, deterministic
  walk-forward (`app/services/regime.py`, `regime-cli`).
- **Reframe Phase A — daily evaluation:** causal walk-forward, train/test split, accuracy vs
  **naive and persistence** baselines, buy-and-hold yardstick, honesty gates as code
  (`app/services/evaluation.py`, `evaluate-cli`). Passed an adversarial 4-auditor review.
- **Reframe Phase B — one engine + reproducibility:** pluggable `PolicyContext → position`
  policies through a single fill/cost path; data pinned + hashed (`app/services/data_cache.py`);
  deterministic run records in `results/`.
- **Phase C — LLM agent foundation (offline):** agent as a third pluggable policy; provider switch
  `anthropic | ollama | none` (keys from env only); prompt-hash response cache
  (`app/services/agent_policy.py`).
- **Phase C — honesty hardening:** per-decision provenance histogram (`AgentStats`), a sixth gate
  warning on near-uniform forecasts, and enforcement tests (CLI-only surface, env-only keys,
  `day_index` invariance) in `tests/test_agent_invariants.py`.
- **Cross-asset robustness study:** 20 assets + 4 sub-periods (`scripts/robustness_study.py`,
  `results/robustness/`, `docs/16`).
- **Deep review + landscape positioning:** `docs/17`; SRI hashes on CDN scripts, hardened reply
  parser, `requirements.lock` (77 pins).
- **Website:** single-file animated site in the Robinhood 2024 brand language with the **real**
  held-out equity series (`design/`), plus proof media (`media/`).
- **Tests:** 52 passing.

### Quarantined
- Legacy 15-second intraday loop (`comparison.py`, `runner.py`) carries deprecation headers; the
  canonical path is the daily evaluation harness.

### Pending
- Live Phase C agent row (gated on a reachable model — Ollama or `ANTHROPIC_API_KEY`).
