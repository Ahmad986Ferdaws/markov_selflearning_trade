# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); the project is a portfolio /
research artifact rather than a released package, so versions are snapshots.

## [Unreleased] — repository hygiene, three small fixes, and a test-coverage batch, then a second batch of one hundred one-concern pull requests

The first set of small pull requests (one concern each) was opened 2026-11-02; the second,
one hundred pull requests #174–#273 each closing one of issues #64–#173, was opened 2026-11-07.
Entries below are grouped by batch.

### Fixed
- The yfinance provider turned a reported volume of `0.0` into `None` ("not reported")
  because it tested truthiness; it now keeps the zero, the same rule the price field
  already followed.
- `str(ApiError)` was an empty string (`Exception.__init__` was never called), so a
  traceback or log line for an unhandled one showed no message.
- The FastAPI app advertised version `0.1.0` in its OpenAPI document; it now reads
  `app.__version__`, and a test keeps `pyproject.toml`, the newest CHANGELOG heading, and
  the OpenAPI document in agreement.
- A stray product-symbol character at the end of the docs/13 title.

### Added
- `regime-cli --symbol / --window / --k / --fee-pct / --slippage-pct`, each overriding one
  setting for one invocation, with usage-error validation before any report runs.
- `scripts/independent_rederivation.py --snapshot PATH`, documented exit codes
  (0 reproduced, 1 mismatch, 2 snapshot changed), and a first output line naming the
  snapshot it scored.
- Repository files: pull-request and issue templates, `CODEOWNERS`, Dependabot for pip and
  GitHub Actions, `CONTRIBUTING.md`, `SECURITY.md`, `CITATION.cff`, `.editorconfig`,
  `.gitattributes` (binary pins and media, collapsed result diffs), an opt-in
  `.pre-commit-config.yaml`, a docs index (`docs/README.md`) and glossary
  (`docs/21-glossary.md`), project URLs / keywords / classifiers in `pyproject.toml`, and
  `robots.txt` plus a favicon cache header for the static site.
- CI: a read-only token, cancellation of superseded runs, a 15-minute job timeout, and a
  Python 3.11 + 3.12 matrix (3.11 is the floor `requires-python` declares; it had never run).
- Tests, 16 new files and roughly 120 tests, each pinning a boundary rather than re-running
  a study: risk-guard thresholds, the web-surface inventory (OpenAPI path tripwire), schema
  contracts, the baseline's three rules, data-hash sensitivity and the snapshot cache,
  regime-labelling boundaries, the Kalman data layer, the provider's price fallback chain,
  version consistency, scope tripwires (no exchange or wallet SDK imported, no
  credential-shaped literal tracked, no dead README doc link), the committed run records
  read back against the README/docs numbers (328 / 298 / 30 / 0; +0.000 on 20/20; 0 of
  11,348 divergences), ledger accounting identities, `ApiError` and `app.db`, the
  re-derivation script's exit codes, and the console-script entry points.

### Changed
- `.gitignore` covers the ruff, mypy, coverage, and hypothesis caches.
- pytest-asyncio's fixture loop scope is explicit, which removes the deprecation warning
  printed at the top of every run; ruff's target version is pinned to 3.11.

### Fixed (2026-11-07 batch)
- The legacy comparison appended its "Sharpe annualization factor unresolved" warning once per
  strategy; it now appears once per report (#175).
- `GET /runs/{id}` priced a position with the newest snapshot of *any* symbol when the trade
  symbol had none; it now reports no price (unrealized 0.0) instead of mixing instruments (#193).
- `CreateRunRequest` silently ignored unknown fields (a typo such as `stratgy` created a baseline
  run) and accepted blank or duplicate watchlist entries; both are now 422s (#202).
- `compare-cli <unknown id>` printed a traceback; it is now a usage error with the message on
  stderr and exit code 2 (#243).
- The Kalman walk-forward report claimed "test Sharpe is positive" when no complete fold fit the
  history; it now says that nothing was evaluated (#178). A test z-std of exactly 0.0 was treated as
  missing and hidden from the report; it is now shown and the far-from-1 caveat fires on it (#186).
- The four study scripts created `results/` directories as a side effect of being imported
  (#181, #184, #187, #192); `scripts/_verify_divergence.py` ran a 20-asset study on import
  and is now behind a `main()` guard (#203).
- `LLM_PROVIDER=" ollama"` (stray whitespace) disabled the agent as an unknown provider (#206).
- The `Typing :: Typed` classifier was declared without the `py.typed` markers it promises (#266).
- The static site's social-card image was a relative path that escaped the deploy root, so link
  previews rendered no image; it is now an absolute URL (#209).

### Added (2026-11-07 batch)
- `regime-cli --years N` selects the matching pinned snapshot, so the 20-year pins are reachable
  from the CLI (#224).
- A typed `ComparisonResponse` model on `GET /runs/{id}/comparison` (#198); indexes on the
  `run_id` foreign keys the legacy queries filter on (#226); an OpenAPI description and licence for
  the legacy app (#229).
- A warning when a corrupt agent response cache is discarded (#214); `load_pair_from_snapshots`
  names every missing pickle at once (#242).
- Repository files: folder READMEs for `design/`, `scripts/`, `results/`, `data/`, `media/` and
  `tests/` (#233, #244, #249, #254, #257, #259); a feature-request issue template (#176); a
  private-reporting link in the issue chooser (#179); `CODE_OF_CONDUCT.md` (#185); indentation
  rules for web files (#196); merge-conflict, JSON and line-ending pre-commit hooks (#201);
  `object-src 'none'` in the site's CSP (#213); glossary entries for data hash, stay-day credit,
  decision provenance, diagonal dominance, sub-period and positive control (#262); a complete
  commands block in `docs/STATUS.md` (#264); lint and single-file instructions in CONTRIBUTING
  (#267); a bot-review checkbox in the PR template (#273).
- Tests, 58 new files, each pinning a boundary: the accuracy and metric helpers,
  `evaluate()`'s size and control edges, stationary distributions, the sparse-cell threshold,
  `regime_feature` fallbacks, the runner lifecycle and its worker-thread fetch, the LLM provider
  layer, `AgentStats` and the forecast bucketing, fail-closed `Settings` defaults, `.env.example`
  keys, the Kalman filter's adaptive Q, process noise, state intervals, strategy config edges, exit
  precedence, two-leg sizing, degenerate metrics and health gates, the fold layout and pair
  qualification, the re-derivation script's hash and label rules against the engine, the committed
  snapshots and study configs, every Markdown link, `CITATION.cff`, the site's SRI pins, and the
  legacy `/runs` lifecycle offline.

### Changed (2026-11-07 batch)
- yfinance polling and the runner's benchmark download run in a worker thread so the legacy loop
  no longer blocks the event loop (#182, #189). The Kalman trailing z-score is vectorised with
  pandas rolling statistics (byte-identical report, pinned by an equivalence test, #218) and the
  pair-data validator scans cells only in object-dtype columns (#238).
- `Policy` is a real `Callable` alias and the policy registry is typed with it (#174); the regime
  lookup, `app.db` accessors and two Kalman forward references carry their real types (#180,
  #234, #222); redundant casts, a lambda, a chained comparison, duplicated helpers and a
  recomputed fee rate are tidied (#190, #230, #252, #188, #183, #245, #177, #240).
- Every `pct_change` in the study scripts and the Kalman ledger passes `fill_method=None`
  explicitly, matching the engine (#212, #250).
- README stops quoting a test count, gains a repository-layout table and links the docs index and
  glossary (#268, #269, #270); SECURITY describes dependency updates (#272).
- Issues #64–#173 record every item of this batch (nine remain open as follow-ups: #165–#173);
  #79 was closed as not-a-bug because the percentage schema already caps buys at 100 %.

## [0.4.1] — 2026-09-08 — deep review: 24 verified findings fixed

### Security (fail-closed)
- **The LLM agent is no longer publicly triggerable.** Review found the legacy web surface
  (`POST /runs {"strategy":"agent"}`, `GET /runs/{id}/comparison`) reached the live
  Anthropic client with no budget ceiling and no cache, while the old tripwire test
  grepped the wrong module. Now guarded at the route, the runner loop, and the comparison
  replay by `ALLOW_LEGACY_AGENT_API` (default **off**), with runtime TestClient
  enforcement tests.

### Kalman framework (methodology corrections — verdict re-earned)
- LOG-model sizing now uses β's **magnitude** (true elasticity hedge; the old 50/50 split
  used only its sign, so dynamic-vs-static hedging was never actually compared).
- Health gating made variant-agnostic (identical z-based causal gate for every variant);
  per-fold warm-up re-blanking removed; calibration likelihood made truly predictive;
  proper Engle–Granger p-values via `coint` (SPY/QQQ: p≈0.43–0.83, weaker than the
  mislabeled 0.21–0.63); REHEDGE missed-fill desync, cooldown ticking, trade counting,
  audit-table alignment, bootstrap final block, `tr0` threading all fixed.
- Corrected 15-fold/20y walk-forward: **no robust edge after costs** — every variant's
  90% bootstrap CI (now printed in the report, with per-variant z-std and gated-bar
  columns) straddles or sits below zero.

### Legacy path correctness
- `as_of` regime replays no longer see their own day's settled close (strict `<`).
- Cross-symbol snapshot mixing refused (replay, runner, and a 422 at run creation).
- `start_run_task` wired into the route: stop/cancel is real, stop flags reset per run,
  requests no longer pin on the infinite loop; dust trades flattened like the ledger;
  replays use the run's own stored `starting_cash`; inert zscore thresholds warn loudly;
  `REGIME_K` reachable end-to-end; `regime-cli` pinned to snapshots; liquidity-floor and
  allocation-cap semantics documented honestly; falsy-zero ingestion checks fixed.

### Canonical path
- Uniform-row fallback in `estimate_transition_matrix` documented, and its never-fired
  receipt (per-step self-prob ≥ 0.600 on all 16,773 published decisions) pinned as a
  regression test. Headline numbers untouched. Tests: **105 passing**.

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
