# Project status (living)

Updated at the end of each dev session. New chats: read this file first—skip long recap unless asked.

## Product direction

- **Core product:** paper trading application (ledger, guards, simulated fills).
- **Regime / training data:** yfinance (Yahoo), liquid symbols (e.g. BTC-USD).
- **Live polling (v1):** yfinance watchlist (default); DEX Screener deferred post–Phase 1 gate.
- **Charts (later):** TradingView on internal bar DB.
- **Excluded:** Binance and similar Asia-centric exchange APIs.

## Implementation state

| Phase | Status | Notes |
|---|---|---|
| 0 — Regime | Done | `regime-cli`, `app/services/regime.py`, tests pass |
| 1 — Loop | Done (legacy intraday) | yfinance ingestion, ledger, guards, baseline, FastAPI `/runs` |
| 2 — LLM + compare | Done (validated) | `agent.py`, `GET /runs/{id}/comparison`, `compare-cli`; headline-integrity blockers closed |
| **Reframe Phase A — daily eval** | **Done (audited)** | `evaluate-cli`, `app/services/evaluation.py`; daily walk-forward, z-score regimes, accuracy + buy-hold, train/test split. See [13-reframed-plan](13-reframed-plan.md), [15-phase-a-results](15-phase-a-results.md) |
| **Reframe Phase B — one engine + repro** | **Done** | Pluggable single engine (policies through one `_policy_metrics`); data pinned + hashed (`data_cache.py`); deterministic run records in `results/`; legacy intraday quarantined. 30 tests. |
| **Phase C — LLM agent policy** | **Foundation done (offline-verified); live run pending** | `app/services/agent_policy.py`: agent is a pluggable `PolicyContext -> position` like any other. Provider switch `anthropic\|ollama\|none` (keys from env only). Prompt-hash response cache → only ~7 unique calls cover 328 test days (reproducible, cheap). `evaluate-cli --provider …` health-checks the provider, adds the agent as the 3rd policy, persists the cache. Verified: always-long agent == buy_hold exactly. **Still needed:** a reachable model (Ollama on the M4, or `ANTHROPIC_API_KEY`) to produce the real agent row. |
| **Phase C — honesty hardening** | **Done (offline; from the article-fit workflow)** | Mined the ReAct/quant-agent article for honesty+enforcement, not capability (the prompt-hash cache already *is* a safe "agentic memory"; ReAct/tools/vector-store/multi-agent rejected as contamination/scope-incompatible). Applied 3 on-identity changes: (1) `AgentStats` now counts decision provenance per tier (`strict_json`/`salvaged_labeled`/`salvaged_single_number`/`unparseable_hold`/`errors`) — `_parse_position` returns a stable tier tag, the closure tallies it, the CLI prints a "% clean" histogram (a *measured* confidence report, not a self-asserted accuracy). (2) **Gate 6** in `evaluate()`: warns when the agent acted on near-uniform forecasts (`max(p_next) < 1/n+0.15`) on ≥25% of test days — finally using the `p_next` hook designed into `PolicyContext`; appends a caveat, never overrides. (3) Three enforcement tests (`tests/test_agent_invariants.py`): agent stays off the web/API surface (CLI-only, not publicly triggerable), keys are env-only (never a call argument), and the decision is invariant to `day_index` (no timeline leak into the cache). Tests: **52 passing**. |
| **Robustness study (cross-asset)** | **Done (57-agent workflow + hand-verified)** | `scripts/robustness_study.py`, `results/robustness/`, [16-robustness-study.md](16-robustness-study.md). Ran the honest eval across **20 assets + 4 sub-periods**. Finding: model edge over persistence is **exactly 0.000 on 20/20** — re-verified as a pointwise identity (**0 divergences / 5,425 predictions**). "Beats buy-hold 10/20" is a downtrend artifact (all 10 winners are net-negative; cash dominates). Only non-zero edge anywhere is early-BTC **−0.167** (model did *worse*, 1 divergence on a 2-day class). Caught + fixed 3 numeric errors in the agent synthesis before publishing. |
| **Website (design/)** | **Done — Robinhood-language rebuild, critiqued + polished** | `design/index.html`: single-file site in the verified Robinhood 2024 (COLLINS) brand language — Newsreader serif / Schibsted Grotesk / IBM Plex Mono, black-white-neutrals + one neon `#CCFF00` pop, product green `#00C805`, pills, dark verdict band, dense disclosure footer. Motion: GSAP + ScrollTrigger + Lenis + Three.js (SRI-pinned CDNs, `.nogsap` + reduced-motion fallbacks). Equity chart shows the **real** held-out series (165 pts from the engine — cash shelves visible), not procedural noise. A 5-critic + adjudicator workflow produced 10 MUST fixes (all applied: payoff timeline race→tie→stamp, contrast pass, sentence-case labels, anchored 3D self-loops, reduced-motion support, small-screen fixes). Proof media regenerated: `media/01-04*.png` + `regime-demo.mp4` (≈21 s, real-time). **+ ui-design-brain refinement pass** (a11y semantics, keyboard chart, scroll-spy nav, AA contrast, 375px fixes) — see Last session. |
| **Long-history study (20y)** | **Done — retires the #1 limitation** | `scripts/long_history_study.py`, `results/long_history/`, [18-long-history.md](18-long-history.md). Same protocol as the robustness study on the max Yahoo span per asset (BTC 12y, ETH 10y, SPY/QQQ/AAPL/GLD/TLT/EURUSD 20y — spans 2008, COVID, 2022). Edge vs persistence **+0.0000 on 8/8**, identity verified pointwise: **0 divergences / 11,348 predictions** (combined total now **0 / 16,773**). Confirms the docs' caveat with data: on uptrending long windows the baseline **loses** to buy-and-hold on 7/8 (the earlier "beats hold" was a downtrend artifact). |
| **Phase C — guards + diagnoser** | **Done (offline; the 3 deferred items)** | (1) **Call-budget ceiling** — `build_agent_policy(max_unique_calls=…)` / `AGENT_MAX_CALLS` (default 64): a run can never fan out unbounded billed calls; fails closed (holds) and warns. (2) **Cache-only replay** — `--cache-only` / `AGENT_CACHE_ONLY`: serves only the persisted response cache, zero live calls, no health-check ping; the mode any public surface must use. (3) **`diagnose-cli`** — offline narration of a run record (edge vs the persistence bar, policy outcomes, every gate warning); reads only the already-computed JSON, so lookahead-safe by construction. Stats now track `budget_holds`/`cache_only_holds`; CI added (`.github/workflows/ci.yml`) running the suite + a determinism grep of the 90.9≡90.9 tie. Tests: **58 passing**. |
| **Model-logic deep dive** | **Done — adversarially verified (2 workflows, all findings applied)** | [19-anatomy-of-zero.md](19-anatomy-of-zero.md), `tests/test_instrument_validation.py`, `scripts/model_logic_study.py`, `scripts/logic_depth_study.py`, `scripts/independent_rederivation.py`, `results/model_logic/` + `results/logic_depth/`. (1) **Positive controls in CI**: the engine detects planted alternation/cycles (+90pp edge) and reproduces the exact zero on sticky synthetic — the zeros are measurements, not malfunctions. (2) **Switch-day anatomy** (new engine metrics + **Gate 7**): BTC 90.9% = 298/298 stay days + 0/30 change days; the model never once predicted a change. (3) **Per-step dominance**: the causal row used at every one of **16,773** decisions had self-prob ≥ **0.600** (> the 1/2 bound) — the identity is forced, step by step; rule-of-three bound: divergence rate < 0.018% at 95%. (4) **Duration + memory-2 tested honestly**: fire 23 + 20 times total, edges uninformative — the obvious fixes don't work. (5) **Honest positive**: the matrix beats sticky baselines on log-loss 28/28 (calibration, never decisions). (6) **Independent re-derivation with zero app imports** reproduces 328/30/0/298 — in CI. A statistician + code-review + replication verification pass caught 9 wording overclaims and 8 code findings; all applied (incl. broadened Gate 7, hash-pinned studies, per-step measurement). Tests: **64 passing**. |
| **Deep review + landscape** | **Done (multi-agent, harvested + hand-verified)** | [17-review-and-landscape.md](17-review-and-landscape.md). 11 review/positioning agents (vs vectorbt/freqtrade, hmmlearn HMMs, TradingAgents arXiv:2412.20138, the persistence/overfitting literature) + 10 verified web searches. The review workflow stalled overnight; findings harvested from its journal and re-verified by hand — **caught a false CRITICAL** (claimed snapshots unreadable; they read fine under pandas 3.0.3). Grades A−/B+ across most dims. **Fixes applied:** SRI integrity hashes on the 4 CDN scripts; hardened `_parse_position` (no more mistaking a stray number for the position) + 4 tests; `requirements.lock` (77 pins). Tests: **49 passing**. Verdict: strong *honest-measurement artifact*, not a strategy; novelty is the framing (persistence bar + honesty-gates-as-code + contamination-proof agent), not technique. |

**Direction (current):** reframed to an **honest daily evaluation harness** (portfolio artifact), per [13-reframed-plan.md](13-reframed-plan.md). The legacy 15s intraday loop is superseded by the daily path.

**Repo:** `app/` implemented; run with Python 3.11+ (3.12 venv recommended).

## Commands

```bash
source .venv/bin/activate   # Python 3.12 venv
evaluate-cli                # Phase A daily eval (accuracy + trading) — current headline
regime-cli                  # Phase 0 regime report
uvicorn app.main:app --reload
compare-cli <run_id>        # legacy intraday comparison
```

## Open decisions

- **DEX Screener provider:** stub only; implement after Phase 1 sign-off if memecoin mode needed.
- **Regime `window`:** default 20; tune via `REGIME_WINDOW` in `.env`.

## Last session

- **Done — website design refinement pass (ui-design-brain skill).** `design/index.html`: accessibility + component-quality upgrade — semantic landmarks and strict h1→h2→h3 outline, scroll-spy nav with `aria-current`, keyboard-operable chart scrub, table caption/scope, ≥44 px touch targets, WCAG AA contrast (`--muted` darkened), ≥12 px micro-text, 375 px tie-band overflow fixed, dead footer link now points at the real results JSON. Brand language, GSAP/Lenis/Three.js stack, SRI pins, reduced-motion fallbacks, and the real 165-pt data series untouched. Verified in-browser at desktop and 375 px; `design/philosophy.md` documents the pass. Note: `design/media/` (proof media referenced in earlier notes) is absent from the repo; nothing on the page references it, but regenerate before deploy if `og:image` is wanted.

### Earlier session

- **Done — Reframe Phase A (daily evaluation), audited.** New `app/services/evaluation.py` + `evaluate-cli`. Tests: **28 passing**.
  - **R1 daily walk-forward** — signal clock == trade clock (kills the 15s mismatch). Causal, no lookahead.
  - **R2 z-score regimes** — `define_states` now thresholds on `±k·rolling_std` (via `label_state`), not fixed ±2%/day. BTC test mix: sideways 72% / bear 14% / bull 14% (was ~99% sideways → it now actually trades).
  - **R3 buy-and-hold** added as the yardstick.
  - **R4 accuracy** — hit-rate, balanced accuracy, log-loss, vs **two** baselines: naive(majority) and **persistence ("tomorrow=today")**.
  - **R5 train/test split** (70/30) — hyperparams chosen on train by balanced accuracy among configs with ≥2 well-represented regimes; reported on held-out test only.
  - **R6 overfit check** + single-regime fallback warnings.
  - **Adversarial audit** (workflow, 4 auditors + adjudicator): 4 LOW findings, all fixed (selection train-isolation `split-1`, honest naive-balanced print, train-region guard, effective train/test sizes). Verdict: numbers trustworthy.
- **HONEST FINDING (BTC, held-out test):** model predicts next-day regime at 90.9% hit-rate — but **persistence scores identically (90.9%)**. The Markov transition matrix adds ~nothing over autocorrelation. Both trading policies lose net of costs (baseline −41%, buy-hold −48%). This is the publishable result; see [15-phase-a-results.md](15-phase-a-results.md).
- **Done — Reframe Phase B (foundation before any LLM).** 30 tests passing.
  - **One engine:** policies are now pluggable functions (`PolicyContext` -> position) that all flow through the same `_walk_forward` + `_policy_metrics`. No duplicated fill math. Custom-policy test proves the Phase-C agent plugs in with zero new engine code.
  - **Reproducibility:** `app/services/data_cache.py` — `load_or_fetch` pins the yfinance pull to a local snapshot (`data/snapshots/`), `history_hash` stamps it, `evaluate-cli` is deterministic and writes a run record to `results/{symbol}_{hash}.json`. Determinism test added.
  - **Legacy quarantined:** `comparison.py` and `runner.py` (the buggy 15s intraday path) carry deprecation headers; the LLM must NOT be wired there.
  - Scoping note: the original "merge ledger+comparison engines / async / tz / lifecycle" items were all in the deprecated intraday path, so they were intentionally NOT polished — the daily path is canonical.
- **Next:** Phase C — add the LLM agent as a third pluggable policy in `evaluation.py` (provider switch anthropic|ollama), run the three-way daily comparison, write up, rename repo.

### Even earlier session
- **Done:** Closed the 4 headline-integrity blockers from the architecture review ([12-architecture-review.md](12-architecture-review.md)), plus a 5th bug the golden tests surfaced. Tests: **22 passing**.
  1. **Point-in-time regime** — `regime_feature(history, as_of=...)`; comparison precomputes one regime per snapshot date via `_build_regime_lookup`. Baseline is now a real regime strategy, not a constant.
  2. **Sharpe annualization** — derived from `poll_interval_seconds` (`_annualization_factor`), not hardcoded √252. Report labels the basis, e.g. `Sharpe (ann@2,102,400/yr)`.
  3. **Win rate** — denominator is closes (SELLs), not all trades. Separate `Closes` row.
  4. **Validation gate** — `_validate_report` warns loudly on constant-regime runs and non-finite metrics; surfaced in `format_report` and the `/comparison` API (`warnings`).
  5. **Guard bug (bonus):** the 20% allocation cap was clamping SELL % too, so positions never fully closed. Cap now applies to BUY only (matches brief: "20% of current *cash*").
- **New tests:** `tests/test_metrics_correctness.py` — golden-number tests for each fix.
- **Next (from review, not yet done):** reproducibility (freeze run config + store regime inputs per snapshot); move blocking yfinance calls to `asyncio.to_thread`; unify run lifecycle on `start_run_task` and drop the `BackgroundTasks` infinite-loop path; minor dead-code cleanup.
- **Note on Sharpe magnitude:** annualizing 15s returns yields large-magnitude Sharpe by nature; the value is now honestly labeled with its per-year basis rather than mislabeled as daily.
- **Blockers:** none headline-affecting.
