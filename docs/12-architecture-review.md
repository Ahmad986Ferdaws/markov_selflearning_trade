# Architecture Review — Markov Self-Learning Trade (v1)

**Status:** Proposed (review)
**Date:** 2026-06-06
**Reviewer:** Senior engineer pass
**Scope:** Whole-project architecture, simulation-only v1. Code + docs + tests as of this date.

---

## 1. Executive summary

This is a **well-structured small system that is one layer of rigor away from doing its job.** The layering (routes → services → repository → DB), the dependency injection, the provider abstraction, and the guards-after-agent enforcement are all correct senior-level choices. The code is clean, typed, and tested (15 passing).

But the project's entire reason to exist is **one number**: agent return minus baseline return, net of costs, computed on an identical, fair replay. Three architectural shortcuts currently distort that number, and the system has no way to tell you it's distorted. The architecture optimizes for "the loop runs" when the actual requirement is "the comparison is *trustworthy*." Those are different systems, and the gap between them is where this review focuses.

**Verdict:** Sound skeleton. Do **not** trust a published comparison result until the regime-as-of-time, return-frequency, and replay-fidelity issues (Section 4) are closed. Phases are marked "Done" in STATUS.md; I'd reclassify Phase 2 as "runs, but not yet valid."

### Scorecard

| Dimension | Grade | One-line |
|---|---|---|
| Separation of concerns | A− | Clean layers, thin routes, logic in services |
| Correctness of the core metric | C | Constant regime + wrong annualization distort the headline |
| Async/concurrency model | C+ | Blocking I/O in async paths; in-memory task registry |
| Data model | B | Right tables; missing time/run-config provenance for fair replay |
| Testability | B+ | Good DI seams; tests assert shape, not financial correctness |
| Reproducibility | C+ | yfinance is non-deterministic; no snapshot of regime inputs |
| Scope discipline | A | Simulation-only held firmly; guards non-overridable |
| Operability | C | Module-global run state; no recovery on restart |

---

## 2. What the architecture gets right

These are deliberate, correct decisions worth preserving:

- **Guards are enforced *after* the agent and cannot be widened.** `enforce_decision` clamps allocation, applies the liquidity floor, and rejects shorts/over-spend. The agent never sees the guards as negotiable. This is exactly the trust boundary the brief demands.
- **Intent vs. enforced is persisted on every `Trade`.** The guard-intervention audit trail is real data, not a log line — it survives to the writeup.
- **Repository-friendly DB layer.** `get_engine`/`get_session_factory`/`get_db` behind a single module means the SQLite→Postgres swap touches one file, not business logic. Correct.
- **Provider interface for ingestion** (`IngestionProvider` ABC, yfinance/dexscreener). The pluggability the docs promise is real, and the dexscreener stub fails loud (`NotImplementedError`) rather than silently — good.
- **Cost model lives in one place and is applied in both the live ledger and the Phase-0 backtest.** No "backtest without costs" lie at the regime layer.
- **Agent contract is defended in depth:** fence-strip → Pydantic validate → one repair retry → HOLD fallback. The loop cannot crash on bad LLM output.

This is the work of someone who knows how to structure a service. The criticism below is about **financial-simulation correctness**, not software hygiene.

---

## 3. Architectural risks (ranked)

### R1 — The system cannot detect when its own headline number is invalid (severity: high)
There is no validation layer between "the loop produced numbers" and "the numbers mean what the report says they mean." The Sharpe field will happily print `47.3` from 15-second returns annualized by √252, and nothing flags it. **A measurement system whose central output has no sanity gate is the core architectural defect.** Everything in Section 4 is a symptom of this missing layer.

> **Recommendation:** add a thin `validation`/`metrics` module that the comparison must pass through, with explicit assertions: return-series frequency matches the annualization factor; regime varied across the replay (else warn "constant-regime run"); equity never non-finite. Fail loud, in the report.

### R2 — Regime feature is time-blind (severity: high)
`regime_feature(benchmark_history)` always returns the latest state of the *whole* history, so within a run/replay it is **constant**. The baseline degenerates from "regime strategy" to "one fixed decision repeated." This is an architecture problem, not a one-liner: the regime layer has no concept of "as of snapshot time *t*," so it *cannot* be point-in-time correct without a signature change. See ADR-1.

### R3 — Two return clocks are conflated (severity: high)
The Phase-0 backtest operates on **daily bars** (√252 correct). The comparison operates on **15-second snapshots** (√252 nonsense). The same `Sharpe` symbol means two different things in two modules with no type or unit to distinguish them. Units-as-comments is how bad numbers ship.

### R4 — Replay fidelity is assumed, not enforced (severity: medium)
The brief's fairness guarantee is "both strategies see the identical snapshot stream." Today that holds *only because* both call the same function on the same rows — but the run that *generated* the snapshots used live yfinance with no record of the regime inputs at each tick. If the benchmark history changes between the live run and the replay (yfinance revises, or you run the comparison a week later), baseline-in-replay ≠ baseline-that-traded-live. The data model has no provenance to catch this.

### R5 — Run lifecycle lives in process memory (severity: medium)
`_active_tasks` / `_stop_flags` are module globals; `BackgroundTasks` launches an infinite loop while the cleaner `start_run_task` is dead code. A restart orphans every "running" run (status stuck, no loop, no recovery). Fine for a laptop demo; a latent operability bug the moment this runs anywhere real.

### R6 — Blocking I/O in async paths (severity: medium)
`YFinanceProvider.poll` and `fetch_daily_history` make synchronous network calls inside `async` functions, blocking the event loop. With one run it's invisible; with two concurrent runs the API stalls. The async signature is writing a check the implementation doesn't honor.

### R7 — Reproducibility is not designed for (severity: medium)
yfinance returns revisable, vendor-dependent data; runs are timestamped via `func.now()`; no seed or input-hash is stored. Re-running the "same" comparison can produce a different headline. For a system whose output is a *published claim*, irreproducibility is an architectural liability, not a detail.

---

## 4. The three correctness forks (ADR-style)

### ADR-1: Make the regime feature point-in-time

**Context:** `regime_feature` reads the latest state of the full history; in replay it is constant (R2).

**Options:**

| Option | Complexity | Fidelity | Notes |
|---|---|---|---|
| A. Slice history to `snapshot.created_at`, recompute state as-of | Med | High | Point-in-time correct; matches walk-forward philosophy already in `regime.py` |
| B. Precompute a per-day regime series once, join snapshots by date | Low | High | Cheaper, cache-friendly; same result for daily regime |
| C. Leave constant, document as "single-regime test" | Low | Low | Honest but guts the comparison's purpose |

**Decision:** **B** for the comparison harness (precompute daily regime series, look up by snapshot date), **A**'s signature (`regime_feature(history, as_of=...)`) for the live loop. B is O(days) not O(snapshots×days) and keeps the replay fast.

**Consequences:** baseline becomes a real regime strategy; `regime_feature` gains an `as_of` parameter (breaking change to one internal caller); the comparison gains a "regimes seen" count that R1's validator can assert is > 1.

### ADR-2: One return clock, explicitly typed

**Context:** √252 is correct for daily bars, wrong for snapshot returns (R3).

**Decision:** Carry the sampling frequency with the return series. Annualize by `√(periods_per_year)` derived from the actual cadence (`poll_interval_seconds`), or report a raw per-period Sharpe labeled `sharpe_per_snapshot`. Never hardcode 252 outside the daily backtest.

**Consequences:** the comparison Sharpe becomes interpretable; a `frequency` field threads through `StrategyMetrics`; R1's validator asserts frequency-vs-factor consistency.

### ADR-3: Define "win" at the round-trip, not the fill

**Context:** `win_rate = wins / trades` counts buys in the denominator (R3 in the code review).

**Decision:** `win_rate = profitable_closes / closes`. Track realized PnL per closed lot.

**Consequences:** the metric matches its name; requires tracking entry basis per close (already partly present via `avg_entry`).

---

## 5. Data model gaps (for fair, reproducible replay)

The four tables (`Run`, `Snapshot`, `Trade`, `Metric`) are the right *set*. What's missing is **provenance for reproducibility**:

- **`Run` should snapshot its config** — `fee_pct`, `slippage_pct`, `max_allocation_pct`, `regime_window`, `benchmark_symbol`, model name — at creation. Today settings are read live, so a `.env` edit retroactively changes how an old run is interpreted.
- **`Snapshot` should store the regime inputs used at that tick** (or at minimum the resolved `state`/`p_next`). This makes replay independent of re-fetching yfinance and closes R4.
- **`Metric` is defined but unused.** Either wire computed metrics into it (so reports are queryable history, not recomputed each call) or drop it.
- **No index on `Snapshot(run_id, created_at)`** — every replay/`_last_price` does an ordered scan. Trivial now, O(n) later.

---

## 6. Testing posture

Tests verify **shape** (`report.baseline.name == "baseline"`, `cash < 1000`) but not **financial correctness**. There is no test that would fail if Sharpe were annualized wrong, if the regime were constant, or if costs were dropped. For a measurement system, that's the coverage that matters most.

> **Recommendation:** add golden-number tests on synthetic data with hand-computable answers — e.g. a flat price series must yield ~0 return and 0 realized PnL net of costs; a known up-only series must yield a positive return *reduced by exactly the cost model*; a two-regime synthetic history must produce > 1 distinct baseline decision. These are the tests that protect the headline.

---

## 7. Prioritized action list

**Must-fix before any result is trusted (the headline-integrity set):**
1. [ ] ADR-1 — point-in-time regime in comparison + live loop
2. [ ] ADR-2 — correct/relabel Sharpe annualization
3. [ ] ADR-3 — win-rate denominator = closes
4. [ ] R1 — add a validation gate that fails loud on inconsistent metrics

**Should-fix for a credible v1:**
5. [ ] Snapshot regime inputs + run config for reproducible replay (R4, R7, Section 5)
6. [ ] Move blocking yfinance calls to `asyncio.to_thread` (R6)
7. [ ] Unify run lifecycle on `start_run_task`; delete the `BackgroundTasks` infinite-loop path (R5)
8. [ ] Golden-number correctness tests (Section 6)

**Cleanup (low risk, do alongside):**
9. [ ] Dead code: `regime.py:129` no-op replace; `runs.py:130` unused `last_price`; `effective_price` discarded return
10. [ ] Index on `Snapshot(run_id, created_at)`
11. [ ] Wire or remove the `Metric` table

---

## 8. Closing assessment

The software architecture is **good**. The simulation architecture is **not yet sound**, and for this project the simulation *is* the product. The fix is not a rewrite — it's adding the one layer the system is missing: a point-in-time, unit-aware, self-validating measurement path. Everything needed to do that is already structured to receive it. Close items 1–4 and this goes from "a loop that runs" to "a result you can publish."
