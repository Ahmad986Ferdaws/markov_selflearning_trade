# 17 — Deep Review & Landscape Positioning

**How this was produced:** a multi-agent review (7 internal dimensions reading the actual repo + 4 external positioning lenses grounded in 10 verified web searches), then adversarial verification. The verification fan-out stalled overnight (machine sleep orphaned the in-flight agents), so the 11 review/positioning agents' findings were **harvested from the run journal and every load-bearing claim re-verified by hand** before writing this — which caught a false CRITICAL (see §Repro). Treat this as reviewed-and-fact-checked, not auto-generated.

---

## 1. What this project is

A **simulation-only "honest instrument"**: a daily, causal, walk-forward evaluation harness that asks whether a Markov regime model — and (Phase C) an LLM agent — can beat dumb baselines net of costs. Its deliverable is a **methodologically-correct null result**, not a strategy or a reusable framework. Across 20 assets the Markov layer's edge over persistence is exactly 0.000 (hand-verified as a pointwise identity), and the long/flat policy's apparent wins are down-beta, not alpha.

## 2. Internal review (grades, then verified findings)

| Dimension | Grade | One-line |
|---|:--:|---|
| Architecture & engineering | **A−** | Genuinely single-engine ("one engine, all policies"); the seam and legacy quarantine are real |
| Statistical & scientific validity | **A−** | Unusually honest and sound; loses points on trading-economics framing + one stale test/grid drift |
| Security & sim-only enforcement | **A−** | Sim-only is *enforced* (no execution/wallet/key code exists), not just claimed; no secrets committed |
| Frontend (the site) | **A−** | Exceptionally honest single-file build; real gaps are CDN integrity + accent contrast |
| Code quality & testing | **B+** | Rigorous on the financial-correctness paths; thin on the agent's parse/cache edges |
| Research contribution / novelty | **B+** | Honest, reproducible artifact; the *result* is a confirmed-known property, not a discovery |
| Reproducibility & data engineering | **B−** | Right architecture; real gaps are no version control + no version lock (see correction below) |

**Verified strengths (the real ones):**
- **One engine, all policies.** Every policy — buy-hold, regime baseline, LLM agent — is the same `callable[[PolicyContext], float]` flowing through one `_walk_forward` + `_policy_metrics`. No duplicated fill math. (`evaluation.py`)
- **Honesty gates as executable code.** `evaluation.py` emits a loud warning when the model fails to clear persistence by >0.02, when a regime dominates >80%, or when metrics are non-finite. Balanced accuracy is defined so it *refuses* to reward the degenerate "always sideways" model. The harness is built to incriminate itself.
- **Sim-only by construction.** There is no order-execution, wallet, or key-handling code to disable — the constraint is structural. Keys come from env only; no secrets in the tree.
- **The null is verified as a structural identity**, not a noisy estimate: 0 divergences / 5,425 held-out predictions, re-derived by hand.

**Verified weaknesses (severity after my fact-check):**
- **HIGH — Frontend CDN has no Subresource Integrity.** The 4 GSAP/Lenis/Three scripts load from cdnjs/jsdelivr with no SRI hash and no `crossorigin`. A hijacked CDN response runs arbitrary JS. *(Fixed — see §5.)*
- **HIGH — `_parse_position` salvage bug.** When JSON parsing fails, it returns the *first* in-range number, which need not be the position; a reply like "in 2024 I'd take 0.7" salvages `0.0`-ish wrong values, and no test covered a stray-number-first reply. *(Fixed — see §5.)*
- **MEDIUM — No version control + snapshots git-ignored.** There is no `.git`, and `.gitignore` excludes `data/snapshots/`, so a fresh clone has neither the pinned data nor history — reproducibility is local-machine-only today.
- **MEDIUM — No version lock.** `pyproject.toml` uses `>=` ranges (numpy, pandas, yfinance); the env has already drifted to pandas 3.0.3. True byte-reproducibility needs a lockfile or pinned versions.
- **MEDIUM — Three services have ~zero test coverage** (ingestion, parts of the API/runner) — acceptable given they're legacy/quarantined, but undocumented as such in the test suite.
- **LOW — `regime.py` mixes labeling, estimation, and I/O** and could be split.

**Correction (why fact-checking mattered):** the review flagged a **CRITICAL** — "every `.pkl` snapshot raises `NotImplementedError` on unpickle, so byte-reproducibility fails outright." **This is false.** All 20 snapshots read cleanly here; the agent asserted "pandas 2.3.3" but the actual environment is **pandas 3.0.3**. The legitimate kernel of truth — pickle is a fragile cross-version format — survives only as a MEDIUM recommendation (prefer Parquet/CSV), not a critical failure.

## 3. Where it sits in the landscape

All four positioning lenses converged on the same read: **this is a measurement artifact, not a competitor.**

**vs. backtesting frameworks** — [vectorbt](https://github.com/polakowo/vectorbt), [backtrader](https://github.com/mementum/backtrader), [zipline-reloaded], [freqtrade](https://github.com/freqtrade/freqtrade), [nautilustrader]. The hand-rolled walk-forward engine **reinvents the wheel for the parts that aren't its purpose** (fills, costs, equity curves — all done faster and battle-tested elsewhere) but is **justified for the narrow slice it cares about**: a transparent, auditable, no-lookahead accuracy-vs-persistence harness you can read end-to-end. It is *not* trying to be vectorbt; using vectorbt would actually be the right call if speed/scale ever mattered.

**vs. HMM / regime tooling** — [hmmlearn `GaussianHMM`], [QuantStart], [QuantInsti], [Hudson & Thames], statistical jump models ([arXiv:2402.05272](https://arxiv.org/pdf/2402.05272)). Two honest points: (1) this project's "Markov" is an **observed-state Markov chain over z-score labels — not a hidden Markov model** (no Baum-Welch, no latent states). That's a fair, clearly-labeled simplification, not a true HMM. (2) Those sources almost universally benchmark **only against buy-and-hold**; this project benchmarks the *prediction task against persistence*, which is the methodologically correct bar — and its real edge over the genre.

**vs. LLM trading agents** — [TradingAgents (arXiv:2412.20138)](https://arxiv.org/abs/2412.20138) (multi-agent LangGraph pipeline, claims up to 30.5% annualized), benchmarks [InvestorBench](https://arxiv.org/pdf/2412.18174)/StockBench (which found GPT-5/Claude-4 *struggle to beat simple baselines, especially in bear markets*), and the **lookahead/memorization literature** ([Memorization Problem, arXiv:2504.14765](https://arxiv.org/html/2504.14765); [Lookahead Bias in LLM Forecasts, arXiv:2512.23847](https://arxiv.org/html/2512.23847v1)). Phase C is **methodologically stronger but deliberately less ambitious**: the agent is **contamination-immune by construction** — it only ever sees a causal `PolicyContext` (today's regime + a bucketed forecast, no dates, no prices, no headlines), so there is nothing for an LLM to "remember." It will never claim 30.5%; it's the thing you'd use to *check* whether a TradingAgents-style number is a profit mirage.

**vs. the academic honesty / persistence literature** — [CFA "Dumb Alpha"](https://blogs.cfainstitute.org/investor/2015/11/10/dumb-alpha-are-your-forecasts-better-than-a-random-walk/), [ECB random-walk FX], ["Is the Naive Baseline Unbeatable", arXiv:2406.14469](https://arxiv.org/html/2406.14469v11), [Bailey "Statistical Overfitting"](https://sdm.lbl.gov/oapapers/ssrn-id2507040-bailey.pdf), Harvey-Liu-Zhu, ["Seven Sins of Quant"](https://portfoliooptimizationbook.com/book/8.2-seven-sins.html). **This is the strand the project actually belongs to.** The field explicitly calls for publishing negative results to fight publication bias; this project does exactly that, in code. The null is not luck — it re-derives, on its own data, the thing this literature already says, and reports it instead of dressing it up.

## 4. The uncomfortable truths (verified)

1. **The "finding" is a confirmed-known property, not a discovery.** "Persistence is hard to beat" is decades-old. The contribution is *engineering integrity and a clean local demonstration*, not new knowledge.
2. **It's an observed-state Markov chain, not an HMM.** Calling it a "Markov model" is fair; implying HMM-level sophistication would not be.
3. **The engine reinvents mature tooling** for everything except its honesty core.
4. **One ~3-year window per asset.** Twenty assets widens the cross-section but not the *time* dimension; a multi-cycle history is untested.
5. **Novelty = framing, not technique.** The persistence benchmark, the in-code honesty gates, and the contamination-proof agent design are the genuinely uncommon parts — and they're design choices, not algorithms.

## 5. Fixes applied during this review

- **Frontend SRI** — added `integrity` + `crossorigin="anonymous"` to all four CDN `<script>` tags (closes the HIGH supply-chain gap).
- **`_parse_position` hardening** — now prefers an explicit `position`/`exposure`/`target` key before any number salvage, and only salvages a bare number when the whole reply *is* essentially that number; added regression tests for the stray-number-first case.

## 6. Prioritized recommendations (by leverage)

1. **`git init` + commit** (it's untracked) — and decide deliberately whether `data/snapshots/` should be committed (needed for the byte-repro claim) or regenerated. *(High leverage, 5 min — your call to run.)*
2. **Pin versions / add a lockfile** — the env already drifted to pandas 3.0.3; lock it.
3. **Phase C live run** — the one experiment that adds new information; needs a reachable model (Ollama on the M4, or `ANTHROPIC_API_KEY`).
4. **Switch snapshots to Parquet/CSV** — kill the pickle-portability risk for good.
5. **Add a longer / multi-cycle history test** — the only thing that could overturn the null.
6. **Reframe the docs' "Markov model" → "observed-state Markov chain"** once, with a one-line note vs. HMMs, to pre-empt the obvious critique.
7. **Test the three uncovered services** or explicitly mark them deprecated in the suite.

## 7. Bottom line

**A good project — for what it actually is.** As a *trading system* it has no edge and isn't trying to. As a **portfolio/learning artifact demonstrating research integrity** — a causal, reproducible harness that benchmarks against the right bar and reports zero when zero is true, then adversarially audits its own write-up — it is genuinely strong and rarer than the 338-skill-repo, 30%-annualized-claim genre it superficially resembles. The honest framing *is* the product. Ship it as that, not as alpha.

*Sources: verified via web search June 2026; see inline links. Internal findings re-checked against the repo before publishing.*
