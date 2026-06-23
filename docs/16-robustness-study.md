# 16 — Robustness Study: does the Markov regime model have an edge?

**Status:** complete, numbers independently re-verified.
**Artifacts:** `scripts/robustness_study.py`, `results/robustness/*.json`, `results/robustness/summary.csv`.
**Method:** the exact daily walk-forward evaluator (`app/services/evaluation.py`) run across 20 assets + 4 sub-periods. Simulation only; yfinance (Yahoo) data; no exchange APIs.

> This study upgrades the headline finding from *"no edge on one asset, one cycle"* (the project's #1 caveat) to a cross-sectional claim. It was produced by a 57-agent adversarial workflow (5 analytical lenses → ~30 independent skeptics → synthesis → completeness critic); every load-bearing number below was then **re-derived from the engine by hand** before publishing, and the three numeric errors the critic caught in the first synthesis draft are corrected here.

---

## 1. The question and the falsification criterion

One sharp null: **the Markov regime model has no edge over persistence, and the long/flat trading policy does not beat buy-and-hold net of costs.** "Persistence" is the trivial forecast *tomorrow's regime = today's*.

The null would be **falsified** if the model's 3-state next-regime *balanced accuracy* exceeded persistence by a pre-set margin (edge > 0.02) on a non-trivial share of assets — *and* if the regime-baseline policy beat buy-and-hold for some reason other than mechanically sitting out declines.

Design: 20 assets (9 crypto, 5 single-name equities, 5 macro/index ETFs, 1 FX cross) over ~3y, plus a temporal halving of BTC and SPY. Hyperparameters `(window, k)` are chosen on a 70% train split (grid `window ∈ {10,20,30}`, `k ∈ {0.2,0.35,0.5,0.75}`); **every number below is on the untouched 30% test set**, with 0.3% fee + 0.5% slippage per fill and no lookahead (verified in `evaluation.py`: trailing `rolling`, transition matrix estimated only from states up to day *t*).

## 2. Headline result: 0 / 20

The edge is **exactly +0.000 on all 20 assets** (mean 0.000, max 0.000, zero variance). Not a noisy zero — a structural one.

**Independent re-verification (this is the corrected, self-checked number):** re-running the walk-forward and comparing the model's prediction `argmax(P[today's state])` against the current state day-by-day gives **0 divergences across all 5,425 held-out test predictions** (20/20 assets, 0 each). Identical pointwise predictions ⟹ identical hit-rate, identical per-class recall, identical balanced accuracy. The zero is an identity, not an estimate. (The first synthesis draft reported "3,825 predictions" and asserted the pointwise identity without showing it — both fixed: the true count is 5,425, and the per-day equality is now confirmed directly, not inferred from equal balanced accuracy.)

## 3. Mechanism: why the edge is structurally zero

The model predicts the argmax of the transition-matrix row for today's state. With volatility-adaptive z-score regime labels and highly autocorrelated regimes, each transition row is **diagonal-dominant** (`P[s,s]` is the row max), so the argmax collapses to the current state — and the model *becomes* persistence. The "sideways" regime holds **44.6%–92.0% of days** across assets (AMZN lowest, NVDA highest) with self-transition near 0.98, so on the overwhelming majority of days the diagonal wins. The headline accuracies (ADA 0.905, AAPL 0.891) are entirely regime *stickiness*, not the transition matrix.

## 4. Trading economics: "beats hold 10/20" is a downtrend artifact, not alpha

The baseline is long/flat only — it can de-risk but never lever or short — so it can beat buy-and-hold essentially only when buy-and-hold loses money. The data bear this out mechanically: of the **11 assets where hold < 0, the baseline beat hold in 10**; in all **9 assets where hold ≥ 0, it lost**. All 10 winners have **negative** baseline returns — cash (0%) beats the strategy in every winning case. A naive binomial on 10/20 vs a coin flip gives two-sided *p* = 1.0.

In up-trends it underperforms for two structural reasons: it forgoes upside while flat, and pays 0.8% round-trip on every regime flip (AAPL +11.2% vs hold +43.0% is the worst case).

**The one honest, narrow positive** — risk reduction, not return: in **7 down-trending assets (ETH, DOGE, ADA, LTC, AVAX, LINK, and MSFT — note one is an equity)** the baseline beat hold on return, Sharpe *and* drawdown simultaneously, with **2–8 trades** (ADA drawdown −43.0% vs −83.7%; ETH −50.1% vs −67.5%). This is beta reduction, not skill — and **even here every one of those Sharpes is still negative** (−0.13 to −0.69). A less-bad money-loser is not investable: cash dominates it. The defensive value is real *only* for an investor contractually forced to be long a single asset with no cash option.

## 5. Temporal fragility

The lone non-zero edge anywhere is **early-BTC at −0.167** — the model did *worse* than persistence. Verified by re-run: in 164 test days there was **exactly 1 divergence**, on day 9, the first time "bear" appeared as an origin state. With no prior bear-origin transitions the estimator returns the uniform fallback row `[⅓,⅓,⅓]` (`regime.py`), and `argmax` breaks the tie to index 0 = "bull". Persistence said "bear"; the actual next state *was* "bear". The one day the model used its matrix, it was wrong. Because that error hit a ~2-day class, balanced accuracy fell by 0.5/3 = exactly 0.167.

This is the only time in the whole study the machinery decoupled from persistence — and it hurt.

## 6. Statistical caveats (and claims that did NOT survive verification)

- **Effective N < 20.** The 9 crypto names co-move (~11–13 effective independent assets). With an exactly-zero effect everywhere, correlation inflation can't manufacture a false positive — moot for *this* conclusion, but it means the null is less independently corroborated than "20" suggests.
- **Refuted — "every directional conclusion flips when the window is halved."** Overstated. Some sub-cases invert (BTC early/late returns flip sign), but not all; it is not a universal flip.
- **Refuted — the −0.167 proves the matrix "carries signal."** No. It's a small-sample artifact on a ~2-day class, not evidence of skill.
- **Selection is itself regime-unstable.** early-BTC chose `k=0.35`; late/full chose `k=0.2`; early-BTC showed a ~33pp train/test overfit gap — a single 3-year window is too short to even fix the model's structure.
- **Methodology note:** the grid actually run includes `k=0.2` (the config/CLI default), which was selected on most assets — a *narrower* band than `evaluate()`'s function-signature default of `{0.3,0.5,0.75}`. Use the `evaluate-cli` grid as the source of truth.

## 7. Scope of the claim, and what would overturn it

The claim is narrow and exact: **on this 20-asset, mostly-sideways, single ~3-year basket, the Markov layer is redundant with persistence (edge structurally 0), and the long/flat policy's apparent wins are down-beta, not alpha.**

This is **not** a theorem that the model can never differ from persistence — the early-BTC divergence proves the machinery is live, not a no-op. The result is a property of this basket's high regime autocorrelation. It would be **overturned** by a market with strong short-horizon mean reversion — where, say, "bull" is most often followed by "sideways" rather than "bull", so the diagonal is *not* the row max — in which the argmax leaves the current state and the matrix adds positive, cost-surviving balanced-accuracy edge at genuine regime turning points. This sample never stress-tested the model on a non-dominant-diagonal regime except once, on a 2-day class.

**Honest verdict: realistic point estimate of the Markov layer's skill ≤ 0.** The value the project demonstrably has is a correctly-built, no-lookahead harness that *reports zero when zero is true* — which most backtests fail to do.

---

## Appendix — per-asset results (held-out test)

| Asset | n | model bal-acc | persistence | **edge** | baseline ret | buy&hold ret |
|---|---:|---:|---:|---:|---:|---:|
| BTC-USD | 328 | 0.871 | 0.871 | **+0.000** | −41.1% | −47.7% |
| ETH-USD | 328 | 0.845 | 0.845 | **+0.000** | −17.0% | −47.5% |
| SOL-USD | 328 | 0.849 | 0.849 | **+0.000** | −43.2% | −60.4% |
| XRP-USD | 328 | 0.782 | 0.782 | **+0.000** | −57.3% | −60.8% |
| DOGE-USD | 328 | 0.824 | 0.824 | **+0.000** | −27.8% | −57.8% |
| ADA-USD | 328 | 0.905 | 0.905 | **+0.000** | −34.0% | −77.9% |
| LTC-USD | 328 | 0.867 | 0.867 | **+0.000** | −36.3% | −56.4% |
| AVAX-USD | 328 | 0.871 | 0.871 | **+0.000** | −39.7% | −70.9% |
| LINK-USD | 328 | 0.864 | 0.864 | **+0.000** | −22.5% | −51.8% |
| AAPL | 224 | 0.891 | 0.891 | **+0.000** | +11.2% | +43.0% |
| NVDA | 224 | 0.870 | 0.870 | **+0.000** | +19.8% | +19.8% |
| TSLA | 224 | 0.844 | 0.844 | **+0.000** | +3.8% | +27.0% |
| MSFT | 224 | 0.879 | 0.879 | **+0.000** | −10.2% | −19.5% |
| AMZN | 224 | 0.781 | 0.781 | **+0.000** | −10.9% | +8.7% |
| SPY | 224 | 0.877 | 0.877 | **+0.000** | +6.4% | +17.8% |
| QQQ | 224 | 0.811 | 0.811 | **+0.000** | +16.3% | +26.9% |
| GLD | 224 | 0.716 | 0.716 | **+0.000** | +18.7% | +28.1% |
| TLT | 224 | 0.706 | 0.706 | **+0.000** | −9.8% | +2.7% |
| USO | 224 | 0.606 | 0.606 | **+0.000** | +75.0% | +76.0% |
| EURUSD=X | 233 | 0.746 | 0.746 | **+0.000** | −9.1% | −1.9% |

**Temporal split:**

| Window | edge vs persistence | baseline ret | buy&hold ret |
|---|---:|---:|---:|
| BTC-USD · early | **−0.167** | +23.3% | +60.4% |
| BTC-USD · late | +0.000 | −29.3% | −28.2% |
| SPY · early | +0.000 | +8.9% | +10.8% |
| SPY · late | +0.000 | −3.4% | +6.5% |

*Reproduce: `python scripts/robustness_study.py` (pins each pull to `data/snapshots/`, writes `results/robustness/`).*
