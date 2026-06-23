# Phase A — Results (honest findings)

Daily walk-forward evaluation of a 3-state Markov regime model on BTC-USD. Run: `evaluate-cli`. Code: `app/services/evaluation.py`. Audited (see session log in STATUS).

## Setup
- **Data:** BTC-USD daily, ~3y (yfinance).
- **Split:** 70% train / 30% test. Hyperparameters (`window`, `k`) chosen on **train only**, by balanced accuracy, among configs that produce ≥2 well-represented regimes. All numbers below are **held-out test**.
- **Costs:** 0.3% fee + 0.5% slippage, applied on turnover.
- **Chosen config:** window=30, k=0.2 (z-score thresholds: `±k·rolling_std`).

## Result (held-out test, n=328 days)

| | Hit-rate | Balanced-acc | Log-loss |
|---|---|---|---|
| **Markov model** | 90.9% | 87.1% | 0.338 |
| Naive (majority) | 72.0% | 33.3% | 0.870 |
| **Persistence ("tomorrow = today")** | **90.9%** | **87.1%** | — |

Regime mix (test): sideways 72%, bear 14%, bull 14%.

| Policy | Return | Sharpe | MaxDD | Trades | Cost |
|---|---|---|---|---|---|
| baseline (regime) | −41.1% | −1.87 | −42.9% | 6 | 4.8% |
| buy & hold | −47.8% | −1.18 | −51.2% | 1 | 0.8% |

## The findings (honest)

1. **The Markov model has no predictive skill over persistence.** It predicts next-day regime at 90.9% — but simply saying "tomorrow's regime = today's" scores *identically* (90.9%). The transition matrix adds essentially nothing; the high accuracy is **regime autocorrelation**, not forecasting skill. The tool detects and warns about this automatically.

2. **It beats the *majority* baseline but that's a low bar.** 90.9% vs 72% looks impressive until you see persistence matches it — the "win" is just that regimes cluster in time.

3. **No trading edge, net of costs.** In this (down-trending) test window both policies lose. The regime baseline lost less than buy-and-hold (it sat out some downside) but paid 6× the costs to do it. Neither makes money.

## Why this is the *right* outcome
The project's value is an honest measurement, not a profitable bot. Phase A delivers exactly that: a daily, leakage-checked, train/test-split harness that produces a credible result and **refuses to flatter the model** — it actively flags that the Markov layer is decorative and that there's no cost-surviving edge. That honesty is the portfolio payoff.

## Caveats (don't overclaim)
- One asset, one ~3-year period (≈ one market cycle). Not a general claim.
- States come from overlapping rolling windows → consecutive days are highly correlated, which is *why* persistence is so strong.
- The test window was net down-trending; a different period could flip the baseline-vs-buy-hold ordering. Reproduce across multiple periods before any broader statement.

## Next
- **Phase B:** one fill engine + reproducibility (freeze config, persist decisions).
- **Phase C:** plug the LLM agent into this same daily harness (anthropic|ollama switch) and add it as a third policy — does an LLM beat the regime baseline *and* buy-and-hold here? Then write up.
