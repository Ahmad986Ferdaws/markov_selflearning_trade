# 18 — Long-history study: the null survives 20 years

**Purpose.** [docs/16](16-robustness-study.md) closed with the project's #1 stated limitation:
*"one ~3-year window per asset; a longer multi-cycle history is the next test."* This is that test.

**Protocol.** Identical to the robustness study — same engine (`app/services/evaluation.evaluate`),
same grid (`windows (10, 20, 30) × k (0.2, 0.35, 0.5, 0.75)`), same 70/30 train/test split, same
costs (0.3% fee + 0.5% slippage) — run on the longest history Yahoo provides per asset
(`scripts/long_history_study.py`, snapshots pinned to `data/snapshots/*_{10,12,20}y.pkl`,
records in `results/long_history/`). The 20-year windows span the **2008 financial crisis, the
2010s bull market, the COVID crash, and the 2022 bear** — multiple full regimes of every kind.

## Result: edge = +0.0000 on 8/8, zero divergences in 11,348 predictions

| Asset | Span | Held-out n | Edge (hit) | Edge (bal) | Divergences | Baseline | Buy & hold |
|---|---|---:|---:|---:|---:|---:|---:|
| BTC-USD | 2014-09 → 2026-07 | 1,299 | +0.0000 | +0.0000 | 0 / 1,299 | +186.4% | +282.4% |
| ETH-USD | 2017-11 → 2026-07 | 954 | +0.0000 | +0.0000 | 0 / 954 | +29.5% | −13.4% |
| SPY | 2006-07 → 2026-07 | 1,508 | +0.0000 | +0.0000 | 0 / 1,508 | +89.8% | +146.5% |
| QQQ | 2006-07 → 2026-07 | 1,508 | +0.0000 | +0.0000 | 0 / 1,508 | +43.8% | +171.4% |
| AAPL | 2006-07 → 2026-07 | 1,508 | +0.0000 | +0.0000 | 0 / 1,508 | +148.4% | +271.8% |
| GLD | 2006-07 → 2026-07 | 1,508 | +0.0000 | +0.0000 | 0 / 1,508 | +92.6% | +109.8% |
| TLT | 2006-07 → 2026-07 | 1,508 | +0.0000 | +0.0000 | 0 / 1,508 | −42.6% | −41.1% |
| EURUSD=X | 2006-07 → 2026-07 | 1,555 | +0.0000 | +0.0000 | 0 / 1,555 | −13.2% | −4.3% |

Aggregate (from `results/long_history/summary.json`): **8/8 assets, 11,348 held-out predictions,
0 divergences, max |edge| = 0.0**. Combined with the 3-year robustness study (5,425 predictions,
20/20 assets), the pointwise identity `argmax(P[today]) == today` now stands verified on
**16,773 predictions with zero exceptions**.

## What the long window adds

1. **The identity is not a window artifact.** Two decades of regime churn — crisis, bull, crash,
   bear, recovery — never once produced a transition row whose argmax left the current state.
   Diagonal dominance is a structural property of z-score regimes on daily bars, not a fluke of
   2023–2026 data.

2. **The "beats buy-hold" caveat is confirmed with data.** Docs 15/16 warned that the baseline
   beating buy-and-hold was a downtrend artifact. On these mostly-uptrending long windows the
   baseline **underperforms** buy-and-hold on **7/8 assets** (it sits in cash during rallies and
   pays trading costs; e.g. QQQ +43.8% vs +171.4%). The single exception is ETH (+29.5% vs
   −13.4%), whose held-out window is net-down — the artifact, again, in the predicted direction.
   Treat the sign of *the edge* as the finding; never the policy-vs-policy ordering.

3. **The honest posture generalizes.** The instrument was built to report null when null is true;
   given 4× more data and windows 6× longer, it reports the same null, pointwise, everywhere.

## Reproduce

```bash
.venv/bin/python scripts/long_history_study.py
```

Snapshots are pinned, so a re-run is deterministic; delete a snapshot (or pass fresh symbols) to
re-pull. Per-asset run records with full reports: `results/long_history/*.json`.
