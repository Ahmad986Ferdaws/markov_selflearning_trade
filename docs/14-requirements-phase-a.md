# Requirements — Phase A (model + accuracy)

Goal: make the model actually produce signal on a sane clock, and produce a **real accuracy number on held-out data**. Daily only. No intraday, no provider switch yet.

Work file: mostly `app/services/regime.py` + a new `app/services/evaluation.py` (or extend `comparison.py`).

---

## R0 — Decide what "accuracy" means (do this first)
Measure and report BOTH, separately:
- **Prediction accuracy** — does the model predict tomorrow's regime? (hit-rate + log-loss)
- **Trading edge** — does acting on it beat buy-and-hold, net of costs?
Don't merge them into one number.

## R1 — Daily engine (kills the clock mismatch)
- Walk the daily history **bar by bar**: one decision per day, signal date == trade date.
- Remove the 15-second poll loop from the evaluation path entirely.
- Acceptance: a run produces one trade-decision per trading day, not hundreds per day.

## R2 — Rescale the thresholds (make it actually trade)
- In `define_states`, set bull/bear from the **return distribution**, not fixed ±2%/day.
- Rule: `bull if rolling_mean > +k·rolling_std`, `bear if < −k·rolling_std`, else sideways. `k` configurable (default ~0.5).
- Acceptance: regime mix is balanced — **no single state > ~80%** of days (today "sideways" is ~99%).

## R3 — Add buy-and-hold (the yardstick)
- Third policy: buy on day 1, hold to the end, net of one entry cost.
- Every report compares **agent vs baseline vs buy-and-hold**.

## R4 — Prediction-accuracy metric (the real "accuracy")
- At each walk-forward step: predicted next state = argmax of `p_next`; compare to the **actual** next-day state.
- Compute: **hit-rate (%)**, **multiclass log-loss** (or Brier).
- Compare against a dumb baseline: "always predict the most common state." Model must beat that to mean anything.

## R5 — Train/test split (stops overfitting)
- Split history: **train = first ~70%** (fit `k`, window), **test = last ~30%** (held out).
- Tune parameters ONLY on train. Report accuracy ONLY on test.
- Never let test data touch parameter selection.

## R6 — Parameter stability check
- Try a few `window` (e.g. 10/20/30) and `k` values.
- Flag if the best-on-train params fall apart on test → that's overfitting, and it's a finding to report, not hide.

## R7 — One output
One command prints:
1. Regime mix (% per state),
2. Prediction accuracy on **test** (hit-rate + log-loss vs naive),
3. Three-way trading comparison net of costs,
4. Validation gate (must show >1 regime, finite metrics).

---

## Acceptance gate (Phase A)
A single daily run prints: a **balanced regime mix**, a **real prediction-accuracy number on held-out data** (that beats the naive predictor or honestly reports it doesn't), and the **three-way net-of-cost comparison**. The model trades on a meaningful fraction of days (not ~99% HOLD).

## Explicitly NOT in Phase A
- Provider switch (Claude/Ollama) → Phase C.
- Reproducibility persistence + merging the two fill engines → Phase B.
- Any UI, DEX, TradingView, multi-asset → later.
