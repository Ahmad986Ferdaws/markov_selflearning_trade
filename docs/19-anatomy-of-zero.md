# 19 — The anatomy of zero: instrument validation, the theorem, and what the model DOES know

Seven questions a serious reviewer would ask about a harness whose every reading is 0.000 —
answered with code, controls, and 28 asset runs (`tests/test_instrument_validation.py`,
`scripts/model_logic_study.py`, `scripts/logic_depth_study.py`,
`results/model_logic/` + `results/logic_depth/`).

## 1. Can the instrument detect edge at all? (positive control)

A pipeline that always prints zero might just be broken. So we planted synthetic state
sequences with *known* structure and ran them through the **exact production path**
(`_walk_forward` → `estimate_transition_matrix` → `_accuracy`):

| Planted structure | Model hit | Persistence hit | Verdict |
|---|---:|---:|---|
| Strict bull/bear alternation | > 95% | < 5% | edge detected (+90pp), every argmax off-diagonal |
| Period-3 cycle (bull→side→bear) | > 95% | < 5% | edge detected |
| 85% alternation / 15% noise | — | — | edge > +50pp detected |
| Sticky chain, p_stay = 0.9 | = persistence | = persistence | **exact tie, 0 divergences** — the identity reproduces |

The same code shows a huge edge when switches are predictable and an exact tie when states
are sticky. **The zeros on real markets are measurements, not malfunctions.**
(`tests/test_instrument_validation.py`, in CI on every push.)

One structural note: a *full-pipeline* anti-persistent control is impossible, because
`causal_states` labels on a **rolling mean** — day-to-day alternation in returns averages
out before it can become alternating labels. The labeler manufactures stickiness by
construction (docs/15's disclosure, now mechanized), which is half of why zero is inevitable.

## 2. Where would skill even live? (switch-day anatomy)

Persistence scores 0% on regime-change days *by definition* — so change days are the only
territory where any predictor can beat it. The engine now measures this directly
(`AccuracyMetrics.n_switch / switch_recall / switch_attempts`, **Gate 7**):

- BTC-USD held-out window: **30 regime changes in 328 days.**
- The model's switch attempts: **0**. Its switch recall: **0/30.**
- Decomposition of the famous 90.9%: **(298 hits on 298 stay days + 0 hits on 30 change
  days) / 328 days = 298/328 = 90.9%.** The number is persistence, restated.

Gate 7 now says this in every report: *"a hit-rate earned only on stay days is
persistence, restated."*

## 3. Why must the argmax stay home? (the theorem, verified)

**Claim.** If the current state's self-transition probability exceeds 1/2, the row's argmax
is the current state (off-diagonals sum to < 1/2, so no single one can exceed the diagonal).
Hence: the argmax forecast is *forced* to equal persistence whenever every visited state has
p_stay > 1/2 — equivalently, expected run length > 2 days.

**Measurement** (28 runs, each at its own chosen `(window, k)`):

- **28/28 runs: every visited state's row is diagonally dominant.**
- Global minimum visited diagonal: **0.5806** (USO) — above the 1/2 bound on every run.
- The single non-dominant row in the whole study (NVDA, k=0.35) is the **uniform fallback
  for a state with zero occurrences** — NVDA produced no bear-labeled day in 3 years, and a
  state that never occurs can never be "today," so the phantom row cannot break the identity.
  (Same mechanism as docs/16's early-BTC −0.167: a uniform row on a 2-day class.)

So the zero edge is structurally **forced, step by step**: whenever the estimated row for
today's state is diagonally dominant, the forecast must equal persistence. This is measured
at the strongest possible granularity — for every one of the 16,773 held-out decisions, the
**causal prefix matrix actually used at that decision** had self-probability ≥ **0.600**
(global minimum across all 28 runs; `logic_depth` `min_prefix_self_prob`, 28/28 runs above
the 1/2 bound throughout). The full-sample fitted matrices agree (28/28 dominant, min
0.5806), and docs/16's early-BTC −0.167 remains the one intermediate-matrix exception ever
observed — a uniform fallback row on a 2-day class, outside these 28 runs. To beat
persistence, a first-order chain would need a regime that flips at least every other day
on average (expected run length ≤ 2) — and even then, with the exit mass concentrated on a
single successor. These are observed across 16,773 predicted held-out days — though note
28 runs over 20 assets, with 8 symbols contributing two overlapping horizons and the
crypto assets co-moving, are far fewer *independent* observations than the day count
suggests.

## 4. Does duration rescue it? (the obvious next model, tested honestly)

A first-order chain can't time switches — but a **semi-Markov / duration-aware** model could,
*if* the hazard h(d) = P(switch tomorrow | d days in regime) climbs above 1/2. Measured on
train regions across all 28 runs:

- The per-run maximum hazard (cells with ≥15 obs) lies in **0.13–0.47 for 23 of 28 runs**;
  two runs had no cell with sufficient support, and three exceeded it (AMZN 0.50, XRP 0.59,
  DOGE 0.63). (These are point estimates screened over ~220 cells: the two crossings are
  within binomial noise of 1/2 — DOGE is 12/19, one-sided p ≈ 0.18 — and conversely cells
  near 0.47 are not significantly *below* 1/2. The hazard evidence bounds how often the
  rule can fire; it cannot rule out true hazards modestly above or below 1/2.)
- A fully **causal** duration-augmented predictor (incremental counts, predicts a switch only
  when the current duration's hazard estimate > 1/2 with ≥15 obs) **fired on 5/28 runs — 23
  attempts total across 16,773 held-out days.**
- Resulting edges vs persistence: **−1.83pp (XRP, 10 attempts) to +0.61pp (DOGE, 10
  attempts)**; the other three firing runs made 1 attempt each. The mean edge is slightly
  negative (−0.07pp across all 28 runs). With 1–10 attempts per run, neither the sign nor
  the size of these edges is informative — the reliable finding is that **the rule almost
  never fires**, not the direction of its edge when it does.

**Verdict: duration information does not rescue the model.** The null deepens — not just
first-order Markov, but the natural duration-dependent extension fails to beat "tomorrow =
today" on these regimes. *Caveat: this is one exploratory model with a pre-fixed rule (no
tuning, no search); it is reported whatever it says, but it is not a swept model family —
a claim about all semi-Markov models would be overreach.*

## 5. How tight is the zero? (evidence of absence, quantified)

`scripts/logic_depth_study.py` **independently re-derives** the pooled receipt with an
incremental estimator (self-checked to equality against the engine's
`estimate_transition_matrix` before scoring): **16,773 held-out predictions, 0 divergences**
— the same total as the two studies, reproduced from raw snapshots in one pass.

With zero events in 16,773 paired trials, the exact one-sided 95% upper bound on the true
divergence rate is `1 − 0.05^(1/16773)` ≈ **0.018%**. This is not "we failed to find a
difference": at 95% confidence, model and persistence decisions can differ on at most
**1 day in ~5,600**. Absence of edge, bounded. (The bound treats days as independent
trials; with the nested horizons and cross-correlated assets noted in §3, the honest bound
on truly independent evidence is looser — the qualitative conclusion is unchanged.)

## 6. Does more memory help? (second-order chains)

A memory-2 chain conditions on the last *two* states — e.g. "just switched into sideways"
vs "long sideways." Across all 28 runs (causal incremental counts, ≥15-obs support,
predict-then-update):

- Exactly **one** switch-dominant context appeared in the entire study (DOGE).
- The memory-2 predictor fired **20 times across all 28 runs**; edges vs persistence
  ranged **−1.22pp to +0.61pp** on the handful of runs where it fired at all — attempt
  counts far too small for the sign to mean anything. The reliable finding, as with
  duration: it almost never fires.

Neither duration (§4) nor an extra state of memory rescues the decisions.

## 7. What the model DOES know (probabilistic skill — an honest positive)

Identical argmax does **not** mean identical forecasts. Scoring the model's *probability
vector* against two causal persistence-equivalent baselines — "sticky-uniform" (global
stay-rate on today's state, exits split evenly) and the stronger "sticky-empirical" (exits
follow the global switch-target distribution) — on the same held-out days:

- **Log-loss: the Markov row wins 28/28 runs against both baselines** (mean improvement
  +0.046 nats/day vs sticky-uniform, **+0.029 vs sticky-empirical**). The 28 runs are not
  independent (§3's caveat applies), but the sign is positive on all 20 distinct assets at
  every horizon tested.
- **Brier: 28/28 vs sticky-uniform, 25/28 vs sticky-empirical** — the effect lives mostly
  in calibration (log-loss) rather than decision-adjacent probability mass.

So the honest summary is sharper than "the model is worthless": **the transition matrix
carries real, consistently measurable probabilistic information — state-specific stay rates
and exit preferences — but that information never once crosses the argmax decision
threshold.** The model *knows* something; it just never changes the answer. A long/flat
policy consumes answers, not calibration — which is exactly why the trading edge is zero
while the log-loss edge is not. (Both facts are reported; neither is hidden in the other.)

## What this changes

The project's claim upgrades from *"we observed zero edge"* to:

1. the instrument **demonstrably detects planted first-order switch structure**
   (alternation, cycles, noisy alternation — positive controls in CI) and reproduces the
   zero-edge identity on sticky chains,
2. the zero is **arithmetically located** (0-for-30 on the only days that count),
3. the zero is **structurally forced at every step** (the causal row used at each of the
   16,773 decisions had self-probability ≥ 0.600 — never once below the 1/2 bound),
4. the zero is **bounded** (divergence rate < 0.018% at 95%, on 16,773 re-derived trials,
   independence caveats stated),
5. the **obvious fixes don't work** (the duration rule fires on only 5/28 runs — 23
   attempts in 16,773 days — with uninformative edges; memory-2 fires 20 times), and
6. what the model *does* contribute is **named and measured** (calibration, 28/28 —
   never decisions).

Reproduce: `pytest tests/test_instrument_validation.py` · `python scripts/model_logic_study.py`
· `python -m scripts.logic_depth_study`
