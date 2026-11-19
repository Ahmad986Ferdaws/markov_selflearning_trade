# 21 — Glossary

Terms the current docs use with a specific meaning. Each entry says where the term is
defined in code or in the docs.

**Regime.** One of three daily labels: `bull`, `bear`, `sideways`. In the default z-score
rule a day is bull when the trailing-window mean return exceeds `k` times the trailing
standard deviation, bear when it is below `-k` times it, sideways otherwise. The rule
lives in one function, `label_state` in `app/services/regime.py`, and is causal by
construction: the window ends at the day being labelled.

**Walk-forward.** The evaluation loop in `app/services/evaluation.py`. At each day *t*
the transition matrix is estimated from states up to and including *t*, a prediction is
made for *t + 1*, and every policy chooses its position for *t + 1* from the same
information. Nothing from *t + 1* onward is visible at *t*. See docs/14.

**Train / test split.** The first 70% of days (by default) are the train region, used
only to choose the window and `k`. Every reported number comes from the held-out test
region the selection never touched.

**Persistence bar.** The baseline "tomorrow's regime = today's". It is the number every
accuracy claim is measured against, because a diagonal-dominant transition matrix
reproduces it exactly. The project's headline is that the model ties this bar on every
asset studied. See docs/16 and docs/19.

**Naive predictor.** Always predict the majority class from the train region. A weaker
bar than persistence; reported for context.

**Balanced accuracy.** Mean per-class recall over the classes that occur. It refuses to
reward "always predict sideways", which raw hit-rate would.

**Switch day.** A held-out day where tomorrow's regime differs from today's. Persistence
scores zero on these by construction, so they are the only days any predictor can beat
it. The engine reports how many there were, the model's recall on them, and how many
times the model predicted a change at all (its *switch attempts*). See docs/19.

**Divergence.** A day where the model's argmax prediction differs from today's state,
that is, from what persistence would predict. The studies count these pointwise; the
published total is 0 across 16,773 held-out predictions.

**Honesty gate.** A check the engine runs on its own output that appends a warning to the
report when it fires: regime-mix imbalance, no edge over naive, no edge over persistence,
possible overfit, constant-regime window, agent acting on low-confidence forecasts, and
all accuracy being stay-day credit. Gates only append; they never change a number.

**Uniform-row fallback.** When a state has never been left, its transition row is set to
1/3 each. Disclosed in `estimate_transition_matrix`; on every scored day of the published
studies this branch never fired (per-step self-probability ≥ 0.600). See docs/19.

**Policy.** A function from a `PolicyContext` (today's state, yesterday's position, the
forecast distribution, the day index) to a target exposure in [0, 1]. Baseline, buy-and-hold,
and the optional LLM agent are all policies through the identical fill and cost path.

**Pinned snapshot.** A yfinance pull saved to `data/snapshots/<symbol>_<years>y.pkl` and
committed. Runs read the pin, never the network, so the result is reproducible from a
fresh clone. `history_hash` names the exact data a run used.

**Run record.** The JSON a run writes under `results/`, keyed by the hash of its complete
report. Identical reports reuse their path; different reports on the same data remain
separate evidence.

**Cache-only mode.** The agent replay mode (`--cache-only`, `AGENT_CACHE_ONLY`) that serves
only the persisted prompt-hash response cache and never calls a provider. Any public
surface must use it.

**Call budget.** `AGENT_MAX_CALLS` (default 64): the hard ceiling on unique billed LLM calls
per run. When it is exhausted the agent holds its prior position and the report says so.

**Data hash.** `history_hash` of the price series a run used: sha256 over the closes rounded
to eight decimals, each paired with its date (`app/services/data_cache.py`). Every run record
carries it as `data_hash`, and the headline record `results/BTC-USD_4a150b23.json` is named by
its first eight characters. Same hash, same data; a refreshed snapshot is a new hash.

**Stay-day credit.** Accuracy earned on days the regime did not change, which persistence also
gets for free. Gate 7 in `app/services/evaluation.py` fires when switch recall is zero: the
hit-rate is then entirely stay-day credit and the report says so in words. See docs/19,
section 2.

**Decision provenance.** How each agent position was obtained, counted in `AgentStats`
(`app/services/agent_policy.py`): `strict_json` (clean JSON), `salvaged_labeled` (a labelled
value found in prose), `salvaged_single_number` (a lone in-range number), `unparseable_hold`
(nothing usable; held), plus `errors` (the call raised), `budget_holds` (call budget spent) and
`cache_only_holds` (cache-only mode, key absent). Reported as a histogram, never as accuracy.

**Diagonal dominance.** A transition row whose self-transition probability exceeds 1/2. The
off-diagonals then sum to less than 1/2, so none can win and `argmax` equals persistence; when
every visited row is dominant the identity is forced, not incidental. Checked per run in
`scripts/model_logic_study.py` and in `tests/test_instrument_validation.py`; see docs/19,
section 3.

**Sub-period.** The early and late halves of the BTC-USD and SPY three-year windows, evaluated
as separate runs by `scripts/robustness_study.py` (`results/robustness/BTC-USD:early.json` and
its siblings) to test temporal fragility. The one non-zero edge in the whole study, early BTC
at −0.167, is here. See docs/16, section 5.

**Positive control.** A synthetic state sequence with planted, predictable switch structure
(alternation, cycles, noisy alternation) run through the production predictor, to prove the
instrument reads non-zero when non-zero structure exists. `tests/test_instrument_validation.py`
runs them in CI; see docs/19, section 1.
