"""The walk-forward's incremental transition counts must be pointwise identical
to re-estimating the matrix from the whole prefix every day — the original
(quadratic) formulation. Bit-exact equality is required: the published zeros
are pointwise identities between the argmax forecast and persistence, so even a
one-ulp drift in a forecast row could flip a tie and change a receipt.

Covered: None warm-up prefixes, None/unknown labels mid-sequence (the estimator
chains the next valid label to the last valid one), the disclosed uniform-row
fallback on a state's first appearance, and walks that start at 0 (no priming)
or mid-series (everything before `start` primed).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.data_cache import load_or_fetch
from app.services.evaluation import _TransitionCounter, _walk_forward, evaluate
from app.services.regime import STATE_ORDER, STATE_TO_IDX, estimate_transition_matrix

BULL, BEAR, SIDE = STATE_ORDER


def _reference(states, start, end):
    """The pre-optimisation formulation, kept verbatim as the oracle."""
    probs, actual, cur_idx = [], [], []
    n = len(states)
    for t in range(start, min(end, n - 1)):
        cur, nxt = states[t], states[t + 1]
        if cur is None or nxt is None or cur not in STATE_TO_IDX or nxt not in STATE_TO_IDX:
            continue
        observed = [s for s in states[: t + 1] if s in STATE_TO_IDX]
        if len(observed) < 2:
            continue
        p = estimate_transition_matrix(pd.Series(observed))
        probs.append(p[STATE_TO_IDX[cur]])
        actual.append(STATE_TO_IDX[nxt])
        cur_idx.append(STATE_TO_IDX[cur])
    return probs, actual, cur_idx


def _assert_identical(states, start, end):
    returns = pd.Series(np.zeros(len(states)))
    wf = _walk_forward(returns, states, start, end, 0.3, 0.5)
    ref_probs, ref_actual, ref_cur = _reference(states, start, end)
    assert len(wf["pred_probs"]) == len(ref_probs)
    for got, want in zip(wf["pred_probs"], ref_probs):
        assert np.array_equal(got, want), (got, want)   # bit-exact, not allclose
    assert wf["actual_idx"] == ref_actual
    assert wf["cur_idx"] == ref_cur
    return wf


def _sticky(n, seed, p_stay=0.85):
    rng = np.random.default_rng(seed)
    cur = int(rng.integers(3))
    out = []
    for _ in range(n):
        out.append(STATE_ORDER[cur])
        if rng.random() > p_stay:
            cur = int(rng.integers(3))
    return out


def test_matches_from_scratch_estimator_with_warmup_prefix():
    states = [None] * 19 + _sticky(400, seed=1)
    _assert_identical(states, start=60, end=len(states))
    _assert_identical(states, start=0, end=len(states))        # no priming at all
    _assert_identical(states, start=250, end=300)              # mid-series window


def test_matches_with_none_and_unknown_labels_mid_sequence():
    states = _sticky(300, seed=2)
    for i in (40, 41, 42, 120, 199, 250):
        states[i] = None
    states[130] = "weird"                                      # not a known state
    wf = _assert_identical(states, start=50, end=len(states))
    assert wf["pred_probs"]                                    # something was scored


def test_uniform_row_fallback_on_first_appearance_is_identical():
    # bear never appears until the test region: its row has no outgoing
    # transitions when it is first the current state -> uniform 1/3 fallback.
    states = [BULL] * 100 + [SIDE] * 50 + [BEAR, BEAR, BULL] + [SIDE] * 30
    wf = _assert_identical(states, start=120, end=len(states))
    uniform = np.ones(3) / 3
    first_bear = wf["cur_idx"].index(STATE_TO_IDX[BEAR])
    assert np.array_equal(wf["pred_probs"][first_bear], uniform)


def test_forecast_rows_are_snapshots_not_views():
    # rows handed to callers must not change as later days are counted
    counter = _TransitionCounter()
    for s in (BULL, BULL, BEAR):
        counter.observe(s)
    row = counter.row(STATE_TO_IDX[BULL])
    before = row.copy()
    for s in (BULL, BULL, BULL):
        counter.observe(s)
    assert np.array_equal(row, before)


def test_counter_matches_estimator_row_by_row():
    states = _sticky(500, seed=3)
    counter = _TransitionCounter()
    for t, s in enumerate(states):
        counter.observe(s)
        if t == 0:
            continue
        p = estimate_transition_matrix(pd.Series(states[: t + 1]))
        for i in range(len(STATE_ORDER)):
            assert np.array_equal(counter.row(i), p[i])


def test_pinned_btc_record_reproduces_exactly():
    # The committed receipt was produced by the from-scratch estimator; the
    # incremental walk must reproduce every accuracy number bit-for-bit.
    rec = json.loads(Path("results/BTC-USD_4a150b23.json").read_text())
    history, src = load_or_fetch("BTC-USD", years=3)
    assert src.startswith("cache:")          # pinned snapshot, never a live pull
    report = evaluate(
        history, symbol="BTC-USD", train_frac=0.7,
        grid_windows=(10, 20, 30), grid_k=(0.2, 0.35, 0.5, 0.75),
        fee_pct=0.3, slippage_pct=0.5,
    )
    assert report.data_hash == rec["data_hash"]
    assert asdict(report.accuracy) == rec["accuracy"]
    assert (report.chosen_window, report.chosen_k) == (rec["chosen_window"], rec["chosen_k"])
    assert [asdict(p) for p in report.policies] == rec["policies"]
