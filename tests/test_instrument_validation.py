"""Instrument validation — prove the harness CAN detect edge before trusting its zeros.

A measurement of 0.000 is only meaningful if the instrument reads non-zero when
non-zero structure exists. These are the positive controls: synthetic state
sequences with PLANTED predictable-switch structure, run through the exact
production path (`_walk_forward` -> `estimate_transition_matrix` -> `_accuracy`).
The engine must show a large edge over persistence there — and reproduce the
zero-edge identity on synthetic sticky sequences. Together they map the
boundary: the zeros on real markets are properties of the markets, not the code.

Also encodes the diagonal-dominance theorem: if every state's self-transition
probability exceeds 1/2 (equivalently, expected run length > 2 days), the
argmax forecast IS persistence — divergence requires a fast-flipping state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.data_cache import load_or_fetch
from app.services.evaluation import _accuracy, _train_freq, _walk_forward, evaluate
from app.services.regime import STATE_ORDER, STATE_TO_IDX, estimate_transition_matrix

BULL, BEAR, SIDE = STATE_ORDER


def _score(states: list[str]):
    """Run a synthetic state sequence through the production predictor + scorer."""
    n = len(states)
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0, 0.01, n))  # returns irrelevant to accuracy
    split = int(n * 0.7)
    wf = _walk_forward(returns, states, split, n, 0.3, 0.5)
    acc = _accuracy(wf["pred_probs"], wf["actual_idx"], _train_freq(states, 0, split), wf["cur_idx"])
    divergences = sum(
        1 for dist, cur in zip(wf["pred_probs"], wf["cur_idx"]) if int(np.argmax(dist)) != cur
    )
    return acc, divergences


# --- positive controls: planted structure MUST be detected ---------------------
def test_detects_planted_alternation():
    # bull/bear strict alternation: persistence is always wrong, the transition
    # matrix is perfectly informative. The engine must find the edge.
    states = [BULL, BEAR] * 200
    acc, div = _score(states)
    assert acc.hit_rate > 0.95                    # model nails the switches
    assert acc.persistence_hit_rate < 0.05        # persistence dies on alternation
    assert acc.hit_rate - acc.persistence_hit_rate > 0.9
    assert acc.switch_recall > 0.95               # skill lives on switch days
    assert acc.switch_attempts > 0                # and the model actually tried
    assert div == acc.n                           # every argmax left the diagonal


def test_detects_planted_cycle():
    # bull -> sideways -> bear -> bull ... period-3 cycle
    states = [BULL, SIDE, BEAR] * 140
    acc, div = _score(states)
    assert acc.hit_rate > 0.95
    assert acc.persistence_hit_rate < 0.05
    assert div > 0                                # the argmax leaves the diagonal


def test_detects_noisy_structure():
    # 85% follow the alternation, 15% random noise: edge must still be large.
    rng = np.random.default_rng(7)
    states = []
    cur = BULL
    cycle = {BULL: BEAR, BEAR: BULL, SIDE: BULL}
    for _ in range(600):
        states.append(cur)
        cur = cycle[cur] if rng.random() < 0.85 else str(rng.choice(list(STATE_ORDER)))
    acc, _ = _score(states)
    assert acc.hit_rate - acc.persistence_hit_rate > 0.5


# --- negative control: sticky synthetic reproduces the zero-edge identity ------
def test_sticky_synthetic_reproduces_identity():
    # Markov-generated states, p_stay = 0.9. With this pinned seed every
    # prefix-estimated row stays diagonally dominant throughout the walk (min
    # self-prob ~0.89, all states well-visited), so the engine must output
    # EXACTLY persistence — zero divergences, identical rates. (The exactness is
    # a property of the seed's realized path, not of p_stay alone.)
    rng = np.random.default_rng(3)
    states = [SIDE]
    for _ in range(999):
        cur = states[-1]
        if rng.random() < 0.9:
            states.append(cur)
        else:
            states.append(rng.choice([s for s in STATE_ORDER if s != cur]))
    acc, div = _score(states)
    assert div == 0                                        # pointwise identity
    assert acc.hit_rate == acc.persistence_hit_rate        # exact tie
    assert acc.balanced_accuracy == acc.persistence_balanced
    assert acc.switch_attempts == 0                        # never predicts a change


# --- the theorem, stated on the estimated matrices -----------------------------
def test_diagonal_dominance_theorem():
    # Sticky sequence -> every self-transition > 1/2 -> argmax must stay home.
    rng = np.random.default_rng(3)
    sticky = [SIDE]
    for _ in range(999):
        cur = sticky[-1]
        sticky.append(cur if rng.random() < 0.9 else rng.choice([s for s in STATE_ORDER if s != cur]))
    P = estimate_transition_matrix(pd.Series(sticky))
    for i in range(len(STATE_ORDER)):
        assert P[i][i] > 0.5
        assert int(np.argmax(P[i])) == i
    # Alternating sequence -> self-transition ~0 -> argmax must leave the diagonal.
    P2 = estimate_transition_matrix(pd.Series([BULL, BEAR] * 200))
    assert P2[STATE_TO_IDX[BULL]][STATE_TO_IDX[BULL]] < 0.5
    assert int(np.argmax(P2[STATE_TO_IDX[BULL]])) == STATE_TO_IDX[BEAR]


# --- the real thing: gate 7 fires on the pinned BTC snapshot -------------------
def test_gate7_fires_on_real_market_data():
    history, src = load_or_fetch("BTC-USD", years=3)
    # refuse to fall back to a live download: this test pins data-dependent
    # invariants and is only meaningful against the committed snapshot
    assert src.startswith("cache:"), "pinned snapshot missing — refusing to test live data"
    rep = evaluate(history, symbol="BTC-USD")
    assert rep.accuracy.n_switch > 0
    assert rep.accuracy.switch_attempts == 0         # never once predicted a change
    assert rep.accuracy.switch_recall == 0.0
    assert any("never predicted a regime change" in w for w in rep.warnings)
