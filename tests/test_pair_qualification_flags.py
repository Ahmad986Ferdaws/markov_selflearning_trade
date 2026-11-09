"""Pins for kalman.pairs.qualify_pair: the overlap flag at exactly min_overlap,
the finite-bar count behind it, the verdict's three notes, and the half-life
edge for a residual whose AR(1) coefficient sits at one.

tests/test_kalman_research.py::test_pair_diagnostics_use_proper_engle_granger
pins that the Engle-Granger p-value is the proper one; this file pins the flags
and the verdict text that the walk-forward report prints from it.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from kalman.data import align_pair, synthetic_pair
from kalman.pairs import PairDiagnostics, qualify_pair

# statsmodels' adfuller deprecation is raised from kalman/pairs.py on every
# qualify_pair call and is already visible, unfiltered, in
# tests/test_kalman_research.py; it is not repeated here.
pytestmark = pytest.mark.filterwarnings("ignore:adfuller currently returns:FutureWarning")

_HL_NOTE = "half-life long relative to window"
_SIGN_NOTE = "hedge-ratio SIGN flipped between halves"
_UNREMARKABLE = "diagnostics unremarkable (NOT proof of edge)"


def _hand_pair(p1: np.ndarray, p2: np.ndarray):
    idx = pd.bdate_range("2020-01-01", periods=len(p1))
    frame = lambda px: pd.DataFrame({"Open": px, "Close": px}, index=idx)
    return align_pair(frame(p1), frame(p2), "A", "B", min_overlap=2)


def _random_walk_residual_pair(seed: int, n: int = 200):
    """p1 = 1.5 p2 + 100 + u with u a random walk WITH drift, so the residual's
    AR(1) coefficient is 1 in the data-generating process. (A driftless random
    walk's OLS phi is biased below 1 by roughly 5/n -- the Dickey-Fuller bias --
    so its sample half-life usually lands under n/4 and does not reliably earn
    the note; the drift pins phi at one in the sample as well.)"""
    rng = np.random.default_rng(seed)
    p2 = 50.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    u = np.cumsum(rng.normal(0.5, 1.0, n))
    return _hand_pair(1.5 * p2 + 100.0 + u, p2)


def _assert_notes_match_fields(d: PairDiagnostics) -> None:
    eg_ok = d.eg_pvalue <= 0.05
    hl_ok = math.isfinite(d.half_life_days) and d.half_life_days <= d.n_train / 4
    assert ("no cointegration evidence" in d.verdict) is (not eg_ok)
    assert (_HL_NOTE in d.verdict) is (not hl_ok)
    assert (_SIGN_NOTE in d.verdict) is (not d.beta_sign_stable)
    assert (_UNREMARKABLE in d.verdict) is (eg_ok and hl_ok and d.beta_sign_stable)


def test_overlap_ok_is_inclusive_at_exactly_min_overlap():
    pair, _ = synthetic_pair(n=300, seed=1)
    at = qualify_pair(pair, train_end=120, min_overlap=120)
    assert at.n_train == 120 and at.overlap_ok
    below = qualify_pair(pair, train_end=119, min_overlap=120)
    assert below.n_train == 119 and not below.overlap_ok
    # the window is [train_start, train_end): the start is honoured too
    shifted = qualify_pair(pair, train_end=220, min_overlap=120, train_start=100)
    assert shifted.n_train == 120 and shifted.overlap_ok


def test_n_train_counts_only_bars_where_both_legs_are_finite():
    pair, _ = synthetic_pair(n=300, seed=1)
    pair.p1.loc[pair.index[50], "Close"] = np.nan        # one gap inside the window
    d = qualify_pair(pair, train_end=120, min_overlap=120)
    assert d.n_train == 119
    assert not d.overlap_ok                                # the single gap flips the flag
    outside = qualify_pair(pair, train_end=50, min_overlap=50)   # bar 50 is excluded
    assert outside.n_train == 50 and outside.overlap_ok


def test_verdict_is_unremarkable_only_when_all_three_diagnostics_pass():
    pair, _ = synthetic_pair(n=400, seed=10)
    d = qualify_pair(pair, train_end=200, min_overlap=100)
    assert d.eg_pvalue <= 0.05
    assert math.isfinite(d.half_life_days) and d.half_life_days <= d.n_train / 4
    assert d.beta_sign_stable
    assert d.verdict == _UNREMARKABLE
    _assert_notes_match_fields(d)


def test_each_failing_diagnostic_adds_its_own_note():
    # Engle-Granger fails alone: the verdict is exactly that one note
    pair, _ = synthetic_pair(n=400, seed=1)
    d = qualify_pair(pair, train_end=200, min_overlap=100)
    assert d.eg_pvalue > 0.05 and d.beta_sign_stable
    assert d.verdict == f"Engle-Granger p={d.eg_pvalue:.2f}: no cointegration evidence at 5%"
    _assert_notes_match_fields(d)
    # the hedge ratio changes sign between the halves: the sign note is appended
    rng = np.random.default_rng(3)
    n = 200
    p2 = 50.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    p1 = np.where(np.arange(n) < n // 2, 2.0 * p2, 300.0 - 2.0 * p2) + rng.normal(0.0, 0.5, n)
    d = qualify_pair(_hand_pair(p1, p2), train_end=n, min_overlap=100, use_log=False)
    assert not d.beta_sign_stable
    assert _SIGN_NOTE in d.verdict and _UNREMARKABLE not in d.verdict
    _assert_notes_match_fields(d)


def test_random_walk_residual_has_a_long_or_infinite_half_life_and_the_note():
    # phi >= 1 in the sample: no finite half-life at all
    d_inf = qualify_pair(_random_walk_residual_pair(seed=5), train_end=200,
                         min_overlap=100, use_log=False)
    assert d_inf.resid_autocorr_1 >= 1.0
    assert not math.isfinite(d_inf.half_life_days)
    # phi just under 1: a finite half-life many times the window
    d_long = qualify_pair(_random_walk_residual_pair(seed=0), train_end=200,
                          min_overlap=100, use_log=False)
    assert 0.99 < d_long.resid_autocorr_1 < 1.0
    assert math.isfinite(d_long.half_life_days)
    assert d_long.half_life_days > 4 * d_long.n_train       # >> n/4
    for d in (d_inf, d_long):
        assert _HL_NOTE in d.verdict and _UNREMARKABLE not in d.verdict
        _assert_notes_match_fields(d)
