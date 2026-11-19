"""A non-finite lagged return makes AdaptiveQ neutral for a whole window.

AdaptiveQ.multiplier returns 1.0 whenever the trailing ddof=1 std is
non-finite, and run_pair_filter builds past_returns with
np.diff(y, prepend=y[0]), so ONE missing observation poisons TWO differences
(bars g and g+1) and the adaptive scaling is silently neutral until the
window has slid past them. This file pins that behaviour; changing it would
move kalman_adaptive numbers on any gappy pair and is a methodology decision
for a separate issue.

tests/test_adaptive_q_edges.py pins short history and zero variance; this
file pins the non-finite path.
"""

from __future__ import annotations

import numpy as np
import pytest

from kalman.core import AdaptiveQ

_WINDOW = 5


def test_nan_in_the_window_returns_neutral():
    ad = AdaptiveQ(sigma_ref=0.01, window=_WINDOW)
    assert ad.multiplier(np.array([0.01, np.nan, 0.01, 0.01, 0.01])) == 1.0


def test_inf_in_the_window_returns_neutral():
    ad = AdaptiveQ(sigma_ref=0.01, window=_WINDOW)
    # numpy warns about inf - inf inside np.std; the multiplier is still neutral
    with pytest.warns(RuntimeWarning, match="invalid value"):
        assert ad.multiplier(np.array([0.01, np.inf, 0.01, 0.01, 0.01])) == 1.0


def test_clean_window_of_the_same_scale_is_not_neutral():
    ad = AdaptiveQ(sigma_ref=0.01, window=_WINDOW)
    assert ad.multiplier(np.array([0.01, -0.01, 0.01, -0.01, 0.01])) != 1.0


def _gappy_returns(gap: int = 10, n: int = 30) -> np.ndarray:
    """Mirror run_pair_filter: rets = np.diff(y, prepend=y[0]) with y[gap] NaN."""
    rng = np.random.default_rng(3)
    y = np.cumsum(rng.normal(0.0, 0.01, n))
    y[gap] = np.nan
    return np.diff(y, prepend=y[0])


def test_one_missing_observation_poisons_two_differences():
    rets = _gappy_returns(gap=10)
    assert np.where(np.isnan(rets))[0].tolist() == [10, 11]


def test_adaptive_q_is_neutral_for_window_bars_after_the_gap():
    rets = _gappy_returns(gap=10)
    # sigma_ref far below the realized scale: every CLEAN window clips to m_max,
    # so a 1.0 can only come from the non-finite path (or from short history)
    ad = AdaptiveQ(sigma_ref=1e-3, gamma=1.0, m_max=4.0, window=_WINDOW)
    mult = {t: ad.multiplier(rets[:t]) for t in range(_WINDOW, len(rets) + 1)}

    for t in range(12, 17):                 # windows rets[t-5:t] contain bar 11
        assert mult[t] == 1.0, t
    assert mult[17] != 1.0                  # rets[12:17] is the first clean window
    assert mult[17] == 4.0
    # the exact neutral set: bar 10's window first, then `window` bars after bar 11
    assert sorted(t for t, m in mult.items() if m == 1.0) == [11, 12, 13, 14, 15, 16]
    assert all(m == 4.0 for t, m in mult.items() if t not in range(11, 17))


def test_neutral_span_scales_with_the_window():
    rets = _gappy_returns(gap=10)
    ad = AdaptiveQ(sigma_ref=1e-3, gamma=1.0, m_max=4.0, window=8)
    neutral = [t for t in range(8, len(rets) + 1) if ad.multiplier(rets[:t]) == 1.0]
    assert neutral == list(range(11, 11 + 8 + 1))      # bars 11..19: gap window + 8
