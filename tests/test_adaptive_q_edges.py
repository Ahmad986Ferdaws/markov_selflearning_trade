"""Exact edges of kalman.core.AdaptiveQ: the window-length threshold, the
gamma = 0 neutral case, the clip equalities at m_min / m_max, equal bounds,
and the construction-time rejections.

tests/test_kalman_core.py::test_adaptive_q_lagged_and_bounded covers the
lagged-input contract and the saturated (well past the bound) cases; this file
pins the boundaries themselves.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from kalman.core import AdaptiveQ

# A two-point alternating series: its mean is exactly 0, so its ddof=1 std is
# the closed form a * sqrt(n / (n - 1)) and can be reasoned about by hand.
_A = 0.02
_SERIES = np.array([_A, -_A, _A, -_A])
_SIGMA = float(np.std(_SERIES, ddof=1))


def test_exactly_window_returns_are_enough_not_window_plus_one():
    ad = AdaptiveQ(sigma_ref=_SIGMA / 1.5, gamma=1.0, m_min=0.5, m_max=2.0, window=4)
    assert ad.multiplier(_SERIES[:3]) == 1.0                  # window - 1: neutral
    assert ad.multiplier(_SERIES) == pytest.approx(1.5)       # window: scaled
    # only the LAST `window` returns count: an older outlier is invisible
    assert ad.multiplier(np.concatenate([[5.0], _SERIES])) == ad.multiplier(_SERIES)


def test_gamma_zero_is_neutral_for_any_volatility():
    ad = AdaptiveQ(sigma_ref=0.01, gamma=0.0, m_min=0.25, m_max=4.0, window=4)
    for scale in (1e-9, 1e-3, 1.0, 1e3):
        assert ad.multiplier(_SERIES * (scale / _A)) == 1.0


def test_ratio_exactly_at_m_max_returns_m_max():
    assert _SIGMA == pytest.approx(_A * math.sqrt(4 / 3))      # the known std
    # sigma_ref = sigma / 2 is an exact halving, so sigma / sigma_ref == 2.0 exactly
    ad = AdaptiveQ(sigma_ref=_SIGMA / 2.0, gamma=1.0, m_min=0.5, m_max=2.0, window=4)
    assert _SIGMA / ad.sigma_ref == 2.0
    assert ad.multiplier(_SERIES) == 2.0


def test_ratio_exactly_at_m_min_returns_m_min():
    ad = AdaptiveQ(sigma_ref=_SIGMA / 0.5, gamma=1.0, m_min=0.5, m_max=2.0, window=4)
    assert _SIGMA / ad.sigma_ref == 0.5
    assert ad.multiplier(_SERIES) == 0.5


def test_equal_bounds_are_accepted_and_pin_the_multiplier():
    ad = AdaptiveQ(sigma_ref=_SIGMA, gamma=1.0, m_min=1.7, m_max=1.7, window=4)
    assert ad.multiplier(_SERIES) == 1.7                      # ratio 1.0: clipped up
    assert ad.multiplier(_SERIES * 100.0) == 1.7              # ratio 100: clipped down
    assert ad.multiplier(_SERIES[:3]) == 1.0                  # short history stays neutral


@pytest.mark.parametrize("kwargs, message", [
    ({"m_min": 2.0, "m_max": 1.0}, "m_min must not exceed m_max"),
    ({"sigma_ref": 0.0}, "sigma_ref must be positive"),
    ({"window": 1}, "window must be an integer >= 2"),
    ({"window": True}, "window must be an integer >= 2"),
    ({"window": 2.0}, "window must be an integer >= 2"),
])
def test_rejected_at_construction(kwargs, message):
    with pytest.raises(ValueError, match=message):
        AdaptiveQ(**{"sigma_ref": 0.01, **kwargs})


def test_numpy_int64_window_is_accepted_and_normalised_to_int():
    ad = AdaptiveQ(sigma_ref=_SIGMA / 2.0, window=np.int64(4))
    assert type(ad.window) is int and ad.window == 4
    assert ad.multiplier(_SERIES[:3]) == 1.0
    assert ad.multiplier(_SERIES) == 2.0
