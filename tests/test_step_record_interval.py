"""Pins for kalman.core.StepRecord.state_interval: centre, width, the
zero-variance case, and the clamp that keeps a floating-point-negative
diagonal from reaching math.sqrt. A real PairFilter record is checked last so
the hand-built records are shown to match what the filter actually emits.
"""

from __future__ import annotations

import numpy as np
import pytest

from kalman.core import PairFilter, StepRecord


def _record(P_post: np.ndarray) -> StepRecord:
    x_post = np.array([1.5, -0.5])
    return StepRecord(t=0, observed=True, x_prior=x_post.copy(), P_prior=P_post.copy(),
                      innovation=0.0, innovation_var=1.0, z=0.0, gain=np.zeros(2),
                      x_post=x_post, P_post=P_post, loglik=None, nis=None)


def test_default_interval_is_centre_plus_minus_two_sd():
    rec = _record(np.diag([4.0, 0.0]))
    lo, hi = rec.state_interval(0)                   # sd = 2, n_sd defaults to 2
    assert (lo, hi) == (1.5 - 4.0, 1.5 + 4.0)
    assert isinstance(lo, float) and isinstance(hi, float)


def test_one_sd_interval_has_width_two_sd_and_is_centred_on_x_post():
    rec = _record(np.diag([4.0, 0.0]))
    lo, hi = rec.state_interval(0, n_sd=1)
    assert (lo, hi) == (-0.5, 3.5)
    assert hi - lo == 4.0
    assert (lo + hi) / 2.0 == 1.5


def test_zero_variance_collapses_to_the_point_estimate():
    rec = _record(np.diag([4.0, 0.0]))
    assert rec.state_interval(1) == (-0.5, -0.5)
    assert rec.state_interval(1, n_sd=10) == (-0.5, -0.5)


def test_tiny_negative_variance_is_clamped_to_zero_width_not_raised():
    P = np.diag([4.0, 0.0])
    P[1, 1] = -1e-15                                 # floating-point drift below zero
    rec = _record(P)
    assert rec.state_interval(1) == (-0.5, -0.5)     # math.sqrt(-1e-15) would raise
    assert rec.state_interval(0) == (-2.5, 5.5)      # the other diagonal is untouched


def test_real_filter_record_has_symmetric_p_post_whose_interval_contains_x_post():
    f = PairFilter.diffuse(q_beta=1e-5, q_alpha=1e-6, r=0.05)
    for t in range(5):
        rec = f.step(100.0 + t, 50.0 + 0.5 * t)
    assert rec.observed
    assert np.array_equal(rec.P_post, rec.P_post.T)
    assert (np.diag(rec.P_post) > 0.0).all()
    for i in (0, 1):
        lo, hi = rec.state_interval(i)
        assert lo <= rec.x_post[i] <= hi
        assert hi - lo == pytest.approx(4.0 * np.sqrt(rec.P_post[i, i]))
