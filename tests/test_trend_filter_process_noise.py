"""Pins for kalman.core.TrendFilter._Q, the dt-aware process-noise matrix.

tests/test_filter_input_contract.py::test_rejected_dt_does_not_advance_filter
pins that a bad dt leaves the step counter alone, and
tests/test_kalman_core.py::test_trend_filter_dt_scaling that a larger dt widens
the prior. This file pins the matrix algebra itself: the dt=1 form, symmetry
and positive definiteness, the dt^3 growth of the level term, linearity in the
adaptive multiplier, and that a rejected dt leaves the state and covariance the
next valid step builds its prior from untouched.
"""

from __future__ import annotations

import numpy as np
import pytest

from kalman.core import TrendFilter

QL, QV = 0.3, 0.6


def _filter() -> TrendFilter:
    return TrendFilter(q_level=QL, q_vel=QV, r=0.25)


def test_unit_dt_and_unit_mult_give_the_familiar_discrete_q():
    Q = _filter()._Q(1.0, 1.0)
    expected = np.array([[QL + QV / 3.0, QV / 2.0],
                         [QV / 2.0,      QV]])
    assert np.array_equal(Q, expected)
    assert Q[0, 0] == pytest.approx(0.5)
    assert Q[0, 1] == Q[1, 0] == 0.3
    assert Q[1, 1] == 0.6


@pytest.mark.parametrize("dt", [0.5, 1.0, 2.0, 7.0])
def test_q_is_symmetric_and_positive_definite(dt):
    Q = _filter()._Q(dt, 1.0)
    assert Q.shape == (2, 2)
    assert np.array_equal(Q, Q.T)
    assert np.linalg.eigvalsh(Q).min() > 0.0
    # det = ql*qv*dt^2 + qv^2*dt^4/12 > 0: strictly PD, not merely PSD
    assert np.linalg.det(Q) == pytest.approx(QL * QV * dt ** 2 + QV ** 2 * dt ** 4 / 12.0)


def test_level_variance_grows_as_ql_dt_plus_qv_dt_cubed_over_three():
    Q1, Q2 = _filter()._Q(1.0, 1.0), _filter()._Q(2.0, 1.0)
    assert Q2[0, 0] == pytest.approx(QL * 2.0 + QV * 8.0 / 3.0)      # 0.6 + 1.6
    assert Q2[0, 1] == pytest.approx(QV * 4.0 / 2.0)                  # qv*dt^2/2
    assert Q2[1, 1] == pytest.approx(QV * 2.0)                        # qv*dt
    # the cubic term makes the level variance grow faster than linearly in dt
    assert Q2[0, 0] > 2.0 * Q1[0, 0]
    assert Q2[1, 1] == pytest.approx(2.0 * Q1[1, 1])                  # velocity: linear


@pytest.mark.parametrize("dt", [0.5, 1.0, 2.0, 7.0])
@pytest.mark.parametrize("mult", [0.5, 3.0])
def test_mult_scales_the_whole_matrix_linearly(dt, mult):
    f = _filter()
    np.testing.assert_allclose(f._Q(dt, mult), mult * f._Q(dt, 1.0), rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("bad_dt", [0.0, -1.0])
def test_zero_or_negative_dt_is_rejected_before_anything_moves(bad_dt):
    f = _filter()
    x_before, P_before = f.x.copy(), f.P.copy()
    with pytest.raises(ValueError, match="dt must be positive"):
        f.step(100.0, dt=bad_dt)
    assert f._t == -1
    assert np.array_equal(f.x, x_before) and np.array_equal(f.P, P_before)
    # the next valid step builds its prior from the untouched state:
    # x_prior = F x,  P_prior = F P F' + Q(dt, 1)
    dt = 2.0
    rec = f.step(100.0, dt=dt)
    F = np.array([[1.0, dt], [0.0, 1.0]])
    assert rec.t == 0
    np.testing.assert_array_equal(rec.x_prior, F @ x_before)
    np.testing.assert_allclose(rec.P_prior, F @ P_before @ F.T + f._Q(dt, 1.0),
                               rtol=0.0, atol=1e-12)
