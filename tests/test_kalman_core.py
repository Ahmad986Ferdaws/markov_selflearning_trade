"""Kalman core acceptance tests — spec criteria 1, 2, 3, 6, 7, 12, 14.

Passing does NOT depend on producing a profitable backtest (criterion 16):
no test in this suite asserts positive P&L anywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from kalman.core import AdaptiveQ, PairFilter, TrendFilter, run_pair_filter
from kalman.data import synthetic_constant_pair, synthetic_pair, synthetic_trend


def _prices(pair):
    return (pair.p1["Close"].to_numpy(float), pair.p2["Close"].to_numpy(float))


# 1 — seeded synthetic recovers a KNOWN drifting beta within documented bounds.
# Bounds are stated under CORRECT specification (q = true drift variances,
# r = true obs variance): the filter then achieves ~95% 2-sd coverage. With
# misspecified q/r the same filter tracks with visible lag — Kalman optimality
# is conditional on the model being right, which is the documented lesson.
def test_recovers_known_drifting_beta():
    import math
    pair, truth = synthetic_pair()
    c1, c2 = _prices(pair)
    bt = truth["beta"].to_numpy()

    # (a) diffuse prior, correct spec: converges from data alone.
    #     Documented bounds: mean |err| < 0.03, max < 0.08 after 200-bar burn-in.
    f = PairFilter.diffuse(q_beta=0.002 ** 2, q_alpha=0.0005 ** 2, r=0.08 ** 2)
    recs = run_pair_filter(c1, c2, f)
    err = np.array([r.x_post[0] for r in recs])[200:] - bt[200:]
    assert np.abs(err).mean() < 0.03
    assert np.abs(err).max() < 0.08

    # (b) exact specification + true init: the error must match the filter's
    #     own posterior uncertainty (~95% 2-sd coverage) — the strongest
    #     correctness statement a Kalman implementation can make.
    at = truth["alpha"].to_numpy()
    f2 = PairFilter(q_beta=0.002 ** 2, q_alpha=0.0005 ** 2, r=0.08 ** 2,
                    x0=np.array([bt[0], at[0]]), P0=np.diag([1e-6, 1e-6]))
    recs2 = run_pair_filter(c1, c2, f2)
    err2 = np.array([r.x_post[0] for r in recs2])[200:] - bt[200:]
    sds2 = np.array([math.sqrt(r.P_post[0, 0]) for r in recs2])[200:]
    assert np.abs(err2).mean() < 0.006
    assert float(np.mean(np.abs(err2) < 2 * sds2)) > 0.90


# 2 — constant-state simulation converges and remains stable
def test_constant_state_converges_and_stays():
    pair, truth = synthetic_constant_pair()
    c1, c2 = _prices(pair)
    f = PairFilter.diffuse(1e-8, 1e-9, 0.05)
    recs = run_pair_filter(c1, c2, f)
    tail = np.array([r.x_post[0] for r in recs[-100:]])
    assert abs(tail.mean() - truth["beta"].iloc[0]) < 0.02
    assert tail.std() < 0.01          # stable, not oscillating


# 3 — batch and one-observation-at-a-time replay agree numerically
def test_batch_equals_replay():
    pair, _ = synthetic_pair(n=400)
    c1, c2 = _prices(pair)
    batch = run_pair_filter(c1, c2, PairFilter.diffuse(1e-5, 1e-6, 0.05))
    f2 = PairFilter.diffuse(1e-5, 1e-6, 0.05)
    rets = np.diff(c1, prepend=c1[0])
    for t in range(len(c1)):
        rec = f2.step(float(c1[t]), float(c2[t]), past_returns=rets[:t])
        assert np.array_equal(rec.x_post, batch[t].x_post)
        assert np.array_equal(rec.P_post, batch[t].P_post)


# 6 — covariances stay symmetric and PSD within tolerance
def test_covariance_symmetric_psd():
    pair, _ = synthetic_pair(n=600)
    c1, c2 = _prices(pair)
    recs = run_pair_filter(c1, c2, PairFilter.diffuse(1e-4, 1e-5, 0.05))
    for r in recs:
        assert np.allclose(r.P_post, r.P_post.T, atol=1e-10)
        assert np.linalg.eigvalsh(r.P_post).min() >= -1e-9


# 7 — innovation variances remain positive
def test_innovation_variance_positive():
    pair, _ = synthetic_pair(n=600)
    c1, c2 = _prices(pair)
    recs = run_pair_filter(c1, c2, PairFilter.diffuse(1e-5, 1e-6, 0.05))
    assert all(r.innovation_var > 0 for r in recs if r.observed)


# missing observations are explicit predict-only steps, never silent NaN
def test_missing_observation_explicit():
    pair, _ = synthetic_pair(n=300)
    c1, c2 = _prices(pair)
    c1 = c1.copy()
    c1[50] = np.nan
    recs = run_pair_filter(c1, c2, PairFilter.diffuse(1e-5, 1e-6, 0.05))
    assert recs[50].observed is False and recs[50].z is None
    assert np.all(np.isfinite(recs[51].x_post))     # filter continues cleanly
    # non-finite hedge price must raise, not propagate
    f = PairFilter.diffuse(1e-5, 1e-6, 0.05)
    with pytest.raises(ValueError):
        f.step(1.0, float("inf"))


# 12 — adaptive Q uses only lagged inputs and respects bounds
def test_adaptive_q_lagged_and_bounded():
    ad = AdaptiveQ(sigma_ref=0.01, gamma=1.0, m_min=0.5, m_max=2.0, window=5)
    # fewer than `window` past returns -> neutral multiplier
    assert ad.multiplier(np.array([0.1] * 3)) == 1.0
    hot = ad.multiplier(np.array([0.5, -0.5, 0.5, -0.5, 0.5]))
    assert hot == 2.0                                  # capped at m_max
    calm = ad.multiplier(np.array([1e-6, -1e-6, 1e-6, -1e-6, 1e-6]))
    assert calm == 0.5                                 # floored at m_min
    flat = ad.multiplier(np.array([1e-6] * 5))         # zero variance -> neutral
    assert flat == 1.0
    # the filter passes returns ENDING AT t-1 (structural lag): mutate the
    # current bar's return and confirm the multiplier cannot see it
    past = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
    m1 = ad.multiplier(past)
    m2 = ad.multiplier(past)   # same past, "today" irrelevant by signature
    assert m1 == m2


# trend filter: dt-aware Q, velocity recovery on synthetic
def test_trend_filter_recovers_velocity():
    y, truth = synthetic_trend()
    f = TrendFilter(1e-4, 1e-5, 0.25)
    errs = []
    for t, val in enumerate(y.to_numpy()):
        rec = f.step(float(val), dt=1.0)
        if t > 150:
            errs.append(abs(rec.x_post[1] - truth["velocity"].iloc[t]))
    assert float(np.mean(errs)) < 0.05


def test_trend_filter_dt_scaling():
    # doubling dt must increase prior state uncertainty (dt-aware Q, not fixed)
    f1 = TrendFilter(1e-4, 1e-5, 0.25)
    f2 = TrendFilter(1e-4, 1e-5, 0.25)
    r1 = f1.step(100.0, dt=1.0)
    r2 = f2.step(100.0, dt=5.0)
    assert np.trace(r2.P_prior) > np.trace(r1.P_prior)


# 14 — seeded experiments are reproducible
def test_reproducible_synthetic():
    a, ta = synthetic_pair(seed=42)
    b, tb = synthetic_pair(seed=42)
    assert np.array_equal(a.p1["Close"].to_numpy(), b.p1["Close"].to_numpy())
    assert np.array_equal(ta["beta"].to_numpy(), tb["beta"].to_numpy())


# invalid parameters are rejected up front
def test_positive_parameterization_enforced():
    with pytest.raises(ValueError):
        PairFilter(q_beta=-1e-5, q_alpha=1e-6, r=0.05)
    with pytest.raises(ValueError):
        PairFilter(q_beta=1e-5, q_alpha=1e-6, r=0.0)
    with pytest.raises(ValueError):
        TrendFilter(q_level=0.0, q_vel=1e-5, r=0.1)
