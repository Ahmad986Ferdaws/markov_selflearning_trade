"""Pins for the signal-only health gate and the innovation diagnostics on
degenerate inputs: an all-NaN stream, a constant |z| stream (CUSUM trip and
re-arm), the rolling z^2 gate's strict threshold and its window, and the
diagnostics' minimum-sample contract.

tests/test_kalman_research.py::test_health_from_signal_causal_and_symmetric
pins causality under a one-bar shock on a random stream; the exact trip bars
and the re-arm are pinned here on streams whose arithmetic is known by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from kalman.core import StepRecord
from kalman.research import health_from_signal, innovation_diagnostics


def test_all_nan_signal_is_healthy_everywhere():
    healthy = health_from_signal(np.full(100, np.nan))
    assert healthy.dtype == bool and len(healthy) == 100 and healthy.all()


def test_constant_abs_z_of_four_trips_the_cusum_within_twenty_bars_and_re_arms():
    z = 4.0 * np.where(np.arange(100) % 2 == 0, 1.0, -1.0)     # |z| == 4, sign alternates
    healthy = health_from_signal(z)
    # the lagged CUSUM adds |z| - 1 - k = 2.5 per bar: 6 * 2.5 = 15 is not > h = 15,
    # 7 * 2.5 = 17.5 is, so the first alarm is bar 7, after which it resets
    alarms = np.flatnonzero(~healthy)
    assert alarms[0] == 7 < 20
    assert healthy[8:14].all()                       # re-armed: True bars follow the alarm
    assert not healthy[14]                           # ...until the next trip, 7 bars later
    # with the rolling z^2 gate switched off the alarms are exactly periodic
    cusum_only = health_from_signal(z, z2_limit=np.inf)
    np.testing.assert_array_equal(np.flatnonzero(~cusum_only), np.arange(7, 100, 7))


def test_rolling_z2_gate_trips_only_when_the_past_window_mean_strictly_exceeds_the_limit():
    z = np.zeros(120)
    z[60:] = 2.0                                     # z^2 = 4 from bar 60 on
    healthy = health_from_signal(z, cusum_h=np.inf)  # isolate the z^2 gate
    # window 40: at bar t the past window holds (t - 60) fours, mean 4 (t - 60) / 40;
    # that equals the 2.5 limit at t = 85 and first exceeds it at t = 86
    assert healthy[:86].all()
    assert not healthy[86:].any()


def test_rolling_z2_gate_uses_only_past_bars():
    z = np.zeros(120)
    z[50] = 100.0
    healthy = health_from_signal(z, cusum_h=np.inf)
    assert healthy[50]                               # bar 50's own z is not in its window
    assert healthy[:51].all()
    assert not healthy[51:91].any()                  # in the window for the next 40 bars
    assert healthy[91:].all()                        # ...then it falls out again


def _rec(t: int, z: float | None) -> StepRecord:
    return StepRecord(t=t, observed=z is not None, x_prior=np.zeros(2), P_prior=np.eye(2),
                      innovation=z, innovation_var=None if z is None else 1.0, z=z,
                      gain=None, x_post=np.zeros(2), P_post=np.eye(2), loglik=None,
                      nis=None if z is None else z * z)


def test_fewer_than_ten_scored_records_returns_only_the_count():
    scored = [_rec(t, 0.1 * t) for t in range(9)]
    unscored = [_rec(9 + i, None) for i in range(5)]      # predict-only bars do not count
    assert innovation_diagnostics(scored + unscored) == {"n": 9}
    assert innovation_diagnostics([]) == {"n": 0}
    full = innovation_diagnostics(scored + [_rec(14, 0.9)])  # the tenth unlocks the rest
    assert full["n"] == 10 and set(full) > {"n"}


def test_fifty_iid_records_report_coverage_and_the_moment_keys():
    zs = np.random.default_rng(0).normal(0.0, 1.0, 50)
    d = innovation_diagnostics([_rec(t, float(v)) for t, v in enumerate(zs)])
    assert set(d) == {"n", "z_mean", "z_std", "z_autocorr_1", "coverage_95", "nis_mean"}
    assert d["n"] == 50
    assert 0.8 <= d["coverage_95"] <= 1.0
    assert d["z_mean"] == pytest.approx(zs.mean())
    assert d["z_std"] == pytest.approx(zs.std(ddof=1))
    assert d["nis_mean"] == pytest.approx(np.mean(zs ** 2))   # nis = z^2 by construction
    assert -1.0 <= d["z_autocorr_1"] <= 1.0
