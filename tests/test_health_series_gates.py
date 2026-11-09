"""Pins for kalman.research.health_series: the lagged NIS window, the |z|
CUSUM with reset, and the relative beta-uncertainty gate.

health_series is public and described in docs/20-kalman-pairs.md but called
by nothing in the repo (the walk-forward uses health_from_signal), so until
now no test fixed its arithmetic. Records are built by hand with explicit
nis, z, x_post[0] (beta) and P_post[0, 0] (beta variance) so every trip bar
below is derivable from kalman/research.py:171-183 without running anything.

tests/test_health_gates_degenerate.py pins health_from_signal; this file pins
the record-based gate.
"""

from __future__ import annotations

import numpy as np

from kalman.core import StepRecord
from kalman.research import health_series


def _rec(t: int, nis: float | None, z: float | None, beta: float, beta_sd: float) -> StepRecord:
    return StepRecord(t=t, observed=True, x_prior=np.array([beta, 0.0]), P_prior=np.eye(2),
                      innovation=0.0, innovation_var=1.0, z=z, gain=None,
                      x_post=np.array([beta, 0.0]),
                      P_post=np.array([[beta_sd ** 2, 0.0], [0.0, 1.0]]),
                      loglik=0.0, nis=nis)


def _records(n: int, nis: float = 1.0, z: float = 0.0, beta: float = 1.0,
             beta_sd: float = 0.1) -> list[StepRecord]:
    return [_rec(t, nis, z, beta, beta_sd) for t in range(n)]


def _false_bars(healthy: np.ndarray) -> list[int]:
    return np.where(~healthy)[0].tolist()


# --- shape and the no-past bar ------------------------------------------------------


def test_neutral_records_are_healthy_everywhere():
    healthy = health_series(_records(60))
    assert healthy.dtype == bool and len(healthy) == 60 and healthy.all()


def test_bar_zero_is_always_healthy_because_it_has_no_past():
    extreme = [_rec(0, 1e9, 50.0, 1.0, 100.0)] + _records(5)[1:]
    assert health_series(extreme)[0]
    assert health_series(extreme, nis_window=2, nis_limit=0.0, cusum_h=0.0,
                         beta_unc_limit=0.0)[0]


# --- NIS window gate ------------------------------------------------------------------


def test_nis_gate_needs_half_a_window_of_finite_past_values_with_defaults():
    # nis_window=40 -> 20 finite past values needed; 3.0 > nis_limit 2.5 trips
    healthy = health_series(_records(60, nis=3.0))
    assert healthy[:20].all()
    assert (~healthy[20:]).all()


def test_nis_gate_never_trips_while_the_finite_count_is_below_half_the_window():
    recs = _records(12, nis=None)
    recs[4].nis = 100.0                          # the only finite value: 1 < 4 // 2
    assert health_series(recs, nis_window=4, nis_limit=2.5).all()


def test_nis_gate_mean_exactly_at_the_limit_does_not_trip_but_a_hair_above_does():
    recs = _records(12)
    for t in range(4, 8):
        recs[t].nis = 2.5                        # window [4, 8) has mean exactly 2.5 at t=8
    assert health_series(recs, nis_window=4, nis_limit=2.5).all()
    recs[7].nis = 2.5 + 1e-9
    assert _false_bars(health_series(recs, nis_window=4, nis_limit=2.5)) == [8]


def test_nis_gate_uses_only_bars_before_t():
    recs = _records(16)
    recs[8].nis = 1e9                            # bar 8's own NIS is not in bar 8's window
    healthy = health_series(recs, nis_window=4, nis_limit=2.5)
    assert healthy[8]
    assert _false_bars(healthy) == [9, 10, 11, 12]   # in the window for the next 4 bars


# --- lagged |z| CUSUM with reset ------------------------------------------------------


def test_cusum_on_constant_abs_z_of_four_alarms_every_seventh_bar_and_resets():
    # each bar adds |z| - 1 - k = 2.5: 6 * 2.5 = 15.0 is not > h = 15, 7 * 2.5 = 17.5 is,
    # so the first alarm is t=7; the counter resets and re-arms, hence 14, 21, 28
    healthy = health_series(_records(30, z=4.0), cusum_k=0.5, cusum_h=15.0)
    assert healthy[6]
    assert _false_bars(healthy) == [7, 14, 21, 28]


def test_cusum_uses_the_absolute_z_so_alternating_signs_trip_identically():
    recs = [_rec(t, 1.0, 4.0 if t % 2 == 0 else -4.0, 1.0, 0.1) for t in range(30)]
    assert _false_bars(health_series(recs)) == [7, 14, 21, 28]


def test_cusum_is_lagged_one_bar():
    recs = _records(12)
    recs[3].z = 100.0                            # 100 - 1.5 = 98.5 > 15 on the NEXT bar
    assert _false_bars(health_series(recs)) == [4]


# --- relative beta-uncertainty gate ---------------------------------------------------


def test_beta_sd_two_on_beta_one_trips_from_bar_one_onward():
    # P_post[0, 0] = 4.0 -> sd 2; ratio 2 > beta_unc_limit 1.0, read at t-1
    healthy = health_series(_records(10, beta=1.0, beta_sd=2.0))
    assert healthy.tolist() == [True] + [False] * 9


def test_beta_sd_one_point_five_on_beta_one_trips_from_bar_one_onward():
    healthy = health_series(_records(10, beta=1.0, beta_sd=1.5))
    assert healthy.tolist() == [True] + [False] * 9


def test_beta_ratio_exactly_one_does_not_trip():
    assert health_series(_records(10, beta=1.0, beta_sd=1.0)).all()
    assert health_series(_records(10, beta=-2.0, beta_sd=2.0)).all()   # |beta| is used


def test_beta_gate_is_skipped_when_beta_is_numerically_zero():
    assert health_series(_records(10, beta=0.0, beta_sd=1.5)).all()
    assert health_series(_records(10, beta=1e-9, beta_sd=1.5)).all()   # not > 1e-9
    assert not health_series(_records(10, beta=2e-9, beta_sd=1.5))[1:].any()


# --- causality across all three gates -------------------------------------------------


def test_mutating_bar_t_after_the_fact_cannot_change_healthy_up_to_t():
    recs = _records(60)
    baseline = health_series(recs)
    recs[25].nis = 1e9
    recs[25].z = 50.0
    recs[25].P_post[0, 0] = 1e4
    shocked = health_series(recs)
    np.testing.assert_array_equal(shocked[:26], baseline[:26])
    assert not shocked[26]                       # the shock is visible from the next bar
