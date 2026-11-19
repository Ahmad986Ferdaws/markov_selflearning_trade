"""innovation_diagnostics on a constant innovation stream.

The lag-1 autocorrelation is undefined when either lagged slice has zero
variance; numpy's corrcoef returned NaN for it but also emitted
'RuntimeWarning: invalid value encountered in divide'. The guard returns the
same NaN without the warning and leaves corrcoef untouched for every stream
with variance, so walk_forward's innovation keys are byte-identical.

tests/test_health_gates_degenerate.py pins the minimum-sample contract and the
moment keys on fifty iid records; the constant-z case and the exact corrcoef
equality are pinned here.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from kalman.core import StepRecord
from kalman.research import innovation_diagnostics


def _rec(t: int, z: float | None) -> StepRecord:
    return StepRecord(t=t, observed=z is not None, x_prior=np.zeros(2), P_prior=np.eye(2),
                      innovation=z, innovation_var=None if z is None else 1.0, z=z,
                      gain=None, x_post=np.zeros(2), P_post=np.eye(2), loglik=None,
                      nis=None if z is None else z * z)


def test_constant_z_reports_nan_autocorrelation_without_a_runtime_warning():
    recs = [_rec(t, 1.0) for t in range(20)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")                 # any RuntimeWarning fails the test
        d = innovation_diagnostics(recs)
    assert d["n"] == 20
    assert d["z_std"] == 0.0
    assert math.isnan(d["z_autocorr_1"])
    assert d["z_mean"] == 1.0 and d["coverage_95"] == 1.0 and d["nis_mean"] == 1.0


def test_a_stream_constant_only_in_its_lagged_slice_is_also_guarded():
    # z[:-1] is constant while z[1:] is not: corrcoef would divide by a zero std
    zs = [1.0] * 19 + [2.0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d = innovation_diagnostics([_rec(t, v) for t, v in enumerate(zs)])
    assert math.isnan(d["z_autocorr_1"])
    assert d["z_std"] > 0.0


def test_iid_records_keep_numpys_exact_lag_one_correlation():
    zs = np.random.default_rng(0).normal(0.0, 1.0, 50)
    d = innovation_diagnostics([_rec(t, float(v)) for t, v in enumerate(zs)])
    expected = float(np.corrcoef(zs[:-1], zs[1:])[0, 1])
    assert d["z_autocorr_1"] == expected                # bit-identical, not approx
    assert -1.0 <= d["z_autocorr_1"] <= 1.0
