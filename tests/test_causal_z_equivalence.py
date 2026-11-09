"""`_causal_z` is vectorised with pandas rolling statistics; this pins it to the
per-bar loop it replaced, NaN for NaN, to 1e-10.

`_reference` is the original loop kept verbatim, so the pandas version is
measured against the semantics the walk-forward was published with: trailing
window ending at t-1, NaN until `window` bars of history exist, NaN when the
window holds a missing value, NaN on a zero spread, NaN when resid[t] itself
is not finite.
"""

from __future__ import annotations

import numpy as np
import pytest

from kalman.research import _causal_z

WINDOWS = (5, 20, 60)


def _reference(resid: np.ndarray, window: int = 60) -> np.ndarray:
    z = np.full(len(resid), np.nan)
    for t in range(window, len(resid)):
        past = resid[t - window:t]                  # ends at t-1: lagged
        m, s = float(np.mean(past)), float(np.std(past, ddof=1))
        if np.isfinite(resid[t]) and s > 0:
            z[t] = (resid[t] - m) / s
    return z


def _series(kind: str, n: int = 1500) -> np.ndarray:
    seed = {"gaussian": 0, "interior_nans": 1, "constant_stretch": 2}[kind]
    r = np.random.default_rng(seed).normal(0.0, 0.1, n)
    if kind == "interior_nans":
        r[[40, 41, 300, 777, 1200]] = np.nan
    elif kind == "constant_stretch":
        r[400:520] = 0.25                           # 120 identical bars -> sd == 0 inside
    return r


@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("kind", ["gaussian", "interior_nans", "constant_stretch"])
def test_vectorised_causal_z_matches_the_loop(kind, window):
    r = _series(kind)
    np.testing.assert_allclose(_causal_z(r, window), _reference(r, window),
                               rtol=0, atol=1e-10, equal_nan=True)


def test_first_finite_bar_and_nan_propagation_span():
    window = 20
    z = _causal_z(_series("interior_nans"), window)
    # exactly `window` bars of lagged history are needed: z[window] is the first finite value
    assert np.isnan(z[:window]).all() and np.isfinite(z[window])
    # a NaN at k blanks bar k itself and the `window` later bars whose trailing window
    # contains it (pandas counts the NaN as missing, min_periods is not met) -- no more
    k = 300
    assert np.isfinite(z[k - 1])
    assert np.isnan(z[k:k + window + 1]).all()
    assert np.isfinite(z[k + window + 1])


def test_constant_stretch_gives_nan_never_inf():
    window = 20
    r = _series("constant_stretch")
    z = _causal_z(r, window)
    assert np.isnan(z[420:520]).all()               # window fully inside the stretch: sd == 0
    assert np.isfinite(z[419])                      # one random bar left in the window
    assert not np.isinf(z).any()
    assert np.array_equal(np.isnan(z), np.isnan(_reference(r, window)))


def test_non_finite_residual_is_nan_at_its_own_bar_and_in_the_windows_after():
    window = 5
    r = _series("gaussian")
    r[100] = np.inf
    with np.errstate(invalid="ignore"):             # the loop's np.std warns on inf - inf
        ref = _reference(r, window)
    z = _causal_z(r, window)
    assert np.isnan(z[100:100 + window + 1]).all() and np.isfinite(z[100 + window + 1])
    np.testing.assert_allclose(z, ref, rtol=0, atol=1e-10, equal_nan=True)
