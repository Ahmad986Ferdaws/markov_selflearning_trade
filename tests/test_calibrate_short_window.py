"""Pins for the training-window floor of the calibration likelihood.

`train_loglik` fits its OLS prior on the first `init` bars (default
max(60, n // 5)) and scores only the bars after them; it returns -inf when
fewer than 10 bars would be scored. `calibrate` now checks that floor up
front and says the window is too short, instead of gridding sixteen -inf
cells and reporting 'calibration failed: all likelihoods -inf'. The
RuntimeError fallback stays for a window that is long enough but whose
likelihoods are all non-finite anyway.

tests/test_kalman_research.py::test_train_loglik_scores_only_post_init_region
pins the predictive (init never scored) property on 400 bars; the exact
boundary from both sides is pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from kalman.research import calibrate, train_loglik

RS = (1e-5, 1e-4, 1e-3, 1e-2)


def _pair(n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(0, 0.01, n)) + 4.0
    y = 1.5 * x + 0.3 + rng.normal(0, 0.01, n)
    return y, x


def test_train_loglik_default_init_refuses_seventy_bars_and_scores_seventy_one():
    y, x = _pair(71)
    # default init = max(60, n // 5) = 60; refused iff 60 >= n - 10, i.e. n <= 70
    assert train_loglik(y[:70], x[:70], 1e-6, 1e-7, 1e-3) == -np.inf
    assert np.isfinite(train_loglik(y[:71], x[:71], 1e-6, 1e-7, 1e-3))


def test_train_loglik_init_boundary_is_init_at_least_n_minus_ten():
    y, x = _pair(100)
    # the `i0 >= n - 10` edge from both sides: init=90 leaves exactly 10 scored bars
    # and is refused; init=89 leaves 11 and is scored
    assert train_loglik(y, x, 1e-6, 1e-7, 1e-3, init=90) == -np.inf
    assert np.isfinite(train_loglik(y, x, 1e-6, 1e-7, 1e-3, init=89))


def test_calibrate_says_seventy_bars_is_too_short():
    y, x = _pair(70)
    with pytest.raises(ValueError, match="too short") as exc:
        calibrate(y, x)
    msg = str(exc.value)
    assert "70 bars" in msg and "more than 70 bars" in msg and "60 init + 10 scored" in msg


def test_calibrate_rejects_misaligned_inputs_before_gridding():
    y, x = _pair(80)
    with pytest.raises(ValueError, match="must align"):
        calibrate(y, x[:79])


def test_calibrate_on_seventy_one_bars_returns_the_full_surface():
    y, x = _pair(71)
    qb, qa, r, surface = calibrate(y, x)
    assert qb > 0
    assert qa == pytest.approx(qb * 0.1)
    assert r in RS
    assert len(surface) == 16
    assert np.isfinite(surface["loglik"]).any()
