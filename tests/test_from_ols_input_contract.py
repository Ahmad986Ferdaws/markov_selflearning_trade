"""Input contract of kalman.core.PairFilter.from_ols.

The module docstring promises that a non-finite input raises; before this
contract existed, unusable training data reached the algebra and surfaced as
numpy's LinAlgError ('Singular matrix', 'Incompatible dimensions'), as
RuntimeWarnings followed by a misleading 'non-finite value reached the filter:
covariance', or as a PSD complaint from _symmetrize. Each case is now a
ValueError that names the training data, and a clean fit is unchanged.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from kalman import research
from kalman.core import PairFilter

_NOISE = (1e-5, 1e-6, 1e-3)   # q_beta, q_alpha, r


def _clean_pair(n: int = 100, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0.0, 0.01, n)) + 4.0
    y = 1.5 * x + 0.3
    return y, x


@pytest.mark.parametrize("level", [3.0, 4.0])
def test_constant_hedge_leg_is_a_value_error_not_linalgerror(level):
    with pytest.raises(ValueError, match="hedge leg is constant"):
        PairFilter.from_ols(np.arange(10.0), np.full(10, level), *_NOISE)


def test_constant_hedge_leg_propagates_through_train_loglik():
    y, _ = _clean_pair()
    with pytest.raises(ValueError, match="hedge leg is constant"):
        research.train_loglik(y, np.full(100, 4.0), *_NOISE)


def test_two_bars_rejected_before_any_algebra_and_without_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")   # a RuntimeWarning here would fail the test
        with pytest.raises(ValueError, match="need at least 3 training bars, got 2"):
            PairFilter.from_ols(np.array([1.0, 2.0]), np.array([1.0, 2.5]), *_NOISE)


def test_three_bars_are_enough():
    f = PairFilter.from_ols(np.array([1.0, 2.1, 2.9]), np.array([1.0, 2.0, 3.0]), *_NOISE)
    assert np.isfinite(f.x).all() and np.isfinite(f.P).all()


def test_nan_in_training_target_names_the_training_data():
    y, x = _clean_pair()
    y[50] = np.nan
    with pytest.raises(ValueError, match="training data contains non-finite"):
        PairFilter.from_ols(y, x, *_NOISE)


def test_inf_in_hedge_leg_names_the_training_data():
    y, x = _clean_pair()
    x[7] = np.inf
    with pytest.raises(ValueError, match="training data contains non-finite"):
        PairFilter.from_ols(y, x, *_NOISE)


def test_mismatched_lengths_rejected():
    y, x = _clean_pair()
    with pytest.raises(ValueError, match="training arrays must align"):
        PairFilter.from_ols(y, x[:-1], *_NOISE)


def test_two_dimensional_training_array_rejected():
    y, x = _clean_pair()
    with pytest.raises(ValueError, match="training arrays must align"):
        PairFilter.from_ols(y.reshape(10, 10), x, *_NOISE)


def test_clean_fit_matches_lstsq_and_keeps_the_data_scaled_prior():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(0.0, 0.01, 100)) + 4.0
    y = 2.0 * x + 1.0 + rng.normal(0.0, 0.01, 100)
    f = PairFilter.from_ols(y, x, *_NOISE)

    A = np.column_stack([x, np.ones(100)])
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    assert np.array_equal(f.x, coef)
    # P0 = 4 * sigma^2 (A'A)^-1, symmetrized: the same algebra as before the contract
    expected = 4.0 * float(res[0] / 98) * np.linalg.inv(A.T @ A)
    assert f.P == pytest.approx(0.5 * (expected + expected.T), rel=1e-12)
    assert np.array_equal(f.P, f.P.T)


def test_seeded_random_walk_pair_recovers_the_true_beta_with_a_psd_prior():
    y, x = _clean_pair()
    f = PairFilter.from_ols(y, x, *_NOISE)
    assert f.x[0] == pytest.approx(1.5)
    assert f.x[1] == pytest.approx(0.3)
    assert np.linalg.eigvalsh(f.P).min() >= 0.0


def test_integer_training_arrays_are_coerced_to_float():
    y = np.array([1, 3, 5, 7, 9])
    x = np.array([0, 1, 2, 3, 4])
    f = PairFilter.from_ols(y, x, *_NOISE)
    assert f.x == pytest.approx([2.0, 1.0])
