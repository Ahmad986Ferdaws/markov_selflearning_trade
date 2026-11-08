"""Exact edges of _validate_controls in app/services/evaluation.py.

tests/test_daily_input_contract.py shows, through evaluate(), that a bad
control is refused at all. These tests call the validator directly and sit on
each comparison: the value that is refused and the neighbouring value that
passes, with the refusal message pinned.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services.evaluation import _validate_controls

DEFAULTS = dict(train_frac=0.7, grid_windows=(10,), grid_k=(0.5,),
                fee_pct=0.3, slippage_pct=0.5, min_train=60)


def check(**overrides):
    return _validate_controls(**{**DEFAULTS, **overrides})


# --- combined one-way cost: strictly below 100 percent -------------------------
def test_fee_and_slippage_summing_to_exactly_100_percent_are_refused():
    with pytest.raises(ValueError, match="combined one-way cost must be below 100 percent"):
        check(fee_pct=50, slippage_pct=50)


def test_fee_and_slippage_summing_to_just_under_100_percent_pass():
    assert check(fee_pct=49.999, slippage_pct=50) is None


def test_numpy_scalar_costs_pass():
    assert check(fee_pct=np.float32(0.3), slippage_pct=np.float32(0.5)) is None


# --- train_frac: the open interval (0, 1) ---------------------------------------
@pytest.mark.parametrize("train_frac", [0.0, 1.0])
def test_train_frac_at_either_end_of_the_interval_is_refused(train_frac):
    with pytest.raises(ValueError, match=r"train_frac must be finite and inside \(0, 1\)"):
        check(train_frac=train_frac)


@pytest.mark.parametrize("train_frac", [1e-9, 1 - 1e-9])
def test_train_frac_a_hair_inside_the_interval_passes(train_frac):
    assert check(train_frac=train_frac) is None


# --- min_train: an integer >= 2, bool excluded -----------------------------------
def test_min_train_of_two_passes():
    assert check(min_train=2) is None


@pytest.mark.parametrize("min_train", [1, True])
def test_min_train_of_one_or_a_bool_is_refused(min_train):
    with pytest.raises(ValueError, match="min_train must be an integer >= 2"):
        check(min_train=min_train)


# --- grid_windows: non-empty integers >= 2, bool excluded -------------------------
def test_grid_window_of_two_passes():
    assert check(grid_windows=(2,)) is None


@pytest.mark.parametrize("grid_windows", [(1,), (True,), ()])
def test_grid_window_of_one_a_bool_or_an_empty_grid_is_refused(grid_windows):
    with pytest.raises(ValueError, match="grid_windows must contain integers >= 2"):
        check(grid_windows=grid_windows)


# --- grid_k: finite and nonnegative, with -0.0 counting as zero -------------------
@pytest.mark.parametrize("grid_k", [(0.0,), (-0.0,)])
def test_grid_k_of_zero_passes_with_either_sign(grid_k):
    assert check(grid_k=grid_k) is None


@pytest.mark.parametrize("grid_k", [(-1e-9,), (float("nan"),)])
def test_grid_k_a_hair_negative_or_nan_is_refused(grid_k):
    with pytest.raises(ValueError, match="grid_k must contain finite nonnegative values"):
        check(grid_k=grid_k)
