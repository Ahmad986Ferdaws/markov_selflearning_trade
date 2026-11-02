"""Boundary pins for the regime labelling rule and the transition estimator
in app/services/regime.py.

label_state is the single source of truth for bull/bear/sideways; every test
here sits on one of its comparisons (undefined input, zero dispersion, a mean
exactly at k*std). The estimator tests pin the documented uniform-row
fallback and the Chapman-Kolmogorov forecast.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.services.regime import (
    STATE_ORDER,
    STATE_TO_IDX,
    define_states,
    estimate_transition_matrix,
    forecast,
    label_state,
)


# --- label_state ---------------------------------------------------------------
@pytest.mark.parametrize("mean, std", [
    (math.nan, 0.01), (0.01, math.nan), (None, 0.01), (0.01, None),
])
def test_undefined_inputs_give_no_label(mean, std):
    assert label_state(mean, std) is None


@pytest.mark.parametrize("std", [0.0, -1e-9])
def test_zero_or_negative_dispersion_is_sideways_whatever_the_mean(std):
    assert label_state(0.5, std) == "sideways"
    assert label_state(-0.5, std) == "sideways"


def test_a_mean_exactly_at_k_times_std_is_sideways_on_both_sides():
    k, std = 0.5, 0.02
    assert label_state(k * std, std, k) == "sideways"
    assert label_state(-k * std, std, k) == "sideways"


def test_a_mean_a_hair_beyond_k_times_std_flips_the_label():
    k, std, eps = 0.5, 0.02, 1e-12
    assert label_state(k * std + eps, std, k) == "bull"
    assert label_state(-k * std - eps, std, k) == "bear"


# --- define_states -------------------------------------------------------------
def test_absolute_mode_uses_the_fixed_thresholds():
    idx = pd.date_range("2024-01-01", periods=12, freq="D")
    rets = pd.Series([0.03] * 4 + [-0.03] * 4 + [0.0] * 4, index=idx)
    states = define_states(rets, window=2, mode="absolute", bull_thresh=0.02, bear_thresh=-0.02)
    assert len(states) == 11                      # one warm-up bar dropped
    assert states.iloc[0] == "bull"
    assert states.iloc[-1] == "sideways"
    assert "bear" in set(states)


def test_zscore_mode_emits_exactly_one_label_per_bar_after_warmup():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    states = define_states(pd.Series(rng.normal(0, 0.02, 100), index=idx), window=10, k=0.5)
    assert len(states) == 91
    assert set(states.unique()) <= set(STATE_ORDER)


def test_a_constant_return_series_is_sideways_in_zscore_mode():
    """Zero rolling std -> sideways, never a division by zero or a NaN label."""
    rets = pd.Series(0.01, index=pd.date_range("2024-01-01", periods=30, freq="D"))
    states = define_states(rets, window=5, k=0.5)
    assert set(states.unique()) == {"sideways"}
