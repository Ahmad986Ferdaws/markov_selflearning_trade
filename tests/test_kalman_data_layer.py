"""Pins for kalman/data.py: the fail-loud validators and the seeded synthetic
generators that the filter tests measure recovery against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalman.data import (
    align_pair,
    load_pair_csv,
    synthetic_constant_pair,
    synthetic_pair,
    synthetic_trend,
)


def _leg(closes, start: str = "2020-01-01") -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    idx = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"Open": closes, "Close": closes}, index=idx)


# --- synthetic generators ----------------------------------------------------------
def test_synthetic_pair_is_reproducible_for_a_seed_and_differs_across_seeds():
    a, truth_a = synthetic_pair(n=300, seed=7)
    b, truth_b = synthetic_pair(n=300, seed=7)
    c, _ = synthetic_pair(n=300, seed=8)
    pd.testing.assert_frame_equal(a.p1, b.p1)
    pd.testing.assert_frame_equal(a.p2, b.p2)
    pd.testing.assert_frame_equal(truth_a, truth_b)
    assert not a.p1["Close"].equals(c.p1["Close"])


def test_synthetic_pair_truth_is_aligned_with_the_prices():
    pair, truth = synthetic_pair(n=300)
    assert len(pair) == 300
    assert truth.index.equals(pair.index)
    assert list(truth.columns) == ["beta", "alpha"]
    assert (pair.p1["Close"] > 0).all() and (pair.p2["Close"] > 0).all()


def test_constant_pair_has_a_flat_beta_path():
    _, truth = synthetic_constant_pair(n=300, beta=2.0)
    assert truth["beta"].iloc[0] == pytest.approx(2.0)
    assert truth["beta"].max() - truth["beta"].min() < 1e-6


def test_synthetic_trend_is_seeded_and_returns_aligned_level_and_velocity():
    y, truth = synthetic_trend(n=200, seed=5)
    y_again, _ = synthetic_trend(n=200, seed=5)
    pd.testing.assert_series_equal(y, y_again)
    assert truth.index.equals(y.index)
    assert list(truth.columns) == ["level", "velocity"]
