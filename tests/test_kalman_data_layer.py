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


# --- align_pair validators ------------------------------------------------------------
def test_align_pair_rejects_a_non_positive_price():
    bad, good = _leg([1.0, 2.0, 0.0, 3.0]), _leg([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="non-positive"):
        align_pair(bad, good, "A", "B", min_overlap=2)


def test_align_pair_rejects_an_unsorted_index():
    good = _leg([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="not sorted"):
        align_pair(good.iloc[::-1], good, "A", "B", min_overlap=2)


def test_align_pair_rejects_a_missing_required_column():
    good = _leg([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="missing columns"):
        align_pair(good.drop(columns=["Open"]), good, "A", "B", min_overlap=2)


def test_align_pair_inner_joins_the_calendar_and_enforces_the_minimum_overlap():
    a = _leg(np.arange(1, 11, dtype=float), start="2020-01-01")   # ten business days
    b = _leg(np.arange(1, 11, dtype=float), start="2020-01-08")   # five of them shared
    pair = align_pair(a, b, "A", "B", min_overlap=5)
    assert len(pair) == 5
    assert pair.index.equals(a.index.intersection(b.index))
    with pytest.raises(ValueError, match="overlap too short"):
        align_pair(a, b, "A", "B", min_overlap=6)
