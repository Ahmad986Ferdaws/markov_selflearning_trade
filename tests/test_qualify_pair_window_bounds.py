"""Window bounds of kalman.pairs.qualify_pair.

The docstring promises diagnostics 'strictly on bars [train_start, train_end)'.
Before these checks a negative bound silently counted from the END of the
series (Python slicing: train_end=-1 on a 600-bar pair returned n_train=599,
lookahead in a training-only diagnostic), an over-long bound silently
truncated, an empty window died inside numpy ('zero-size array to reduction
operation maximum'), and three finite bars died inside statsmodels ('sample
size is too short'). Each is now a ValueError that names the offending values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalman.data import align_pair, synthetic_pair
from kalman.pairs import PairDiagnostics, qualify_pair

# statsmodels' adfuller deprecation is raised from kalman/pairs.py on every
# successful qualify_pair call and is already visible, unfiltered, in
# tests/test_kalman_research.py; it is not repeated here.
pytestmark = pytest.mark.filterwarnings("ignore:adfuller currently returns:FutureWarning")


def _hand_pair(p1: np.ndarray, p2: np.ndarray):
    idx = pd.bdate_range("2020-01-01", periods=len(p1))
    frame = lambda px: pd.DataFrame({"Open": px, "Close": px}, index=idx)
    return align_pair(frame(p1), frame(p2), "A", "B", min_overlap=2)


def _short_pair(nan_at: int | None = None):
    """Eight hand-picked bars that are far from collinear, so a 4-bar window
    runs through statsmodels without a CollinearityWarning."""
    p2 = np.array([50.0, 51.0, 49.5, 52.0, 51.2, 50.4, 53.0, 52.1])
    p1 = 1.5 * p2 + 100.0 + np.array([0.3, -0.4, 0.5, -0.2, 0.1, -0.6, 0.4, -0.1])
    if nan_at is not None:
        p1[nan_at] = np.nan
    return _hand_pair(p1, p2)


def test_empty_window_is_rejected_by_name():
    pair, _ = synthetic_pair(n=600)
    with pytest.raises(ValueError, match=r"empty training window \[100, 100\)"):
        qualify_pair(pair, train_end=100, train_start=100)


def test_start_after_end_is_rejected():
    pair, _ = synthetic_pair(n=600)
    with pytest.raises(ValueError, match=r"empty training window \[60, 50\)"):
        qualify_pair(pair, train_end=50, train_start=60)


def test_negative_train_end_no_longer_reads_from_the_end_of_the_series():
    pair, _ = synthetic_pair(n=600)
    with pytest.raises(ValueError, match="train_end must be a non-negative int, got -1"):
        qualify_pair(pair, train_end=-1)


def test_negative_train_start_is_rejected():
    pair, _ = synthetic_pair(n=600)
    with pytest.raises(ValueError, match="train_start must be a non-negative int, got -50"):
        qualify_pair(pair, train_end=100, train_start=-50)


def test_train_end_past_the_series_no_longer_truncates_silently():
    pair, _ = synthetic_pair(n=600)
    with pytest.raises(ValueError, match="train_end=601 exceeds the 600-bar pair"):
        qualify_pair(pair, train_end=len(pair) + 1)
    with pytest.raises(ValueError, match="train_end=10000 exceeds the 600-bar pair"):
        qualify_pair(pair, train_end=10_000)


@pytest.mark.parametrize("bad", [True, 100.0, np.float64(100.0), "100"])
def test_non_integer_bounds_are_rejected(bad):
    pair, _ = synthetic_pair(n=600)
    with pytest.raises(ValueError, match="train_end must be a non-negative int"):
        qualify_pair(pair, train_end=bad)


def test_train_end_exactly_len_is_accepted():
    pair, _ = synthetic_pair(n=600)
    d = qualify_pair(pair, train_end=len(pair), min_overlap=100)
    assert isinstance(d, PairDiagnostics) and d.n_train == 600


def test_numpy_integer_bounds_match_python_ints():
    pair, _ = synthetic_pair(n=600)
    native = qualify_pair(pair, train_end=300, train_start=50, min_overlap=100)
    grid = qualify_pair(pair, train_end=np.int64(300), train_start=np.int64(50), min_overlap=100)
    assert grid == native


def test_three_finite_bars_are_rejected_with_the_finite_count():
    pair = _short_pair(nan_at=1)                      # bars 0..3 leave 3 finite
    with pytest.raises(ValueError, match="only 3 finite training bars; need at least 4"):
        qualify_pair(pair, train_end=4, min_overlap=2)


def test_four_finite_bars_are_the_smallest_accepted_window():
    d = qualify_pair(_short_pair(), train_end=4, min_overlap=2)
    assert isinstance(d, PairDiagnostics) and d.n_train == 4
    # the NaN case passes too once the window reaches four finite bars
    assert qualify_pair(_short_pair(nan_at=1), train_end=5, min_overlap=2).n_train == 4
