"""The three history-size refusals of evaluate() in app/services/evaluation.py,
pinned at their exact edges.

evaluate() works on n = rows - 1 daily returns and refuses, in order:
fewer than min_train + 30 returns; a split (int(n * train_frac)) that leaves
no walked train region (split <= min_train + 1); a split that leaves no test
region (split >= n - 1). Each test sits one row on each side of a threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.evaluation import evaluate

GRID = dict(grid_windows=(10,), grid_k=(0.5,))      # one config: fast, deterministic


def _history(rows: int, seed: int = 0) -> pd.DataFrame:
    """A seeded random-walk Close frame with `rows` bars (rows - 1 returns)."""
    rng = np.random.default_rng(seed)
    closes = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, rows))
    return pd.DataFrame({"Close": closes},
                        index=pd.date_range("2020-01-01", periods=rows, freq="D"))


# --- refusal 1: fewer than min_train + 30 returns -------------------------------
@pytest.mark.parametrize("min_train", [60, 20])
def test_one_return_short_of_min_train_plus_30_is_not_enough_history(min_train):
    n = min_train + 29
    with pytest.raises(ValueError, match=f"Not enough history: {n} returns"):
        evaluate(_history(n + 1), min_train=min_train, **GRID)


@pytest.mark.parametrize("min_train", [60, 20])
def test_exactly_min_train_plus_30_returns_is_accepted(min_train):
    report = evaluate(_history(min_train + 31), min_train=min_train, **GRID)
    assert report.test_size > 0


# --- refusal 2: split <= min_train + 1 leaves no walked train region ------------
def test_a_split_at_min_train_plus_one_is_a_too_small_train_region():
    # 200 returns, min_train 60: split = int(200 * 0.3075) = 61 <= 61
    with pytest.raises(ValueError, match="Train region too small: split=61 <= min_train=60"):
        evaluate(_history(201), train_frac=0.3075, min_train=60, **GRID)


def test_a_split_one_past_min_train_plus_one_walks_a_two_day_train_region():
    # split = int(200 * 0.3125) = 62: the first accepted split
    report = evaluate(_history(201), train_frac=0.3125, min_train=60, **GRID)
    assert report.train_size == 2


# --- refusal 3: split >= n - 1 leaves no test region ----------------------------
def test_a_split_at_n_minus_one_is_an_empty_test_region():
    # a 200-row frame has 199 returns: split = int(199 * 0.999) = 198 >= 198
    with pytest.raises(ValueError, match="Test region empty: split=198 >= n-1=198"):
        evaluate(_history(200), train_frac=0.999, **GRID)


def test_a_split_at_n_minus_two_leaves_exactly_one_test_day():
    # split = int(199 * (197.5 / 199)) = 197: the last accepted split
    report = evaluate(_history(200), train_frac=197.5 / 199, **GRID)
    assert report.test_size == 1
