"""The sparse-transition warning in app/services/regime.py, pinned at exactly
min_count.

sparse_cell_warnings counts from->to pairs between consecutive labels and
warns for every cell with 0 < count < min_count. A run of identical labels
fills exactly one cell (state->state) and nothing else, so each test below
knows precisely which cell holds how many counts.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.services.regime import STATE_ORDER, sparse_cell_warnings


def _self_loop(state: str, transitions: int) -> pd.Series:
    """`transitions + 1` identical labels: `transitions` counts in the
    state->state cell, zero in every other cell."""
    return pd.Series([state] * (transitions + 1))


@pytest.mark.parametrize("state", STATE_ORDER)
def test_a_cell_at_exactly_min_count_does_not_warn(state):
    assert sparse_cell_warnings(_self_loop(state, 5), min_count=5) == []


@pytest.mark.parametrize("state", STATE_ORDER)
def test_a_cell_one_below_min_count_warns_once_naming_the_cell_and_the_count(state):
    warnings = sparse_cell_warnings(_self_loop(state, 4), min_count=5)
    assert warnings == [f"Sparse transition {state}->{state}: 4 counts (<5)"]


def test_the_default_threshold_is_twenty():
    assert sparse_cell_warnings(_self_loop("bull", 20)) == []
    assert sparse_cell_warnings(_self_loop("bull", 19)) == [
        "Sparse transition bull->bull: 19 counts (<20)"
    ]


def test_a_cell_with_zero_counts_does_not_warn():
    assert sparse_cell_warnings(pd.Series(["bull"]), min_count=5) == []       # no pair at all
    assert sparse_cell_warnings(pd.Series([], dtype=str), min_count=5) == []


def test_labels_outside_state_order_are_ignored_rather_than_counted():
    # 'unknown' is not a state: the pairs into and out of it are skipped, and
    # only the two bull->bull pairs on either side of it are counted.
    broken_run = pd.Series(["bull", "bull", "unknown", "bull", "bull"])
    assert sparse_cell_warnings(broken_run, min_count=3) == [
        "Sparse transition bull->bull: 2 counts (<3)"
    ]
    assert sparse_cell_warnings(pd.Series(["bull", "unknown", "bull"]), min_count=3) == []


@pytest.mark.parametrize("labels", [
    ["bull", "bear", "sideways", "bull"],     # every visited cell holds exactly 1
    ["bull", "bull"],
    ["bear"] * 40,
])
def test_a_min_count_of_one_never_warns(labels):
    assert sparse_cell_warnings(pd.Series(labels), min_count=1) == []
