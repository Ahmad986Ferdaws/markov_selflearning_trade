"""Pins for WalkForwardConfig's field validation.

walk_forward's fold loop advances with `fold_start += wf.test`, so a zero or
negative test window would never terminate, and an empty entry/exit grid
would hit `wf.entry_exit_grid[0]` with an IndexError inside the first fold.
The dataclass now rejects both at construction. This file only constructs
configs; it never calls walk_forward with a bad one, because a regression
there would hang the suite rather than fail it.

tests/test_walk_forward_fold_layout.py and tests/test_kalman_report_zero_folds.py
pin the layouts that positive windows produce; the accepted shapes are
re-asserted here so the defaults cannot drift.
"""

from __future__ import annotations

import pytest

from kalman.research import WalkForwardConfig


@pytest.mark.parametrize("field, value", [
    ("test", 0),
    ("test", -1),
    ("validation", 0),
    ("train", 0),
])
def test_a_window_below_one_bar_is_rejected_by_name(field, value):
    with pytest.raises(ValueError, match=rf"WalkForwardConfig\.{field} must be an int >= 1"):
        WalkForwardConfig(**{field: value})


def test_a_bool_window_is_rejected_even_though_bool_is_an_int():
    with pytest.raises(ValueError, match=r"WalkForwardConfig\.train must be an int >= 1"):
        WalkForwardConfig(train=True)


def test_a_float_window_is_rejected():
    with pytest.raises(ValueError, match=r"WalkForwardConfig\.test must be an int >= 1"):
        WalkForwardConfig(test=250.0)


def test_an_empty_entry_exit_grid_is_rejected():
    with pytest.raises(ValueError, match=r"WalkForwardConfig\.entry_exit_grid"):
        WalkForwardConfig(entry_exit_grid=())


def test_a_grid_cell_that_is_not_a_pair_is_rejected():
    with pytest.raises(ValueError, match=r"WalkForwardConfig\.entry_exit_grid"):
        WalkForwardConfig(entry_exit_grid=((2.0, 0.5), (1.5,)))


def test_the_default_config_constructs_unchanged():
    wf = WalkForwardConfig()
    assert (wf.train, wf.validation, wf.test) == (1000, 250, 250)
    assert wf.expanding is True
    assert wf.entry_exit_grid == ((1.5, 0.5), (2.0, 0.5), (2.5, 1.0))


def test_the_reduced_test_layout_constructs_unchanged():
    wf = WalkForwardConfig(train=400, validation=120, test=120)
    assert (wf.train, wf.validation, wf.test) == (400, 120, 120)
    assert WalkForwardConfig(train=1, validation=1, test=1).test == 1     # the floor itself
