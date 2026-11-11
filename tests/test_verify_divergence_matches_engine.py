"""scripts/_verify_divergence.py re-implements causal_states, the walk-forward
prediction loop, train_freq and balanced accuracy so the docs/16 claim
(0 divergences / 5425) is checked independently of the engine. Until now only
select_and_test and walk had tests, and nothing proved the four helpers agree
with app/services/evaluation.py on the same data. These tests pin them
against each other on a seeded 400-day series for the three (window, k)
pairs the study grid spans, and pin that the verifier searches the grid the
published robustness study searched.

Exact float equality is intended: both copies perform the same per-class
divisions in the same order, so their balanced accuracies are bit-identical.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from app.services import evaluation

script = importlib.import_module("scripts._verify_divergence")
study = importlib.import_module("scripts.robustness_study")

START, END, N_DAYS = 60, 279, 400
GRID_CASES = [(10, 0.2), (20, 0.5), (30, 0.75)]   # the last collapses to one class in test


def _returns() -> pd.Series:
    return pd.Series(np.random.default_rng(5).normal(0.0005, 0.02, N_DAYS),
                     index=pd.bdate_range("2019-01-01", periods=N_DAYS))


def _engine_walk(returns: pd.Series, states: list) -> dict:
    return evaluation._walk_forward(returns, states, START, END, 0.0, 0.0, policies={})


@pytest.mark.parametrize("window, k", GRID_CASES)
def test_script_states_match_engine_states(window, k):
    returns = _returns()
    states = script.causal_states(returns, window, k)
    assert states == evaluation.causal_states(returns, window, k)
    assert states[:window - 1] == [None] * (window - 1)          # same warm-up


@pytest.mark.parametrize("window, k", GRID_CASES)
def test_script_walk_matches_engine_walk_forward_day_by_day(window, k):
    returns = _returns()
    states = script.causal_states(returns, window, k)
    preds, actual, cur = script.walk(returns, states, START, END)
    wf = _engine_walk(returns, states)
    assert preds == [int(np.argmax(dist)) for dist in wf["pred_probs"]]
    assert actual == wf["actual_idx"]
    assert cur == wf["cur_idx"]
    assert len(preds) == END - START == 219                      # no day skipped by either


@pytest.mark.parametrize("window, k", GRID_CASES)
def test_script_walk_matches_engine_from_the_first_labelled_day(window, k):
    # start=window: bar window-1 is the first label, so at t=window exactly two
    # states are observed -- the smallest history either copy will estimate from
    returns = _returns()
    states = script.causal_states(returns, window, k)
    assert len([s for s in states[:window + 1] if s is not None]) == 2
    preds, actual, cur = script.walk(returns, states, window, END)
    wf = evaluation._walk_forward(returns, states, window, END, 0.0, 0.0, policies={})
    assert preds == [int(np.argmax(dist)) for dist in wf["pred_probs"]]
    assert actual == wf["actual_idx"]
    assert cur == wf["cur_idx"]
    assert len(preds) == END - window                             # t=window is predicted


@pytest.mark.parametrize("window, k", GRID_CASES)
def test_script_train_freq_matches_engine_train_freq(window, k):
    states = script.causal_states(_returns(), window, k)
    assert np.array_equal(script.train_freq(states, START, END + 1),
                          evaluation._train_freq(states, START, END + 1))


@pytest.mark.parametrize("window, k", GRID_CASES)
def test_script_balanced_accuracy_equals_engine_balanced_accuracy_exactly(window, k):
    returns = _returns()
    states = script.causal_states(returns, window, k)
    preds, actual, _ = script.walk(returns, states, START, END)
    wf = _engine_walk(returns, states)
    train_freq = evaluation._train_freq(states, START, END + 1)
    engine = evaluation._accuracy(wf["pred_probs"], wf["actual_idx"], train_freq)
    assert script.bal_acc(preds, actual) == engine.balanced_accuracy


def test_balanced_accuracy_is_non_trivial_on_the_first_two_grid_points():
    # guards the comparison above against a degenerate all-one-class series
    returns = _returns()
    for window, k in GRID_CASES[:2]:
        states = script.causal_states(returns, window, k)
        preds, actual, _ = script.walk(returns, states, START, END)
        assert len(set(actual)) == 3
        assert 0.0 < script.bal_acc(preds, actual) < 1.0


def test_verifier_searches_the_grid_the_published_study_searched():
    assert script.GRID_W == study.GRID_WINDOWS == (10, 20, 30)
    assert script.GRID_K == study.GRID_K == (0.2, 0.35, 0.5, 0.75)
