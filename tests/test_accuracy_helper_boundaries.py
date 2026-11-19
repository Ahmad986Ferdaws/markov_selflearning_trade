"""Boundary pins for the scoring helpers in app/services/evaluation.py.

_accuracy, _policy_metrics and _train_freq each have an empty-input branch
and _accuracy has a second, persistence-less branch (no cur_idx) that the
train-region selection walk relies on. These tests sit on those branches:
nothing scored, nothing traded, a region with no labels, and the exact
definition of a "switch attempt".
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from app.services.evaluation import (
    AccuracyMetrics,
    PolicyResult,
    _accuracy,
    _policy_metrics,
    _train_freq,
)

UNIFORM = np.ones(3) / 3


def _dist(predicted: int) -> np.ndarray:
    """A forecast whose argmax is `predicted` (all mass on one state)."""
    v = np.zeros(3)
    v[predicted] = 1.0
    return v


# --- _accuracy: nothing to score -----------------------------------------------
def test_no_predictions_scores_every_metric_at_zero_with_n_zero():
    acc = _accuracy([], [], UNIFORM)
    assert isinstance(acc, AccuracyMetrics)
    assert acc.n == 0
    assert all(value == 0 for value in asdict(acc).values())


# --- _accuracy: the persistence-less path (no cur_idx) -------------------------
def test_without_cur_idx_the_persistence_and_switch_counters_stay_zero():
    probs = [_dist(1), _dist(0)]
    actual = [1, 1]
    acc = _accuracy(probs, actual, UNIFORM)
    assert acc.n == 2 and acc.hit_rate == 0.5            # predictions were scored
    assert acc.persistence_hit_rate == 0.0
    assert acc.persistence_balanced == 0.0
    assert (acc.n_switch, acc.switch_attempts) == (0, 0)
    assert acc.switch_recall == 0.0


# --- _accuracy: what counts as a switch attempt --------------------------------
@pytest.mark.parametrize("actual, right", [(1, True), (0, False), (2, False)])
def test_predicting_a_state_other_than_today_is_a_switch_attempt_right_or_wrong(actual, right):
    acc = _accuracy([_dist(1)], [actual], UNIFORM, cur_idx=[0])   # today 0, model says 1
    assert acc.switch_attempts == 1
    assert acc.hit_rate == (1.0 if right else 0.0)


def test_predicting_today_again_is_not_a_switch_attempt_even_when_the_regime_moves():
    acc = _accuracy([_dist(0)], [1], UNIFORM, cur_idx=[0])        # today 0, model says 0
    assert acc.switch_attempts == 0
    assert (acc.n_switch, acc.switch_recall) == (1, 0.0)


# --- _train_freq: a region with no labels ---------------------------------------
def test_train_freq_over_an_all_none_region_is_uniform():
    np.testing.assert_allclose(_train_freq([None] * 10, 0, 10), UNIFORM)
    # labels outside [start, end) do not leak into the region's frequencies
    states = ["bull", "bull", None, None, "bear"]
    np.testing.assert_allclose(_train_freq(states, 2, 4), UNIFORM)


# --- _policy_metrics: nothing traded, and the single entry fill -----------------
def test_no_positions_gives_a_zero_policy_row():
    assert _policy_metrics("x", [], [], 0.0) == PolicyResult("x", 0.0, 0.0, 0.0, 0, 0.0)


def test_constant_long_on_a_flat_market_pays_exactly_one_entry_cost():
    cost_rate = 0.008
    result = _policy_metrics("long", [1.0] * 5, [0.0] * 5, cost_rate)
    assert result.num_trades == 1
    assert result.total_cost == cost_rate
    assert result.total_return == pytest.approx(-cost_rate)
