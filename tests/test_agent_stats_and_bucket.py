"""Pins for the agent policy's bookkeeping helpers in app/services/agent_policy.py:
AgentStats tallying, the round-half-even forecast bucketing, and the state
truncation in _format_context. No provider is ever called.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.services.agent_policy import AgentStats, _bucket, _format_context


# --- AgentStats -----------------------------------------------------------------------
def test_tally_increments_a_known_tier_and_ignores_an_unknown_one():
    stats = AgentStats()
    stats.tally("strict_json")
    stats.tally("strict_json")
    stats.tally("salvaged_labeled")
    stats.tally("not_a_tier")                      # silently ignored, never an attribute
    assert (stats.strict_json, stats.salvaged_labeled) == (2, 1)
    assert not hasattr(stats, "not_a_tier")
    assert stats.decisions == 3


def test_decisions_is_the_sum_of_all_seven_counters():
    stats = AgentStats(errors=1, strict_json=2, salvaged_labeled=3, salvaged_single_number=4,
                       unparseable_hold=5, budget_holds=6, cache_only_holds=7)
    assert stats.decisions == 28


# --- _bucket: round-half-even, so exact ties land on the EVEN multiple --------------------
# Only ties whose ratio is exactly representable are pinned: 0.025/0.05 is exactly 0.5 and
# 0.125/0.05 exactly 2.5, whereas 0.075/0.05 is 1.4999999999999998 in binary floating point
# and is therefore not a tie at all (it rounds down to 0.05).
@pytest.mark.parametrize("value, step, expected", [
    (0.025, 0.05, 0.0),      # ratio 0.5  -> 0 (even)
    (0.125, 0.05, 0.1),      # ratio 2.5  -> 2 (even)
    (0.225, 0.05, 0.2),      # ratio 4.5  -> 4 (even)
    (0.075, 0.05, 0.05),     # not a tie in float: 1.4999... -> 1
    (0.1249, 0.05, 0.1),
    (1.0, 0.05, 1.0),
    (0.0, 0.05, 0.0),
    (0.25, 0.1, 0.2),        # the position step: ratio 2.5 -> 2 (even)
    (0.05, 0.1, 0.0),        # ratio 0.5 -> 0
])
def test_bucket_rounds_half_to_even(value, step, expected):
    assert _bucket(value, step) == expected


@pytest.mark.parametrize("value", [0.3, 0.7, 0.9, 0.35, 0.65])
def test_bucket_output_carries_no_float_noise_beyond_four_decimals(value):
    out = _bucket(value, 0.05)
    assert out == round(out, 4)
    assert len(repr(out).split(".")[-1]) <= 4


# --- _format_context: only the states in STATE_ORDER are ever reported --------------------
def _ctx(p_next, state="bull", prev_pos=0.25):
    return SimpleNamespace(state=state, prev_pos=prev_pos, p_next=np.asarray(p_next, dtype=float))


def test_format_context_reports_at_most_the_three_known_states():
    out = _format_context(_ctx([0.1, 0.2, 0.3, 0.4]))      # a fourth entry has no state name
    assert list(out["forecast_next_regime"]) == ["bull", "bear", "sideways"]
    assert out["most_likely_next"] == "sideways"             # argmax over the three kept entries
    assert out["current_position"] == 0.2                    # 0.25 bucketed half-even at step 0.1


def test_format_context_with_a_short_forecast_reports_only_what_is_present():
    out = _format_context(_ctx([0.3, 0.7], state="bear", prev_pos=0.0))
    assert out["forecast_next_regime"] == {"bull": 0.3, "bear": 0.7}
    assert out["most_likely_next"] == "bear"
    assert out["today_regime"] == "bear"


def test_format_context_with_an_empty_forecast_says_unknown():
    out = _format_context(_ctx([], state="sideways", prev_pos=1.0))
    assert out["forecast_next_regime"] == {}
    assert out["most_likely_next"] == "unknown"
    assert out["current_position"] == 1.0
