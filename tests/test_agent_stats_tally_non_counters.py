"""AgentStats.tally (app/services/agent_policy.py) only counts the dataclass's own counters.

The gate used to be ``hasattr(self, tier)``, which is also true for the
``decisions`` property and the ``tally`` method itself: ``tally("decisions")``
raised AttributeError (property without a setter) and ``tally("tally")`` raised
TypeError, while a genuinely unknown name was silently ignored. The gate is now
the dataclass field set, so every non-counter name is ignored the same way.
"""

from __future__ import annotations

import dataclasses

import pytest

from app.services.agent_policy import AgentStats

COUNTERS = ("errors", "strict_json", "salvaged_labeled", "salvaged_single_number",
            "unparseable_hold", "budget_holds", "cache_only_holds")


@pytest.mark.parametrize("name", ["decisions", "tally", "not_a_tier"])
def test_tallying_a_name_that_is_not_a_counter_neither_raises_nor_changes_anything(name):
    stats = AgentStats(strict_json=2, errors=1)
    before = dataclasses.asdict(stats)
    stats.tally(name)                                   # must not raise
    assert dataclasses.asdict(stats) == before
    assert stats.decisions == 3
    assert not hasattr(stats, "not_a_tier")


def test_tallying_a_counter_still_increments_it():
    stats = AgentStats()
    stats.tally("strict_json")
    stats.tally("strict_json")
    assert stats.strict_json == 2
    assert stats.decisions == 2


def test_the_tally_gate_is_exactly_the_seven_counters():
    assert set(AgentStats().__dataclass_fields__) == set(COUNTERS)
    stats = AgentStats()
    for name in COUNTERS:
        stats.tally(name)
    assert dataclasses.asdict(stats) == {name: 1 for name in COUNTERS}
    assert stats.decisions == 7
