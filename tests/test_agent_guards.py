"""Phase C — fail-closed guards on live LLM calls, and the offline diagnoser.

The call-budget ceiling and cache-only replay mode are the two guards that make
the agent safe to run (and eventually to show publicly): a run can never fan out
unbounded billed calls, and a public surface can never trigger a live call at
all. Both fail CLOSED — when in doubt, hold the previous position.
"""

from __future__ import annotations

import json

import numpy as np

from app.cli.diagnose import format_diagnosis
from app.services.agent_policy import ResponseCache, build_agent_policy
from app.services.evaluation import PolicyContext


class FakeProvider:
    id, model = "fake", "fake-1"

    def __init__(self, reply: str = '{"position": 0.5}'):
        self.reply = reply
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return self.reply


def _ctx(state="bull", prev=0.0, p=(0.6, 0.2, 0.2), i=0) -> PolicyContext:
    return PolicyContext(state=state, prev_pos=prev, p_next=np.array(p, dtype=float), day_index=i)


# --- call-budget ceiling ------------------------------------------------------
def test_budget_ceiling_caps_unique_calls():
    p = FakeProvider()
    fn = build_agent_policy(p, ResponseCache(), max_unique_calls=2)
    fn(_ctx(state="bull"))
    fn(_ctx(state="bear"))
    out3 = fn(_ctx(state="sideways", prev=0.7))  # 3rd unique context -> over budget
    assert p.calls == 2                     # the ceiling held: no 3rd billed call
    assert out3 == 0.7                      # fail closed: held the previous position
    assert fn.stats.budget_holds == 1


def test_budget_ceiling_still_serves_cache_hits():
    p = FakeProvider()
    cache = ResponseCache()
    fn = build_agent_policy(p, cache, max_unique_calls=1)
    first = fn(_ctx(state="bull"))
    again = fn(_ctx(state="bull"))           # identical context -> cache hit, no budget cost
    assert first == again == 0.5
    assert p.calls == 1 and cache.hits == 1
    assert fn.stats.budget_holds == 0


# --- cache-only (replay) mode -------------------------------------------------
def test_cache_only_never_calls_provider():
    p = FakeProvider()
    fn = build_agent_policy(p, ResponseCache(), cache_only=True)
    out = fn(_ctx(prev=0.3))
    assert p.calls == 0                     # a public surface can never spend the key
    assert out == 0.3
    assert fn.stats.cache_only_holds == 1


def test_cache_only_replays_persisted_cache(tmp_path):
    # a normal run populates the cache...
    p1 = FakeProvider('{"position": 0.8}')
    path = tmp_path / "cache.json"
    c1 = ResponseCache(path)
    build_agent_policy(p1, c1)(_ctx(state="bull"))
    c1.save()
    # ...then replay mode serves it with zero live calls
    p2 = FakeProvider('{"position": 0.1}')  # would return 0.1 if (wrongly) called
    fn = build_agent_policy(p2, ResponseCache(path), cache_only=True)
    assert fn(_ctx(state="bull")) == 0.8    # replayed from disk
    assert p2.calls == 0


# --- offline diagnoser --------------------------------------------------------
def test_diagnose_narrates_the_null_result():
    rec = json.loads(open("results/BTC-USD_4a150b23.json").read())
    out = format_diagnosis(rec)
    assert "IDENTICAL to persistence" in out          # the zero-edge identity
    assert "persistence, not skill" in out
    assert "Every policy lost money" in out           # both policies negative
    assert "Honesty gates" in out and "! " in out     # the engine's warning surfaced
    assert "90.9%" in out


def test_diagnose_flags_positive_edge_for_verification():
    rec = {
        "symbol": "X", "data_hash": "abc", "policies": [],
        "accuracy": {"hit_rate": 0.9, "persistence_hit_rate": 0.8,
                     "balanced_accuracy": 0.9, "persistence_balanced": 0.8, "n": 100},
        "warnings": [],
    }
    out = format_diagnosis(rec)
    # even a positive edge is met with skepticism, not celebration
    assert "verify pointwise before believing it" in out
