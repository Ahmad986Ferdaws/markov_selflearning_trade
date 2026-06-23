"""Phase C — LLM agent policy. Fully offline via a FakeProvider (no live LLM)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.agent_policy import (
    ResponseCache,
    _format_context,
    _parse_position,
    build_agent_policy,
    make_provider,
)
from app.services.evaluation import DEFAULT_POLICIES, PolicyContext, evaluate


class FakeProvider:
    """Records call count; returns a fixed reply."""

    id = "fake"
    model = "fake-1"

    def __init__(self, reply: str = '{"position": 1.0, "reasoning": "go long"}'):
        self.reply = reply
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return self.reply


def _ctx(state="bull", prev=0.0, p=(0.6, 0.2, 0.2), i=0) -> PolicyContext:
    return PolicyContext(state=state, prev_pos=prev, p_next=np.array(p, dtype=float), day_index=i)


# --- parsing / clamping -----------------------------------------------------
def test_parses_position():
    fn = build_agent_policy(FakeProvider('{"position":0.5,"reasoning":"x"}'))
    assert fn(_ctx()) == 0.5


def test_position_clamped_high_and_low():
    assert build_agent_policy(FakeProvider('{"position": 4.2}'))(_ctx()) == 1.0
    assert build_agent_policy(FakeProvider('{"position": -3}'))(_ctx()) == 0.0


def test_malformed_json_holds_previous():
    fn = build_agent_policy(FakeProvider("not json at all, sorry"))
    assert fn(_ctx(prev=0.3)) == 0.3


def test_salvages_labeled_number_from_prose():
    pos, why = _parse_position("I would take position 0.7 long here", 0.0)
    assert pos == 0.7 and "salvage" in why


def test_labeled_value_beats_stray_number():
    # a stray z-score 0.5 precedes the real exposure 0.8 — must NOT return 0.5
    pos, why = _parse_position("Given a 0.5 z-score I'd set exposure to 0.8", 0.0)
    assert pos == 0.8


def test_multiple_unlabeled_numbers_hold():
    # two plausible numbers, no label -> too ambiguous, hold rather than guess
    pos, why = _parse_position("the 0.3 and 0.9 cases both look fine", 0.4)
    assert pos == 0.4 and "hold" in why


def test_single_bare_number_salvaged():
    pos, why = _parse_position("0.6", 0.0)
    assert pos == 0.6


def test_accepts_alt_key_exposure():
    pos, _ = _parse_position('{"exposure": 0.25, "reasoning": "x"}', 0.0)
    assert pos == 0.25


def test_no_number_falls_back():
    pos, why = _parse_position("totally unparseable", 0.25)
    assert pos == 0.25 and "hold" in why


# --- caching / reproducibility ---------------------------------------------
def test_identical_contexts_hit_cache_once():
    p = FakeProvider()
    cache = ResponseCache()
    fn = build_agent_policy(p, cache)
    c = _ctx(state="bull", prev=0.0, p=(0.6, 0.2, 0.2))
    fn(c); fn(c); fn(c)
    assert p.calls == 1      # provider invoked exactly once
    assert cache.hits == 2   # other two served from cache


def test_nearby_contexts_bucket_together():
    # forecasts within the 0.05 bucket must collapse to ONE provider call
    p = FakeProvider()
    fn = build_agent_policy(p, ResponseCache())
    fn(_ctx(p=(0.61, 0.20, 0.19)))
    fn(_ctx(p=(0.62, 0.19, 0.19)))
    assert p.calls == 1


def test_distinct_states_call_provider_separately():
    p = FakeProvider()
    fn = build_agent_policy(p, ResponseCache())
    fn(_ctx(state="bull"))
    fn(_ctx(state="bear"))
    assert p.calls == 2


def test_cache_persists_and_reloads(tmp_path):
    path = tmp_path / "agent.json"
    cache = ResponseCache(path)
    build_agent_policy(FakeProvider(), cache)(_ctx())
    cache.save()
    assert path.exists()
    assert len(ResponseCache(path)._d) == 1  # survives a fresh load


# --- error handling ---------------------------------------------------------
def test_provider_error_holds_and_counts():
    class Boom:
        id, model = "boom", "x"

        def complete(self, system, user):
            raise RuntimeError("down")

    fn = build_agent_policy(Boom())
    assert fn(_ctx(prev=0.4)) == 0.4
    assert fn.stats.errors == 1


# --- provider selection -----------------------------------------------------
class _S:
    llm_provider = "none"
    anthropic_api_key = ""
    anthropic_model = "m"
    ollama_model = "q"
    ollama_host = "http://h"
    agent_temperature = 0.0


def test_make_provider_none_by_default():
    assert make_provider(_S()) is None


def test_make_provider_anthropic_requires_key():
    s = _S(); s.llm_provider = "anthropic"
    assert make_provider(s) is None      # empty key -> disabled
    s.anthropic_api_key = "sk-test"
    assert make_provider(s).id == "anthropic"


def test_make_provider_ollama():
    s = _S(); s.llm_provider = "ollama"
    assert make_provider(s).id == "ollama"


# --- integration: same engine, no special-casing ----------------------------
def test_agent_flows_through_engine_like_any_policy():
    rng = np.random.default_rng(0)
    closes = pd.Series(100 * np.cumprod(1 + rng.normal(0.001, 0.02, 400)))
    hist = pd.DataFrame({"Close": closes})

    policies = dict(DEFAULT_POLICIES)
    policies["agent"] = build_agent_policy(FakeProvider('{"position": 1.0}'))
    report = evaluate(hist, symbol="TEST", min_train=60, policies=policies)

    names = [p.name for p in report.policies]
    assert "agent" in names
    # An always-long agent must reproduce buy_hold EXACTLY — proof it runs through
    # the identical fill + cost path, with zero engine changes.
    agent = next(p for p in report.policies if p.name == "agent")
    bh = next(p for p in report.policies if p.name == "buy_hold")
    assert abs(agent.total_return - bh.total_return) < 1e-9
    assert agent.num_trades == bh.num_trades


def test_format_context_buckets_and_orders():
    ctx = _ctx(state="sideways", prev=0.33, p=(0.137, 0.143, 0.72))
    out = _format_context(ctx)
    assert out["today_regime"] == "sideways"
    assert out["most_likely_next"] == "sideways"   # argmax of p
    assert out["current_position"] == 0.3          # bucketed to 0.1
    assert set(out["forecast_next_regime"]) == {"bull", "bear", "sideways"}
