"""Offline pins for the legacy Claude agent's fallbacks (app/services/agent.py).

get_agent_decision and get_agent_decision_sync have four behaviours no test
covered: the key-less HOLD, fence-stripping through the real parse_decision,
the one-shot JSON repair turn, and the HOLD-with-reason on a provider error.
tests/test_agent.py only exercises parse_decision; elsewhere these functions
are monkeypatch targets.

No network and no real key: a fake `anthropic` module is installed in
sys.modules for the duration of each test, its `messages.create(**kwargs)`
returns scripted objects exposing `.content[0].text` (or raises), and every
call's keyword arguments are recorded.
"""

from __future__ import annotations

import json
import logging
import sys
from types import SimpleNamespace

from app.config import Settings
from app.schemas.agent import AgentDecision
from app.services import agent as agent_service
from app.services.regime import RegimeFeature

FEATURE = RegimeFeature(state="bull", p_next={"bull": 0.5, "bear": 0.2, "sideways": 0.3})
METRICS = {"price": 100.0, "volume": 5.0}
KEYLESS = Settings(anthropic_api_key="")
KEYED = Settings(anthropic_api_key="offline-test-placeholder")
REPAIR_INSTRUCTION = "Invalid JSON. Reply with valid JSON only matching the schema."


def _install_fake_anthropic(monkeypatch, replies):
    """Fake `anthropic` whose clients serve `replies` in order: a str becomes
    `.content[0].text`, an exception instance is raised. Returns the log of
    constructed clients and of every create(**kwargs)."""
    log = SimpleNamespace(constructed=[], calls=[])
    script = list(replies)

    def create_sync(**kwargs):
        log.calls.append(kwargs)
        reply = script.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return SimpleNamespace(content=[SimpleNamespace(text=reply)])

    async def create_async(**kwargs):
        return create_sync(**kwargs)

    class FakeAsync:
        def __init__(self, api_key):
            log.constructed.append(("async", api_key))
            self.messages = SimpleNamespace(create=create_async)

    class FakeSync:
        def __init__(self, api_key):
            log.constructed.append(("sync", api_key))
            self.messages = SimpleNamespace(create=create_sync)

    monkeypatch.setitem(sys.modules, "anthropic",
                        SimpleNamespace(AsyncAnthropic=FakeAsync, Anthropic=FakeSync))
    return log


def _expected_user_turn() -> dict:
    return {"role": "user", "content": json.dumps({
        "symbol": "BTC-USD",
        "metrics": METRICS,
        "regime": {"state": FEATURE.state, "p_next": FEATURE.p_next},
    })}


# --- (a) no key: HOLD, and the client is never built ------------------------

async def test_async_without_a_key_holds_and_never_constructs_a_client(monkeypatch, caplog):
    log = _install_fake_anthropic(monkeypatch, ["should never be requested"])
    caplog.set_level(logging.WARNING, logger="app.services.agent")

    decision = await agent_service.get_agent_decision(METRICS, FEATURE, "BTC-USD", KEYLESS)

    assert decision == AgentDecision(action="HOLD", percentage=0, reasoning="missing API key")
    assert log.constructed == [] and log.calls == []
    assert "No ANTHROPIC_API_KEY" in caplog.text


def test_sync_without_a_key_holds_and_never_constructs_a_client(monkeypatch):
    log = _install_fake_anthropic(monkeypatch, ["should never be requested"])

    decision = agent_service.get_agent_decision_sync(METRICS, FEATURE, "BTC-USD", KEYLESS)

    assert decision == AgentDecision(action="HOLD", percentage=0, reasoning="missing API key")
    assert log.constructed == [] and log.calls == []


# --- (b) a fenced reply parses through the real parse_decision ---------------

async def test_a_fenced_json_reply_parses_to_the_decision(monkeypatch):
    log = _install_fake_anthropic(
        monkeypatch, ['```json\n{"action":"BUY","percentage":15}\n```'])

    decision = await agent_service.get_agent_decision(METRICS, FEATURE, "BTC-USD", KEYED)

    assert (decision.action, decision.percentage) == ("BUY", 15)
    assert log.constructed == [("async", "offline-test-placeholder")]
    assert len(log.calls) == 1                              # no repair turn was needed
    call = log.calls[0]
    assert call["model"] == KEYED.anthropic_model
    assert call["system"] == agent_service.SYSTEM_PROMPT
    assert call["messages"] == [_expected_user_turn()]


# --- (c) one invalid reply triggers exactly one repair turn -----------------

async def test_an_invalid_reply_is_repaired_once_with_the_raw_reply_as_the_assistant_turn(monkeypatch):
    log = _install_fake_anthropic(
        monkeypatch, ["not json", '{"action":"SELL","percentage":40,"reasoning":"repaired"}'])

    decision = await agent_service.get_agent_decision(METRICS, FEATURE, "BTC-USD", KEYED)

    assert decision == AgentDecision(action="SELL", percentage=40, reasoning="repaired")
    assert len(log.calls) == 2
    repair_messages = log.calls[1]["messages"]
    assert len(repair_messages) == 3
    assert repair_messages[0] == _expected_user_turn()
    assert repair_messages[1] == {"role": "assistant", "content": "not json"}
    assert repair_messages[2] == {"role": "user", "content": REPAIR_INSTRUCTION}
    assert log.calls[1]["system"] == agent_service.SYSTEM_PROMPT


# --- (d) a provider error degrades to HOLD with the reason -------------------

async def test_async_provider_error_holds_with_the_reason_and_logs_an_error(monkeypatch, caplog):
    _install_fake_anthropic(monkeypatch, [RuntimeError("boom")])
    caplog.set_level(logging.ERROR, logger="app.services.agent")

    decision = await agent_service.get_agent_decision(METRICS, FEATURE, "BTC-USD", KEYED)

    assert decision == AgentDecision(action="HOLD", percentage=0, reasoning="agent error: boom")
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert [r.getMessage() for r in errors] == ["agent call failed"]
    assert "RuntimeError: boom" in caplog.text               # the traceback travels with it


def test_sync_provider_error_holds_with_the_reason_and_logs_an_error(monkeypatch, caplog):
    _install_fake_anthropic(monkeypatch, [RuntimeError("boom")])
    caplog.set_level(logging.ERROR, logger="app.services.agent")

    decision = agent_service.get_agent_decision_sync(METRICS, FEATURE, "BTC-USD", KEYED)

    assert decision == AgentDecision(action="HOLD", percentage=0, reasoning="agent error: boom")
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert [r.getMessage() for r in errors] == ["agent sync call failed"]
    assert "RuntimeError: boom" in caplog.text
