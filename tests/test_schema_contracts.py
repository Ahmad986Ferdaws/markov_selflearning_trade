"""Contract pins for the pydantic schemas in app/schemas/.

The guards and the ledger assume a percentage is already inside [0, 100] and
an action is one of three literals. These tests make the schema layer own
that promise, so no downstream code needs a defensive re-check.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.schemas.agent import AgentDecision
from app.schemas.snapshot import NormalizedSnapshot
from app.schemas.trade import TradeIntent


@pytest.mark.parametrize("pct", [-0.01, 100.01])
def test_trade_intent_rejects_a_percentage_outside_the_closed_interval(pct):
    with pytest.raises(ValidationError):
        TradeIntent(action="BUY", percentage=pct, symbol="BTC-USD")


@pytest.mark.parametrize("pct", [0, 100])
def test_trade_intent_accepts_both_ends_of_the_interval(pct):
    assert TradeIntent(action="BUY", percentage=pct, symbol="BTC-USD").percentage == pct


def test_trade_intent_rejects_an_action_outside_the_three_literals():
    with pytest.raises(ValidationError):
        TradeIntent(action="SHORT", percentage=10, symbol="BTC-USD")


def test_trade_intent_requires_a_symbol_and_defaults_the_rest():
    with pytest.raises(ValidationError):
        TradeIntent(action="HOLD")
    intent = TradeIntent(action="HOLD", symbol="BTC-USD")
    assert (intent.percentage, intent.reasoning) == (0, "")


def test_agent_decision_bare_hold_is_a_valid_reply():
    """The prompt invites {"action": "HOLD"}; requiring percentage used to
    force a second billed repair call for nothing."""
    decision = AgentDecision(action="HOLD")
    assert (decision.percentage, decision.reasoning) == (0.0, "")


def test_agent_decision_rejects_lowercase_actions_and_out_of_range_percentages():
    with pytest.raises(ValidationError):
        AgentDecision(action="buy")
    with pytest.raises(ValidationError):
        AgentDecision(action="BUY", percentage=100.5)


def test_agent_decision_coerces_an_integer_percentage_to_float():
    decision = AgentDecision(action="BUY", percentage=15)
    assert decision.percentage == 15.0
    assert isinstance(decision.percentage, float)
