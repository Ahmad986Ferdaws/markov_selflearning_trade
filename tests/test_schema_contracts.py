"""Contract pins for the pydantic schemas in app/schemas/.

The guards and the ledger assume a percentage is already inside [0, 100] and
an action is one of three literals. These tests make the schema layer own
that promise, so no downstream code needs a defensive re-check.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

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


def test_snapshot_defaults_source_to_yfinance_and_volume_to_none():
    ts = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    snap = NormalizedSnapshot(symbol="BTC-USD", price=1.0, timestamp=ts)
    assert (snap.source, snap.volume) == ("yfinance", None)


def test_truncated_metrics_is_json_safe_and_keeps_a_real_zero_volume():
    ts = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    snap = NormalizedSnapshot(symbol="BTC-USD", price=1.5, volume=0.0, timestamp=ts)
    metrics = snap.truncated_metrics()
    assert set(metrics) == {"symbol", "price", "volume", "timestamp"}
    assert metrics["timestamp"] == "2026-01-02T03:04:05+00:00"   # zone preserved
    assert metrics["volume"] == 0.0                              # zero is not None
    json.dumps(metrics)                                          # ledger stores it as text
