"""Boundary pins for app/services/guards.py — the hardcoded risk guards.

Each test sits exactly on a threshold the guard compares against (the
allocation cap, the liquidity floor, zero cash) so a future '<' vs '<=' slip
shows up as a red test rather than a silently different fill.
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.schemas.trade import TradeIntent
from app.services.guards import effective_price, enforce_decision


def _enforce(intent: TradeIntent, settings: Settings, *, cash=1_000.0, qty=0.0,
             price=100.0, liquidity=None):
    return enforce_decision(intent, cash=cash, position_qty=qty, price=price,
                            liquidity_usd=liquidity, settings=settings)


# --- HOLD and zero-size intents ---------------------------------------------
def test_hold_intent_is_returned_as_hold_with_no_note():
    out = _enforce(TradeIntent(action="HOLD", percentage=50, symbol="BTC-USD"), Settings())
    assert (out.action, out.percentage, out.guard_note) == ("HOLD", 0.0, None)


@pytest.mark.parametrize("action", ["BUY", "SELL"])
def test_zero_percentage_collapses_to_hold_without_a_note(action):
    out = _enforce(TradeIntent(action=action, percentage=0, symbol="BTC-USD"),
                   Settings(), qty=1.0)
    assert out.action == "HOLD"
    assert out.guard_note is None
