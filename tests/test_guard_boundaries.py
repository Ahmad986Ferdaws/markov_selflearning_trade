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


# --- allocation cap (BUY only) ------------------------------------------------
def test_buy_exactly_at_the_cap_is_not_clamped():
    s = Settings(max_allocation_pct=20.0)
    out = _enforce(TradeIntent(action="BUY", percentage=20.0, symbol="BTC-USD"), s)
    assert (out.action, out.percentage, out.guard_note) == ("BUY", 20.0, None)


def test_buy_just_over_the_cap_is_clamped_with_a_note():
    s = Settings(max_allocation_pct=20.0)
    out = _enforce(TradeIntent(action="BUY", percentage=20.01, symbol="BTC-USD"), s)
    assert (out.action, out.percentage) == ("BUY", 20.0)
    assert "clamped" in out.guard_note


def test_sell_percentage_is_not_subject_to_the_allocation_cap():
    """SELL is '% of the position to exit' and must be able to reach 100%."""
    s = Settings(max_allocation_pct=20.0)
    out = _enforce(TradeIntent(action="SELL", percentage=100.0, symbol="BTC-USD"), s, qty=2.0)
    assert (out.action, out.percentage, out.guard_note) == ("SELL", 100.0, None)


def test_buy_with_no_cash_is_rejected():
    out = _enforce(TradeIntent(action="BUY", percentage=10.0, symbol="BTC-USD"), Settings(), cash=0.0)
    assert out.action == "HOLD"
    assert "insufficient cash" in out.guard_note


# --- liquidity floor ------------------------------------------------------------
def test_liquidity_floor_is_inert_when_no_liquidity_is_reported():
    """yfinance reports no liquidity, so the shipped pipeline never trips this
    guard; the test pins that None means 'unknown', not 'zero'."""
    s = Settings(liquidity_floor_usd=50_000.0)
    out = _enforce(TradeIntent(action="BUY", percentage=10.0, symbol="BTC-USD"), s, liquidity=None)
    assert out.action == "BUY"


def test_liquidity_exactly_at_the_floor_passes():
    s = Settings(liquidity_floor_usd=50_000.0)
    out = _enforce(TradeIntent(action="BUY", percentage=10.0, symbol="BTC-USD"), s, liquidity=50_000.0)
    assert out.action == "BUY"


def test_liquidity_just_below_the_floor_holds_with_a_note():
    s = Settings(liquidity_floor_usd=50_000.0)
    out = _enforce(TradeIntent(action="SELL", percentage=10.0, symbol="BTC-USD"), s,
                   qty=1.0, liquidity=49_999.99)
    assert out.action == "HOLD"
    assert "liquidity" in out.guard_note and "floor" in out.guard_note


# --- effective price --------------------------------------------------------------
def test_effective_price_marks_buys_up_and_sells_down_by_the_same_amount():
    s = Settings(fee_pct=0.3, slippage_pct=0.5)
    buy_px, buy_fee = effective_price(100.0, "BUY", s)
    sell_px, sell_fee = effective_price(100.0, "SELL", s)
    assert buy_px == pytest.approx(100.8)
    assert sell_px == pytest.approx(99.2)
    assert buy_fee == pytest.approx(sell_fee) == pytest.approx(0.8)


def test_effective_price_is_the_identity_with_zero_costs():
    s = Settings(fee_pct=0.0, slippage_pct=0.0)
    for side in ("BUY", "SELL"):
        px, fee = effective_price(123.45, side, s)
        assert (px, fee) == (123.45, 0.0)
