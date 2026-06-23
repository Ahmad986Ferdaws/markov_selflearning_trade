from app.config import Settings
from app.schemas.trade import TradeIntent
from app.services.guards import enforce_decision


def test_allocation_clamped():
    s = Settings(max_allocation_pct=20.0, liquidity_floor_usd=0)
    intent = TradeIntent(action="BUY", percentage=100.0, symbol="BTC-USD")
    out = enforce_decision(intent, cash=1000, position_qty=0, price=100, liquidity_usd=None, settings=s)
    assert out.percentage == 20.0
    assert "clamped" in (out.guard_note or "")


def test_sell_rejected_without_position():
    s = Settings()
    intent = TradeIntent(action="SELL", percentage=50, symbol="BTC-USD")
    out = enforce_decision(intent, cash=1000, position_qty=0, price=100, liquidity_usd=None, settings=s)
    assert out.action == "HOLD"
    assert out.guard_note is not None
