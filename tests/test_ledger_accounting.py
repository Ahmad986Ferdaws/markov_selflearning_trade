"""Accounting identities for the paper ledger (app/services/ledger.py).

Every test runs against an in-memory SQLite database with one Run. Costs are
the shipped defaults (0.3% fee + 0.5% slippage, 20% allocation cap) so the
numbers below are the ones a real legacy run would produce.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.models.base import Base
from app.models.entities import Run, Trade
from app.schemas.trade import TradeIntent
from app.services.ledger import apply_trade, mark_to_market, portfolio_value

SETTINGS = Settings(fee_pct=0.3, slippage_pct=0.5, max_allocation_pct=20.0)


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


@pytest.fixture
def run(db):
    row = Run(cash=1000.0, starting_cash=1000.0, position_qty=0.0,
              trade_symbol="BTC-USD", watchlist="BTC-USD")
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _intent(action: str, pct: float) -> TradeIntent:
    return TradeIntent(action=action, percentage=pct, symbol="BTC-USD")


def test_a_round_trip_at_a_flat_price_loses_exactly_the_costs(db, run):
    buy = apply_trade(db, run, _intent("BUY", 10.0), price=100.0, settings=SETTINGS)
    sell = apply_trade(db, run, _intent("SELL", 100.0), price=100.0, settings=SETTINGS)
    db.refresh(run)
    assert run.position_qty == 0.0 and run.avg_entry_price == 0.0      # fully flat again
    assert run.cash == pytest.approx(1000.0 + run.realized_pnl)        # cash moved only by P&L
    assert run.realized_pnl == pytest.approx(-(buy.fees_paid + sell.fees_paid))
    assert run.realized_pnl < 0
    # 10% of 1000 round-tripped through 0.8% each way: 100 * (1 - 99.2 / 100.8)
    assert run.cash == pytest.approx(1000.0 - 100.0 * (1 - 99.2 / 100.8))


def test_a_plain_hold_writes_no_trade_row(db, run):
    assert apply_trade(db, run, _intent("HOLD", 0.0), price=100.0, settings=SETTINGS) is None
    assert db.query(Trade).count() == 0
    db.refresh(run)
    assert (run.cash, run.position_qty) == (1000.0, 0.0)


def test_a_guarded_rejection_is_recorded_as_a_hold_with_the_reason(db, run):
    trade = apply_trade(db, run, _intent("SELL", 50.0), price=100.0, settings=SETTINGS)
    assert trade is not None
    assert (trade.intent_action, trade.enforced_action) == ("SELL", "HOLD")
    assert (trade.quantity, trade.fees_paid, trade.enforced_percentage) == (0.0, 0.0, 0.0)
    assert "no position" in trade.guard_note
    db.refresh(run)
    assert (run.cash, run.position_qty, run.realized_pnl) == (1000.0, 0.0, 0.0)


def test_a_buy_over_the_cap_is_clamped_and_both_percentages_are_kept(db, run):
    trade = apply_trade(db, run, _intent("BUY", 100.0), price=100.0, settings=SETTINGS)
    assert (trade.intent_percentage, trade.enforced_percentage) == (100.0, 20.0)
    db.refresh(run)
    assert run.cash == pytest.approx(800.0)                 # exactly 20% committed
    assert run.position_qty * trade.execution_price == pytest.approx(200.0)
    assert trade.execution_price == pytest.approx(100.8)
