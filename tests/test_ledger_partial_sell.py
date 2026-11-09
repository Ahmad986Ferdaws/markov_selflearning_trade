"""The paper ledger's partial SELL (app/services/ledger.py apply_trade):
the exited share is priced at the fee- and slippage-adjusted price, realised
P&L is booked against the average entry, the remaining position keeps its
basis unchanged (a partial exit never re-averages), and only a full exit
flattens quantity and basis to 0.0.

tests/test_ledger_accounting.py pins buys and a 100% exit; no test sold part
of a position. Same in-memory fixture and the shipped default costs (0.3% fee
+ 0.5% slippage = 0.8% each way, 20% allocation cap), so the numbers are the
ones a real legacy run would produce.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.models.base import Base
from app.models.entities import Run
from app.schemas.trade import TradeIntent
from app.services.ledger import apply_trade, portfolio_value

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


def test_a_partial_sell_realises_pnl_against_entry_and_leaves_the_basis_unchanged(db, run):
    apply_trade(db, run, _intent("BUY", 10.0), price=100.0, settings=SETTINGS)
    db.refresh(run)
    q0, e0, cash0 = run.position_qty, run.avg_entry_price, run.cash
    assert q0 == pytest.approx(100.0 / 100.8)                # 10% of 1000 at 100 * 1.008
    assert e0 == pytest.approx(100.8)

    trade = apply_trade(db, run, _intent("SELL", 50.0), price=110.0, settings=SETTINGS)
    db.refresh(run)

    assert (trade.enforced_action, trade.enforced_percentage, trade.guard_note) == ("SELL", 50.0, None)
    assert trade.quantity == pytest.approx(q0 / 2)
    assert trade.execution_price == pytest.approx(109.12)    # 110 * (1 - 0.008)
    assert trade.fees_paid == pytest.approx(q0 / 2 * 110.0 * 0.008)
    assert run.position_qty == pytest.approx(q0 / 2)
    assert run.avg_entry_price == e0                          # a partial exit never re-averages
    assert run.realized_pnl == pytest.approx(q0 / 2 * (109.12 - 100.8))
    assert run.cash - cash0 == pytest.approx(q0 / 2 * 109.12)
    assert portfolio_value(run, 110.0) == pytest.approx(run.cash + q0 / 2 * 110.0)


def test_successive_partial_sells_keep_the_basis_until_the_final_flatten(db, run):
    apply_trade(db, run, _intent("BUY", 10.0), price=100.0, settings=SETTINGS)
    db.refresh(run)
    q0, e0 = run.position_qty, run.avg_entry_price

    apply_trade(db, run, _intent("SELL", 50.0), price=110.0, settings=SETTINGS)
    apply_trade(db, run, _intent("SELL", 50.0), price=120.0, settings=SETTINGS)
    db.refresh(run)
    assert run.position_qty == pytest.approx(q0 / 4)
    assert run.avg_entry_price == e0

    apply_trade(db, run, _intent("SELL", 100.0), price=130.0, settings=SETTINGS)
    db.refresh(run)
    assert (run.position_qty, run.avg_entry_price) == (0.0, 0.0)
