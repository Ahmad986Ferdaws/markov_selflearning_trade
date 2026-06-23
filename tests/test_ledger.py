from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.models.base import Base
from app.models.entities import Run
from app.schemas.trade import TradeIntent
from app.services.ledger import apply_trade


def test_buy_reduces_cash():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    run = Run(cash=1000.0, position_qty=0.0, trade_symbol="BTC-USD", watchlist="BTC-USD")
    db.add(run)
    db.commit()
    db.refresh(run)
    settings = Settings(fee_pct=0.3, slippage_pct=0.5, max_allocation_pct=20)
    intent = TradeIntent(action="BUY", percentage=10, symbol="BTC-USD")
    apply_trade(db, run, intent, price=100.0, settings=settings)
    db.refresh(run)
    assert run.cash < 1000.0
    assert run.position_qty > 0
    db.close()
