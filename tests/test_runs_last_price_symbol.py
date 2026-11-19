"""GET /runs/{id} marks a position only to its own symbol's price.

_last_price (app/api/routes/runs.py) used to fall back to the newest snapshot
of ANY symbol when the trade symbol had none, so a BTC position was marked
to an ETH price. With no BTC snapshot the unrealized P&L must read 0.0, the
value every caller already treats as "no price".
"""

from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.models.base import Base
from app.models.entities import Run, Snapshot


def _db():
    # TestClient serves requests on a worker thread; share one in-memory DB
    engine = create_engine("sqlite:///:memory:",
                           connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _client(db) -> TestClient:
    from app.dependencies import get_db_dep, get_settings_dep
    from app.main import app

    app.dependency_overrides[get_db_dep] = lambda: db
    app.dependency_overrides[get_settings_dep] = lambda: Settings()
    return TestClient(app)


def _btc_run_holding_one_unit_bought_at_100(db) -> None:
    db.add(Run(id=1, strategy="baseline", watchlist="BTC-USD,ETH-USD", trade_symbol="BTC-USD",
               cash=900.0, starting_cash=1000.0, position_qty=1.0, avg_entry_price=100.0,
               status="running"))


def test_a_position_is_not_marked_to_another_symbols_snapshot():
    db = _db()
    _btc_run_holding_one_unit_bought_at_100(db)
    db.add(Snapshot(run_id=1, symbol="ETH-USD", price=2000.0, volume=1.0, source="test"))
    db.commit()

    body = _client(db).get("/runs/1").json()

    assert body["trade_symbol"] == "BTC-USD"
    assert body["position_qty"] == 1.0
    assert body["unrealized_pnl"] == 0.0          # was 1900.0: (2000 - 100) * 1 off an ETH price


def test_a_position_is_marked_to_its_own_symbols_newest_snapshot():
    db = _db()
    _btc_run_holding_one_unit_bought_at_100(db)
    db.add(Snapshot(run_id=1, symbol="ETH-USD", price=2000.0, volume=1.0, source="test"))
    db.add(Snapshot(run_id=1, symbol="BTC-USD", price=150.0, volume=1.0, source="test"))
    db.commit()

    body = _client(db).get("/runs/1").json()

    assert body["unrealized_pnl"] == 50.0         # (150 - 100) * 1, the BTC price only
