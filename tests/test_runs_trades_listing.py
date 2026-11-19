"""GET /runs/{id}/trades (app/api/routes/runs.py list_trades): the response
mirrors every TradeRecordResponse field from the Trade row, is ordered by
created_at (not by id), and keeps the zero-quantity guard HOLD row that a
rejected intent leaves in the history.

The lifecycle test only ever reads an empty run and the web-surface test a
404, so the serialisation and the ordering were unpinned. Offline: a shared
in-memory StaticPool database behind the dependency overrides, start_run_task
and request_stop replaced by no-ops, trades booked through the real
apply_trade with the shipped default costs (0.3% fee + 0.5% slippage, 20% cap).
created_at is written explicitly so SQLite's same-second CURRENT_TIMESTAMP
cannot tie the two rows.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.models.base import Base
from app.models.entities import Run, Trade
from app.schemas.runs import TradeRecordResponse
from app.schemas.trade import TradeIntent
from app.services.ledger import apply_trade


@pytest.fixture
def api(monkeypatch):
    from app.api.routes import runs as runs_module
    from app.dependencies import get_db_dep, get_settings_dep
    from app.main import app

    monkeypatch.setattr(runs_module, "start_run_task", lambda run_id, settings: None)
    monkeypatch.setattr(runs_module, "request_stop", lambda run_id: None)
    engine = create_engine("sqlite:///:memory:",
                           connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    settings = Settings()
    app.dependency_overrides[get_db_dep] = lambda: db
    app.dependency_overrides[get_settings_dep] = lambda: settings
    try:
        yield SimpleNamespace(client=TestClient(app), db=db, settings=settings)
    finally:
        app.dependency_overrides.clear()
        db.close()


def _intent(action: str, pct: float) -> TradeIntent:
    return TradeIntent(action=action, percentage=pct, symbol="BTC-USD")


def test_trades_listing_mirrors_fields_orders_by_created_at_and_keeps_the_guard_row(api):
    created = api.client.post("/runs", json={"strategy": "baseline",
                                             "ingestion_provider": "yfinance",
                                             "watchlist": ["BTC-USD"]}).json()
    run = api.db.get(Run, created["id"])
    settings = Settings()
    guard = apply_trade(api.db, run, _intent("SELL", 50.0), 100.0, settings)   # no position
    buy = apply_trade(api.db, run, _intent("BUY", 10.0), 100.0, settings)
    assert (guard.id, buy.id) == (1, 2)
    guard.created_at = datetime(2024, 1, 1, 0, 0, 2)        # inserted first, happened later
    buy.created_at = datetime(2024, 1, 1, 0, 0, 1)
    api.db.commit()

    resp = api.client.get(f"/runs/{created['id']}/trades")

    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 2
    assert [r["id"] for r in rows] == [2, 1]                # by created_at, not by id
    assert rows[0]["created_at"] == "2024-01-01T00:00:01"
    assert rows[1]["created_at"] == "2024-01-01T00:00:02"
    assert set(rows[0]) == set(TradeRecordResponse.model_fields)

    buy_row, guard_row = rows
    assert buy_row["symbol"] == "BTC-USD"
    assert (buy_row["intent_action"], buy_row["intent_percentage"]) == ("BUY", 10.0)
    assert (buy_row["enforced_action"], buy_row["enforced_percentage"]) == ("BUY", 10.0)
    assert buy_row["execution_price"] == pytest.approx(100.8)              # 100 * (1 + 0.008)
    assert buy_row["quantity"] == pytest.approx(100.0 / 100.8)             # 10% of 1000 cash
    assert buy_row["fees_paid"] > 0
    assert buy_row["guard_note"] is None

    assert (guard_row["intent_action"], guard_row["enforced_action"],
            guard_row["enforced_percentage"], guard_row["quantity"],
            guard_row["fees_paid"]) == ("SELL", "HOLD", 0.0, 0.0, 0.0)
    assert guard_row["intent_percentage"] == 50.0
    assert guard_row["execution_price"] == 100.0
    assert guard_row["guard_note"] == "reject SELL: no position (no shorts in v1)"
    assert api.db.query(Trade).count() == 2                 # nothing else was written
