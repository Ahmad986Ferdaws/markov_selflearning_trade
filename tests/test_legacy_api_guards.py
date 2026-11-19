"""Runtime enforcement: the legacy web surface can NEVER spend the API key.

Review found the old tripwire grepped for 'agent_policy' while the web routes
reach app/services/agent.py transitively (runs -> runner, comparison). These
tests exercise the actual HTTP surface: with the default settings
(allow_legacy_agent_api=False) an agent run is refused at creation, and the
comparison replay completes WITHOUT ever calling the live-agent function —
proven by monkeypatching it to explode.
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


def _client(db, settings: Settings) -> TestClient:
    from app.dependencies import get_db_dep, get_settings_dep
    from app.main import app

    app.dependency_overrides[get_db_dep] = lambda: db
    app.dependency_overrides[get_settings_dep] = lambda: settings
    return TestClient(app)


def test_post_runs_refuses_agent_strategy_by_default():
    client = _client(_db(), Settings(anthropic_api_key="sk-would-be-spent"))
    resp = client.post("/runs", json={"strategy": "agent", "ingestion_provider": "yfinance",
                                      "watchlist": ["BTC-USD"]})
    assert resp.status_code == 400
    assert "CLI-only" in resp.text


def test_post_runs_rejects_trade_symbol_outside_watchlist():
    client = _client(_db(), Settings())
    resp = client.post("/runs", json={"strategy": "baseline", "ingestion_provider": "yfinance",
                                      "watchlist": ["ETH-USD"], "trade_symbol": "BTC-USD"})
    assert resp.status_code == 422
    assert "not in the watchlist" in resp.text


def test_comparison_endpoint_never_calls_live_agent(monkeypatch):
    import pandas as pd

    from app.services import agent as agent_service
    from app.services import comparison as comparison_module

    def _boom(*a, **k):  # any call == the key was about to be spent
        raise AssertionError("live agent call reached from the web surface")

    monkeypatch.setattr(agent_service, "get_agent_decision_sync", _boom)
    monkeypatch.setattr(comparison_module.agent_service, "get_agent_decision_sync", _boom)

    # offline benchmark history so the endpoint needs no network
    hist = pd.read_pickle("data/snapshots/BTC-USD_3y.pkl")
    monkeypatch.setattr(comparison_module, "fetch_daily_history", lambda *a, **k: hist)

    db = _db()
    run = Run(id=1, strategy="agent", watchlist="BTC-USD", trade_symbol="BTC-USD",
              cash=1000.0, starting_cash=1000.0, status="stopped")
    db.add(run)
    db.add(Snapshot(run_id=1, symbol="BTC-USD", price=50_000.0, volume=1.0, source="test"))
    db.add(Snapshot(run_id=1, symbol="BTC-USD", price=50_500.0, volume=1.0, source="test"))
    db.commit()

    client = _client(db, Settings(anthropic_api_key="sk-would-be-spent"))
    resp = client.get("/runs/1/comparison")   # would raise via _boom if called
    assert resp.status_code == 200
    assert resp.json()["snapshot_count"] == 2
