"""GET /runs/{id}/comparison is described by a typed response model.

The route used to return an untyped dict built from dataclass __dict__s, so
the OpenAPI document said nothing about the payload. Now ComparisonResponse
and StrategyMetricsResponse appear in components.schemas and the JSON keys
are unchanged: seven top-level keys, ten per strategy leg. The benchmark
history comes from the pinned BTC pickle, so nothing touches the network.
"""

from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.models.base import Base
from app.models.entities import Run, Snapshot

TOP_LEVEL_KEYS = {
    "report_text", "headline_agent_minus_baseline", "baseline", "agent",
    "snapshot_count", "sharpe_periods_per_year", "warnings",
}
METRIC_KEYS = {
    "name", "total_return", "annual_sharpe", "max_drawdown", "win_rate",
    "num_trades", "total_costs", "guard_interventions", "num_closes", "regimes_seen",
}


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


def test_openapi_describes_the_comparison_payload_with_named_schemas():
    spec = _client(_db()).get("/openapi.json").json()
    schemas = spec["components"]["schemas"]
    assert set(schemas["ComparisonResponse"]["properties"]) == TOP_LEVEL_KEYS
    assert set(schemas["StrategyMetricsResponse"]["properties"]) == METRIC_KEYS
    response = spec["paths"]["/runs/{run_id}/comparison"]["get"]["responses"]["200"]
    assert response["content"]["application/json"]["schema"]["$ref"].endswith("/ComparisonResponse")


def test_comparison_json_keeps_exactly_the_legacy_keys(monkeypatch):
    from app.services import comparison as comparison_module

    # offline benchmark history so the endpoint needs no network
    hist = pd.read_pickle("data/snapshots/BTC-USD_3y.pkl")
    monkeypatch.setattr(comparison_module, "fetch_daily_history", lambda *a, **k: hist)

    db = _db()
    db.add(Run(id=1, strategy="baseline", watchlist="BTC-USD", trade_symbol="BTC-USD",
               cash=1000.0, starting_cash=1000.0, status="stopped"))
    db.add(Snapshot(run_id=1, symbol="BTC-USD", price=50_000.0, volume=1.0, source="test"))
    db.add(Snapshot(run_id=1, symbol="BTC-USD", price=50_500.0, volume=1.0, source="test"))
    db.commit()

    resp = _client(db).get("/runs/1/comparison")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == TOP_LEVEL_KEYS
    assert set(body["baseline"]) == METRIC_KEYS
    assert set(body["agent"]) == METRIC_KEYS
    assert (body["baseline"]["name"], body["agent"]["name"]) == ("baseline", "agent")
    assert body["snapshot_count"] == 2
    assert isinstance(body["warnings"], list)
    assert body["report_text"].startswith("=== Comparison Report")
