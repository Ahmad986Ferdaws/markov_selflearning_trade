"""The public web surface, inventoried.

Pins on app/main.py and the routes it mounts: /health answers, an ApiError
becomes a JSON ``{"detail": ...}`` carrying its status code, and the OpenAPI
inventory lists exactly the legacy /runs endpoints plus /health. Nothing on
this surface may evaluate, pull data, or reach an LLM provider; the last test
is the tripwire that notices if an endpoint is ever added.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.models.base import Base


def _db():
    engine = create_engine("sqlite:///:memory:",
                           connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _client() -> TestClient:
    from app.dependencies import get_db_dep, get_settings_dep
    from app.main import app

    db = _db()
    app.dependency_overrides[get_db_dep] = lambda: db
    app.dependency_overrides[get_settings_dep] = lambda: Settings()
    return TestClient(app)


def test_health_answers_ok_as_json():
    resp = _client().get("/health")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.json() == {"status": "ok"}


@pytest.mark.parametrize("method, path", [
    ("GET", "/runs/999"),
    ("GET", "/runs/999/trades"),
    ("POST", "/runs/999/stop"),
    ("GET", "/runs/999/comparison"),
])
def test_unknown_run_is_a_json_404_not_a_server_error(method, path):
    resp = _client().request(method, path)
    assert resp.status_code == 404
    assert resp.json() == {"detail": "Run 999 not found"}


def test_dexscreener_provider_is_refused_with_a_400():
    resp = _client().post("/runs", json={"strategy": "baseline",
                                        "ingestion_provider": "dexscreener",
                                        "watchlist": ["BTC-USD"]})
    assert resp.status_code == 400
    assert "deferred" in resp.json()["detail"]


def test_openapi_inventory_is_exactly_the_legacy_runs_surface_plus_health():
    """Tripwire: the web API must stay the quarantined legacy surface. Adding an
    endpoint that evaluates, refreshes data, or talks to a provider would make
    this fail, which is the point: such a route needs the same scrutiny the
    agent-policy invariants already enforce."""
    paths = _client().get("/openapi.json").json()["paths"]
    assert set(paths) == {
        "/health",
        "/runs",
        "/runs/{run_id}",
        "/runs/{run_id}/trades",
        "/runs/{run_id}/stop",
        "/runs/{run_id}/comparison",
    }
    forbidden = ("evaluate", "agent", "snapshot", "refresh", "provider")
    assert not [p for p in paths if any(w in p for w in forbidden)]
