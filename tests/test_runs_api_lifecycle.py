"""The legacy /runs lifecycle, end to end and offline: create, read, stop, trades.

The routes in app/api/routes/runs.py are exercised through the TestClient on
a shared in-memory SQLite database. start_run_task and request_stop are
replaced by recorders, so no polling task is started and yfinance is never
reached; the pins are what the HTTP layer writes and returns.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.models.base import Base
from app.models.entities import Run


def _db():
    # TestClient serves requests on a worker thread; share one in-memory DB
    engine = create_engine("sqlite:///:memory:",
                           connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


@pytest.fixture
def api(monkeypatch):
    from app.api.routes import runs as runs_module
    from app.dependencies import get_db_dep, get_settings_dep
    from app.main import app

    started: list[tuple[int, Settings]] = []
    stopped: list[int] = []
    monkeypatch.setattr(runs_module, "start_run_task",
                        lambda run_id, settings: started.append((run_id, settings)))
    monkeypatch.setattr(runs_module, "request_stop", lambda run_id: stopped.append(run_id))

    db = _db()
    settings = Settings()
    app.dependency_overrides[get_db_dep] = lambda: db
    app.dependency_overrides[get_settings_dep] = lambda: settings
    return SimpleNamespace(client=TestClient(app), db=db, settings=settings,
                           started=started, stopped=stopped)


def _create(api, **body):
    payload = {"strategy": "baseline", "ingestion_provider": "yfinance",
               "watchlist": ["BTC-USD", "ETH-USD"]}
    payload.update(body)
    return api.client.post("/runs", json=payload)


def test_creating_a_baseline_run_starts_it_once_with_the_settings_cash(api):
    resp = _create(api)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "running"
    assert body["strategy"] == "baseline"
    assert body["cash"] == api.settings.starting_cash
    assert body["trade_symbol"] == "BTC-USD"                  # watchlist[0] when none is given
    assert body["watchlist"] == ["BTC-USD", "ETH-USD"]
    assert (body["position_qty"], body["unrealized_pnl"], body["realized_pnl"]) == (0.0, 0.0, 0.0)
    assert [run_id for run_id, _ in api.started] == [body["id"]]  # the task was requested once
    assert api.started[0][1] is api.settings


def test_an_empty_watchlist_trades_the_benchmark_symbol(api):
    body = _create(api, watchlist=[]).json()
    assert body["watchlist"] == []
    assert body["trade_symbol"] == api.settings.benchmark_symbol


def test_reading_a_run_mirrors_the_stored_row(api):
    created = _create(api).json()
    row = api.db.get(Run, created["id"])
    assert row.starting_cash == api.settings.starting_cash

    body = api.client.get(f"/runs/{created['id']}").json()
    assert body == created                                     # no snapshot yet, so nothing to mark
    assert (body["id"], body["strategy"], body["ingestion_provider"], body["status"]) == (
        row.id, row.strategy, row.ingestion_provider, row.status)
    assert body["watchlist"] == row.watchlist.split(",")
    assert (body["trade_symbol"], body["cash"], body["position_qty"], body["realized_pnl"]) == (
        row.trade_symbol, row.cash, row.position_qty, row.realized_pnl)


def test_a_fresh_run_has_no_trades(api):
    created = _create(api).json()
    resp = api.client.get(f"/runs/{created['id']}/trades")
    assert resp.status_code == 200
    assert resp.json() == []


def test_stopping_a_run_marks_it_stopped_and_asks_the_runner_to_stop(api):
    created = _create(api).json()
    resp = api.client.post(f"/runs/{created['id']}/stop")
    assert resp.status_code == 200
    assert resp.json()["status"] == "stopped"
    assert api.stopped == [created["id"]]
    api.db.refresh(api.db.get(Run, created["id"]))
    assert api.db.get(Run, created["id"]).status == "stopped"
    assert api.client.get(f"/runs/{created['id']}").json()["status"] == "stopped"
