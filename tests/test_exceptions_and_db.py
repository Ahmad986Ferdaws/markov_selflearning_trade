"""Pins for the two small infrastructure modules: app/exceptions.py and app/db.py."""

from __future__ import annotations

import json

import pytest
from sqlalchemy import inspect
from sqlalchemy.orm import Session

from app import db as db_module
from app.config import Settings
from app.exceptions import ApiError, api_error_handler


# --- ApiError ------------------------------------------------------------------------
def test_api_error_defaults_to_400_and_keeps_its_message():
    exc = ApiError("boom")
    assert (exc.message, exc.status_code) == ("boom", 400)
    assert str(exc) == "boom"
    assert exc.args == ("boom",)


async def test_api_error_handler_returns_the_status_and_a_detail_body():
    response = await api_error_handler(None, ApiError("not here", 418))
    assert response.status_code == 418
    assert json.loads(response.body) == {"detail": "not here"}
    assert response.media_type == "application/json"


# --- app.db ------------------------------------------------------------------------------
@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """Point app.db at a throwaway SQLite file and reset its module-level
    engine and session factory for the duration of one test."""
    url = f"sqlite:///{tmp_path / 'paper.db'}"
    monkeypatch.setattr(db_module, "get_settings", lambda: Settings(database_url=url))
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)
    return url


def test_get_engine_is_built_once_from_settings_with_sqlite_thread_args(fresh_db):
    engine = db_module.get_engine()
    assert db_module.get_engine() is engine                    # cached, not rebuilt
    assert str(engine.url) == fresh_db
    with engine.connect() as conn:
        assert conn.exec_driver_sql("select 1").scalar() == 1


def test_init_db_creates_every_mapped_table(fresh_db):
    db_module.init_db()
    tables = set(inspect(db_module.get_engine()).get_table_names())
    assert {"runs", "snapshots", "trades", "metrics"} <= tables


def test_get_db_yields_a_session_and_closes_it_afterwards(fresh_db):
    db_module.init_db()
    gen = db_module.get_db()
    session = next(gen)
    assert isinstance(session, Session)
    assert session.is_active
    with pytest.raises(StopIteration):
        next(gen)                                                # the finally: ran
    assert not session.in_transaction()
