"""Pins for the legacy runner's stop-flag and task-registry lifecycle
(app/services/runner.py): start_run_task registers a task and clears the stop
flag, a finished loop pops both registries, request_stop on an unknown id is
harmless, and an empty poll sleeps and loops instead of breaking out.

Offline: the session factory is an in-memory StaticPool database, the benchmark
fetch returns an empty DataFrame, the ingestion provider is a fake, and
poll_interval_seconds=0 keeps every loop fast.
"""

from __future__ import annotations

import asyncio

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.models.base import Base
from app.models.entities import Run, Snapshot, Trade
from app.services import runner

RUN_ID = 1


@pytest.fixture
def factory(monkeypatch):
    """Fresh registries, an in-memory DB behind get_session_factory and an
    offline benchmark fetch, for the duration of one test."""
    engine = create_engine("sqlite:///:memory:",
                           connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    monkeypatch.setattr(runner, "get_session_factory", lambda: session_factory)
    monkeypatch.setattr(runner, "fetch_daily_history", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(runner, "_active_tasks", {})
    monkeypatch.setattr(runner, "_stop_flags", {})
    return session_factory


def _add_run(factory, status: str) -> None:
    with factory() as db:
        db.add(Run(id=RUN_ID, status=status, watchlist="BTC-USD", trade_symbol="BTC-USD"))
        db.commit()


def _status(factory) -> str:
    with factory() as db:
        return db.get(Run, RUN_ID).status


def _settings() -> Settings:
    return Settings(poll_interval_seconds=0)


async def test_start_run_task_registers_the_task_and_clears_the_stop_flag(factory):
    _add_run(factory, "stopped")

    task = runner.start_run_task(RUN_ID, _settings())

    assert isinstance(task, asyncio.Task)
    assert runner._active_tasks[RUN_ID] is task
    assert runner._stop_flags[RUN_ID] is False
    await asyncio.wait_for(task, timeout=2.0)          # a stopped row exits promptly
    assert RUN_ID not in runner._active_tasks
    assert RUN_ID not in runner._stop_flags
    assert _status(factory) == "stopped"


def test_request_stop_on_an_unknown_run_only_sets_the_flag(factory):
    runner.request_stop(999)                            # no task to cancel, no raise
    assert runner._stop_flags[999] is True
    assert 999 not in runner._active_tasks


async def test_an_empty_poll_sleeps_and_loops_until_the_stop_flag_is_set(factory, monkeypatch):
    _add_run(factory, "running")
    polls: list[list[str]] = []

    class _EmptyProvider:
        async def poll(self, watchlist):
            polls.append(list(watchlist))
            if len(polls) == 2:          # the loop came back after the first empty poll
                runner._stop_flags[RUN_ID] = True
            return []

    monkeypatch.setattr(runner, "get_provider", lambda name: _EmptyProvider())
    runner._stop_flags[RUN_ID] = False

    await asyncio.wait_for(runner.run_loop(RUN_ID, _settings()), timeout=2.0)

    assert polls == [["BTC-USD"], ["BTC-USD"]]
    assert _status(factory) == "stopped"
    with factory() as db:                               # nothing was persisted or traded
        assert db.query(Snapshot).count() == 0
        assert db.query(Trade).count() == 0
    assert RUN_ID not in runner._stop_flags
    assert RUN_ID not in runner._active_tasks
