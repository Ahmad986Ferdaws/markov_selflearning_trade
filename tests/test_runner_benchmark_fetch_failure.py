"""A run whose benchmark download fails must stop, not run forever
(app/services/runner.py run_loop).

Before the fix the benchmark fetch sat outside any try: a raising
fetch_daily_history (yfinance down, or regime.py's "No data returned") or a
cancel that landed mid-download killed the coroutine before the finalisation,
so the Run row stayed status="running", _active_tasks kept the finished task and
_stop_flags was never popped. Both paths must now reach the finalisation.

Offline: the session factory is an in-memory StaticPool database, the fetch is
a fake that raises or sleeps on its worker thread, and poll_interval_seconds=0.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.models.base import Base
from app.models.entities import Run
from app.services import runner

RUN_ID = 1


@pytest.fixture
def factory(monkeypatch):
    """Fresh registries and an in-memory DB behind get_session_factory holding
    one running run; each test installs its own benchmark fetch."""
    engine = create_engine("sqlite:///:memory:",
                           connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    with session_factory() as db:
        db.add(Run(id=RUN_ID, status="running", watchlist="BTC-USD", trade_symbol="BTC-USD"))
        db.commit()
    monkeypatch.setattr(runner, "get_session_factory", lambda: session_factory)
    monkeypatch.setattr(runner, "_active_tasks", {})
    monkeypatch.setattr(runner, "_stop_flags", {})
    return session_factory


def _status(factory) -> str:
    with factory() as db:
        return db.get(Run, RUN_ID).status


def _settings() -> Settings:
    return Settings(poll_interval_seconds=0)


async def test_a_raising_benchmark_fetch_stops_the_run_and_logs_one_error(factory, monkeypatch, caplog):
    def dead_fetch(*args, **kwargs):
        raise RuntimeError("yfinance down")

    monkeypatch.setattr(runner, "fetch_daily_history", dead_fetch)
    caplog.set_level(logging.ERROR, logger="app.services.runner")

    await runner.run_loop(RUN_ID, _settings())              # returns, does not raise

    assert _status(factory) == "stopped"
    assert RUN_ID not in runner._active_tasks
    assert RUN_ID not in runner._stop_flags
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert f"run_id={RUN_ID}" in errors[0].getMessage()
    assert "stopping run" in errors[0].getMessage()
    assert "yfinance down" in caplog.text                   # the traceback travels with it


async def test_a_stop_requested_during_the_benchmark_fetch_still_finalises(factory, monkeypatch):
    fetch_started = threading.Event()

    def slow_fetch(*args, **kwargs):
        fetch_started.set()
        time.sleep(0.2)
        return pd.DataFrame()

    monkeypatch.setattr(runner, "fetch_daily_history", slow_fetch)

    task = runner.start_run_task(RUN_ID, _settings())
    # request_stop before the task has run at all cancels an unstarted
    # coroutine, whose body never executes; the issue is a cancel that lands
    # on the in-flight download, so yield until the worker thread is inside it.
    for _ in range(200):
        if fetch_started.is_set():
            break
        await asyncio.sleep(0.005)
    assert fetch_started.is_set()
    runner.request_stop(RUN_ID)

    await asyncio.wait_for(task, 2.0)                      # completes instead of hanging

    assert runner._active_tasks == {}
    assert runner._stop_flags == {}
    assert _status(factory) == "stopped"
