"""request_stop cancels a live run loop (app/services/runner.py).

The module docstring used to list an "inert request_stop cancel path" as a
known limit. It is not inert: request_stop cancels the registered task, the
loop body catches CancelledError at the provider poll and breaks out, and the
finalisation marks the row stopped and pops both registries. This pins the
cancel landing inside `provider.poll`, not on the sleep between polls.

Offline: in-memory StaticPool database behind get_session_factory, an empty
benchmark fetch, a provider that parks inside poll on an Event that is never
set, and poll_interval_seconds=0.
"""

from __future__ import annotations

import asyncio
import contextlib

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
    engine = create_engine("sqlite:///:memory:",
                           connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    with session_factory() as db:
        db.add(Run(id=RUN_ID, status="running", watchlist="BTC-USD", trade_symbol="BTC-USD"))
        db.commit()
    monkeypatch.setattr(runner, "get_session_factory", lambda: session_factory)
    monkeypatch.setattr(runner, "fetch_daily_history", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(runner, "_active_tasks", {})
    monkeypatch.setattr(runner, "_stop_flags", {})
    return session_factory


class _ParkedProvider:
    """poll signals that it is waiting, then awaits an Event nobody sets."""

    def __init__(self) -> None:
        self.waiting = asyncio.Event()
        self.polls: list[list[str]] = []
        self._never = asyncio.Event()

    async def poll(self, watchlist):
        self.polls.append(list(watchlist))
        self.waiting.set()
        await self._never.wait()
        return []                                           # unreachable: cancelled first


async def test_request_stop_cancels_a_loop_parked_inside_the_provider_poll(factory, monkeypatch):
    provider = _ParkedProvider()
    monkeypatch.setattr(runner, "get_provider", lambda name: provider)

    task = runner.start_run_task(RUN_ID, Settings(poll_interval_seconds=0))
    try:
        # run_loop first awaits the benchmark fetch in a worker thread, so wait
        # for poll to signal readiness on the clock, not by counting loop passes.
        await asyncio.wait_for(provider.waiting.wait(), 2.0)
    except TimeoutError:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        pytest.fail("the loop never reached provider.poll")
    assert not task.done()
    assert runner._active_tasks[RUN_ID] is task

    runner.request_stop(RUN_ID)

    await asyncio.wait_for(task, 2.0)                      # the cancel is not inert
    assert provider.polls == [["BTC-USD"]]                 # cancelled during the first poll
    with factory() as db:
        assert db.get(Run, RUN_ID).status == "stopped"
        assert db.query(Snapshot).count() == 0
        assert db.query(Trade).count() == 0
    assert RUN_ID not in runner._active_tasks
    assert RUN_ID not in runner._stop_flags
