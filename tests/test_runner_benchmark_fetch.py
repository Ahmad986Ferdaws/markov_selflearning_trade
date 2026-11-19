"""run_loop must fetch the benchmark history off the event loop.

`fetch_daily_history` is a blocking yfinance download; the runner awaits it
through asyncio.to_thread so a slow download no longer stalls every other task
on the loop. Offline: the session factory is an in-memory StaticPool database
holding one already-stopped run (so the loop exits on its first iteration) and
the fetch is a recorder that returns an empty DataFrame.
"""

from __future__ import annotations

import threading

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.models.base import Base
from app.models.entities import Run
from app.schemas.trade import TradeIntent
from app.services import runner


def _factory_with_a_stopped_run():
    engine = create_engine("sqlite:///:memory:",
                           connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as db:
        db.add(Run(id=1, status="stopped", watchlist="BTC-USD", trade_symbol="BTC-USD"))
        db.commit()
    return factory


async def test_benchmark_history_is_fetched_on_a_worker_thread(monkeypatch):
    factory = _factory_with_a_stopped_run()
    monkeypatch.setattr(runner, "get_session_factory", lambda: factory)
    seen: list[tuple[str, tuple, dict]] = []

    def recorder(*args, **kwargs):
        seen.append((threading.current_thread().name, args, kwargs))
        return pd.DataFrame()

    monkeypatch.setattr(runner, "fetch_daily_history", recorder)
    loop_thread = threading.current_thread().name
    settings = Settings()

    await runner.run_loop(1, settings)

    assert len(seen) == 1
    thread_name, args, kwargs = seen[0]
    assert thread_name != loop_thread
    assert thread_name != threading.main_thread().name
    assert (args, kwargs) == ((settings.benchmark_symbol,), {"years": 3})
    with factory() as db:
        assert db.get(Run, 1).status == "stopped"
    assert 1 not in runner._active_tasks
    assert 1 not in runner._stop_flags


def test_trade_intent_is_imported_at_module_level_not_inside_the_agent_branch():
    assert runner.TradeIntent is TradeIntent
