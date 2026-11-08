"""YFinanceProvider.poll must not stall the event loop.

yfinance does blocking network I/O; `poll` hands that work to a worker thread
(`asyncio.to_thread(self._poll_sync, ...)`). Proven here by racing `poll`
against a coroutine that ticks every 10 ms while the fake ticker's `fast_info`
blocks with time.sleep for 0.3 s: the ticker keeps advancing only if the loop
is free. No network: yf.Ticker is monkeypatched.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import threading
import time

from app.services import ingestion
from app.services.ingestion import YFinanceProvider


class _FastInfo:
    def __init__(self, **attrs):
        self.__dict__.update(attrs)


def _install_sleepy_ticker(monkeypatch, seen_threads: list[str], delay: float) -> None:
    class _SleepyTicker:
        def __init__(self, symbol: str):
            self.symbol = symbol

        @property
        def fast_info(self) -> _FastInfo:
            seen_threads.append(threading.current_thread().name)
            time.sleep(delay)                        # a slow, blocking round trip
            return _FastInfo(last_price=123.5, last_volume=7.0)

        def history(self, **kwargs):
            raise AssertionError("fast_info carried a price; history() must not be reached")

    monkeypatch.setattr(ingestion.yf, "Ticker", _SleepyTicker)


async def test_poll_leaves_the_event_loop_free_while_yfinance_blocks(monkeypatch):
    _install_sleepy_ticker(monkeypatch, seen_threads=[], delay=0.3)
    ticks = 0

    async def tick_forever():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    background = asyncio.create_task(tick_forever())
    try:
        snaps = await YFinanceProvider().poll(["BTC-USD"])
    finally:
        background.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await background

    # ~30 ticks fit in a 0.3 s poll; a blocked loop would record 0.
    assert ticks > 5, f"the loop advanced only {ticks} ticks while poll was in flight"
    assert [(s.symbol, s.price, s.volume, s.source) for s in snaps] == [
        ("BTC-USD", 123.5, 7.0, "yfinance"),
    ]
    assert snaps[0].timestamp.tzinfo is not None


async def test_the_blocking_calls_run_on_a_worker_thread_not_the_loop_thread(monkeypatch):
    seen: list[str] = []
    _install_sleepy_ticker(monkeypatch, seen_threads=seen, delay=0.0)
    await YFinanceProvider().poll(["BTC-USD", "ETH-USD"])
    assert len(seen) == 2
    assert all(name != threading.current_thread().name for name in seen), seen


def test_poll_sync_is_the_plain_synchronous_helper_behind_poll(monkeypatch):
    _install_sleepy_ticker(monkeypatch, seen_threads=[], delay=0.0)
    assert not inspect.iscoroutinefunction(YFinanceProvider._poll_sync)
    assert inspect.iscoroutinefunction(YFinanceProvider.poll)
    [snap] = YFinanceProvider()._poll_sync(["BTC-USD"])
    assert (snap.symbol, snap.price, snap.volume) == ("BTC-USD", 123.5, 7.0)
