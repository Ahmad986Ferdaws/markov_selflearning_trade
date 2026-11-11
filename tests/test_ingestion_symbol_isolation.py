"""One bad ticker in the watchlist must not cost the other symbols their quote.

YFinanceProvider._poll_sync (app/services/ingestion.py) used to let an
exception from one symbol's fast_info or history() escape the loop, so every
other symbol's snapshot in that poll was lost and run_loop retried the whole
watchlist 15 s later, forever. A failing symbol is now skipped with one WARNING
naming it; the silent skip of an empty intraday history is unchanged. No
network: yf.Ticker is monkeypatched to in-memory doubles.
"""

from __future__ import annotations

import logging

import pandas as pd

from app.services import ingestion
from app.services.ingestion import YFinanceProvider


class _FastInfo:
    def __init__(self, **attrs):
        self.__dict__.update(attrs)


class _FakeTicker:
    def __init__(self, fast_info: _FastInfo, history: pd.DataFrame | None = None):
        self.fast_info = fast_info
        self._history = history if history is not None else pd.DataFrame()

    def history(self, **kwargs) -> pd.DataFrame:
        return self._history


class _RaisingTicker:
    """A ticker whose fast_info lookup blows up, as yfinance does for a bad symbol."""

    @property
    def fast_info(self) -> _FastInfo:
        raise RuntimeError("no such ticker")

    def history(self, **kwargs) -> pd.DataFrame:
        raise AssertionError("fast_info raised; history() must not be reached")


def _install(monkeypatch, tickers: dict) -> None:
    monkeypatch.setattr(ingestion.yf, "Ticker", lambda symbol: tickers[symbol])


async def test_a_raising_symbol_is_skipped_and_the_others_still_report(monkeypatch, caplog):
    _install(monkeypatch, {
        "BAD-USD": _RaisingTicker(),
        "BTC-USD": _FakeTicker(_FastInfo(last_price=50_000.0)),
    })
    with caplog.at_level(logging.WARNING, logger="app.services.ingestion"):
        snaps = await YFinanceProvider().poll(["BAD-USD", "BTC-USD"])

    assert [(s.symbol, s.price, s.source) for s in snaps] == [("BTC-USD", 50_000.0, "yfinance")]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "BAD-USD" in warnings[0].getMessage()
    assert "no such ticker" in warnings[0].getMessage()


async def test_a_watchlist_of_only_bad_symbols_returns_nothing_instead_of_raising(monkeypatch, caplog):
    _install(monkeypatch, {"BAD-USD": _RaisingTicker()})
    with caplog.at_level(logging.WARNING, logger="app.services.ingestion"):
        snaps = await YFinanceProvider().poll(["BAD-USD"])
    assert snaps == []
    assert [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING] == [
        "yfinance lookup failed for BAD-USD: no such ticker",
    ]


async def test_an_empty_intraday_history_is_still_skipped_silently(monkeypatch, caplog):
    _install(monkeypatch, {"NO-BARS": _FakeTicker(_FastInfo(), history=pd.DataFrame())})
    with caplog.at_level(logging.WARNING, logger="app.services.ingestion"):
        snaps = await YFinanceProvider().poll(["NO-BARS"])
    assert snaps == []
    assert caplog.records == []                      # no warning: not an error, just no bar yet
