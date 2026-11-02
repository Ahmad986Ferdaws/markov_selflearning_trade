"""Pins for app/services/ingestion.py with yfinance replaced by an in-memory
double. No network: yf.Ticker is monkeypatched to a fake whose fast_info and
history() are set per test.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.services import ingestion
from app.services.ingestion import DexScreenerProvider, YFinanceProvider, get_provider


class _FastInfo:
    def __init__(self, **attrs):
        self.__dict__.update(attrs)


class _FakeTicker:
    def __init__(self, fast_info: _FastInfo, history: pd.DataFrame | None = None):
        self.fast_info = fast_info
        self._history = history if history is not None else pd.DataFrame()

    def history(self, **kwargs) -> pd.DataFrame:
        return self._history


def _install(monkeypatch, tickers: dict[str, _FakeTicker]) -> None:
    monkeypatch.setattr(ingestion.yf, "Ticker", lambda symbol: tickers[symbol])


async def test_a_reported_zero_volume_survives_as_zero(monkeypatch):
    _install(monkeypatch, {"BTC-USD": _FakeTicker(_FastInfo(last_price=100.0, last_volume=0.0))})
    [snap] = await YFinanceProvider().poll(["BTC-USD"])
    assert snap.volume == 0.0
    assert snap.volume is not None


async def test_an_unreported_volume_is_none(monkeypatch):
    _install(monkeypatch, {"BTC-USD": _FakeTicker(_FastInfo(last_price=100.0))})
    [snap] = await YFinanceProvider().poll(["BTC-USD"])
    assert snap.volume is None
