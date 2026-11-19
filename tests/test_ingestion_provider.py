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


async def test_snapshot_carries_source_symbol_and_an_aware_timestamp(monkeypatch):
    _install(monkeypatch, {"BTC-USD": _FakeTicker(_FastInfo(last_price=100.0, last_volume=5.0))})
    [snap] = await YFinanceProvider().poll(["BTC-USD"])
    assert (snap.symbol, snap.source, snap.price, snap.volume) == ("BTC-USD", "yfinance", 100.0, 5.0)
    assert snap.timestamp.tzinfo is not None


async def test_camel_case_last_price_is_the_first_fallback(monkeypatch):
    _install(monkeypatch, {"ETH-USD": _FakeTicker(_FastInfo(lastPrice=101.5))})
    [snap] = await YFinanceProvider().poll(["ETH-USD"])
    assert snap.price == 101.5


async def test_intraday_history_is_the_last_fallback_and_an_empty_one_skips_the_symbol(monkeypatch):
    bars = pd.DataFrame({"Close": [1.0, 2.0, 3.5]},
                        index=pd.date_range("2026-01-01", periods=3, freq="min"))
    _install(monkeypatch, {
        "HAS-BARS": _FakeTicker(_FastInfo(), history=bars),
        "NO-BARS": _FakeTicker(_FastInfo(), history=pd.DataFrame()),
    })
    snaps = await YFinanceProvider().poll(["HAS-BARS", "NO-BARS"])
    assert [(s.symbol, s.price) for s in snaps] == [("HAS-BARS", 3.5)]


def test_provider_factory_names_and_the_deferred_stub():
    assert isinstance(get_provider("yfinance"), YFinanceProvider)
    assert isinstance(get_provider("dexscreener"), DexScreenerProvider)
    with pytest.raises(ValueError, match="Unknown ingestion provider"):
        get_provider("binance")


async def test_dexscreener_stub_refuses_to_poll():
    with pytest.raises(NotImplementedError):
        await DexScreenerProvider().poll(["X"])
