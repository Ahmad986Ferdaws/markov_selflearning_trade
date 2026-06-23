"""Pluggable market data ingestion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone

import yfinance as yf

from app.schemas.snapshot import NormalizedSnapshot


class IngestionProvider(ABC):
    @abstractmethod
    async def poll(self, watchlist: list[str]) -> list[NormalizedSnapshot]:
        ...


class YFinanceProvider(IngestionProvider):
    """Poll latest quotes for watchlist symbols."""

    async def poll(self, watchlist: list[str]) -> list[NormalizedSnapshot]:
        snapshots: list[NormalizedSnapshot] = []
        now = datetime.now(timezone.utc)
        for symbol in watchlist:
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            price = getattr(info, "last_price", None) or getattr(info, "lastPrice", None)
            if price is None:
                hist = ticker.history(period="1d", interval="1m")
                if hist.empty:
                    continue
                price = float(hist["Close"].iloc[-1])
            else:
                price = float(price)
            vol = getattr(info, "last_volume", None)
            snapshots.append(
                NormalizedSnapshot(
                    symbol=symbol,
                    price=price,
                    volume=float(vol) if vol else None,
                    timestamp=now,
                    source="yfinance",
                )
            )
        return snapshots


class DexScreenerProvider(IngestionProvider):
    """Stub for post-Phase-1 gate — not implemented in v1 first pass."""

    async def poll(self, watchlist: list[str]) -> list[NormalizedSnapshot]:
        raise NotImplementedError("DEX Screener provider deferred until after Phase 1 gate")


def get_provider(name: str) -> IngestionProvider:
    if name == "yfinance":
        return YFinanceProvider()
    if name == "dexscreener":
        return DexScreenerProvider()
    raise ValueError(f"Unknown ingestion provider: {name}")
