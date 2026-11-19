"""Pluggable market data ingestion."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone

import yfinance as yf

from app.schemas.snapshot import NormalizedSnapshot

logger = logging.getLogger(__name__)


class IngestionProvider(ABC):
    @abstractmethod
    async def poll(self, watchlist: list[str]) -> list[NormalizedSnapshot]:
        ...


class YFinanceProvider(IngestionProvider):
    """Poll latest quotes for watchlist symbols."""

    def _poll_sync(self, watchlist: list[str]) -> list[NormalizedSnapshot]:
        """Blocking yfinance round trips for every symbol; `poll` runs this in a
        worker thread so the event loop stays free for the other runs."""
        snapshots: list[NormalizedSnapshot] = []
        now = datetime.now(timezone.utc)
        for symbol in watchlist:
            # One bad ticker must not cost the other symbols their quote for
            # this poll (the runner would otherwise retry the whole watchlist
            # forever and never persist a snapshot).
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.fast_info
                price = getattr(info, "last_price", None)
                if price is None:                       # `or` would eat a real 0.0
                    price = getattr(info, "lastPrice", None)
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
                        # None means "not reported"; a real 0.0 (a halted or
                        # pre-open bar) must survive as 0.0, exactly as for price.
                        volume=float(vol) if vol is not None else None,
                        timestamp=now,
                        source="yfinance",
                    )
                )
            except Exception as exc:
                logger.warning("yfinance lookup failed for %s: %s", symbol, exc)
                continue
        return snapshots

    async def poll(self, watchlist: list[str]) -> list[NormalizedSnapshot]:
        return await asyncio.to_thread(self._poll_sync, watchlist)


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
