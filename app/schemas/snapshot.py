from datetime import datetime

from pydantic import BaseModel, Field


class NormalizedSnapshot(BaseModel):
    symbol: str
    price: float
    volume: float | None = None
    timestamp: datetime
    source: str = "yfinance"

    def truncated_metrics(self) -> dict:
        return {
            "symbol": self.symbol,
            "price": self.price,
            "volume": self.volume,
            "timestamp": self.timestamp.isoformat(),
        }
