from typing import Literal

from pydantic import BaseModel, Field


class CreateRunRequest(BaseModel):
    strategy: Literal["baseline", "agent"] = "baseline"
    ingestion_provider: Literal["yfinance", "dexscreener"] = "yfinance"
    watchlist: list[str] = Field(default_factory=lambda: ["BTC-USD"])
    trade_symbol: str | None = None


class RunStatusResponse(BaseModel):
    id: int
    strategy: str
    ingestion_provider: str
    status: str
    watchlist: list[str]
    trade_symbol: str
    cash: float
    position_qty: float
    unrealized_pnl: float
    realized_pnl: float


class TradeRecordResponse(BaseModel):
    id: int
    symbol: str
    intent_action: str
    intent_percentage: float
    enforced_action: str
    enforced_percentage: float
    quantity: float
    execution_price: float
    fees_paid: float
    guard_note: str | None
    created_at: str
