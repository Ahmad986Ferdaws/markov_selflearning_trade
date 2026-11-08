from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CreateRunRequest(BaseModel):
    # extra="forbid": a typo such as "stratgy" used to be ignored and quietly
    # created a baseline run instead of the one asked for.
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["baseline", "agent"] = "baseline"
    ingestion_provider: Literal["yfinance", "dexscreener"] = "yfinance"
    watchlist: list[str] = Field(default_factory=lambda: ["BTC-USD"])
    trade_symbol: str | None = None

    @field_validator("watchlist")
    @classmethod
    def _strip_and_reject_blank_or_duplicate_symbols(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for entry in value:
            symbol = entry.strip()
            if not symbol:
                raise ValueError("watchlist entries must not be blank")
            if symbol in cleaned:
                raise ValueError(f"watchlist entry {symbol!r} is duplicated")
            cleaned.append(symbol)
        return cleaned

    @field_validator("trade_symbol")
    @classmethod
    def _strip_trade_symbol(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None


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
