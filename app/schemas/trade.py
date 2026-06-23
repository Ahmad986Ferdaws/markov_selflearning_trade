from typing import Literal

from pydantic import BaseModel, Field


class TradeIntent(BaseModel):
    action: Literal["BUY", "SELL", "HOLD"]
    percentage: float = Field(ge=0, le=100, default=0)
    symbol: str
    reasoning: str = ""
