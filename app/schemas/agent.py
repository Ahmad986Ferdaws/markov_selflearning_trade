from typing import Literal

from pydantic import BaseModel, Field


class AgentDecision(BaseModel):
    action: Literal["BUY", "SELL", "HOLD"]
    percentage: float = Field(ge=0, le=100)
    reasoning: str = ""
