from typing import Literal

from pydantic import BaseModel, Field


class AgentDecision(BaseModel):
    action: Literal["BUY", "SELL", "HOLD"]
    # default 0: a bare {"action": "HOLD"} is a VALID reply (the prompt invites
    # it); requiring percentage forced a second billed repair call for nothing
    percentage: float = Field(default=0.0, ge=0, le=100)
    reasoning: str = ""
