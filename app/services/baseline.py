"""Non-LLM regime baseline strategy."""

from app.schemas.trade import TradeIntent
from app.services.regime import RegimeFeature


def baseline_decision(feature: RegimeFeature, symbol: str) -> TradeIntent:
    """Map regime to BUY/SELL/HOLD with percentage."""
    p_bull = feature.p_next.get("bull", 0.0)
    state = feature.state

    if state == "bull" and p_bull >= 0.4:
        return TradeIntent(action="BUY", percentage=15.0, symbol=symbol, reasoning=f"regime={state} p_bull={p_bull:.2f}")
    if state == "bear":
        return TradeIntent(action="SELL", percentage=50.0, symbol=symbol, reasoning=f"regime={state} exit bias")
    return TradeIntent(action="HOLD", percentage=0.0, symbol=symbol, reasoning=f"regime={state} wait")
