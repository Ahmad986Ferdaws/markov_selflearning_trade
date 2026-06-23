"""Claude agent — strict JSON contract."""

from __future__ import annotations

import json
import logging
import re

from app.config import Settings
from app.schemas.agent import AgentDecision
from app.services.regime import RegimeFeature

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a paper trading decision engine. Respond with raw JSON only — no markdown, no backticks, no prose.
Schema: {"action": "BUY"|"SELL"|"HOLD", "percentage": 0-100, "reasoning": "short data-backed note"}
Use HOLD when uncertain. percentage is fraction of cash (BUY) or position (SELL) to use."""


def _strip_fences(text: str) -> str:
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return m.group(0) if m else text


def parse_decision(raw: str) -> AgentDecision:
    cleaned = _strip_fences(raw)
    data = json.loads(cleaned)
    return AgentDecision.model_validate(data)


async def get_agent_decision(
    metrics: dict,
    feature: RegimeFeature,
    symbol: str,
    settings: Settings,
) -> AgentDecision:
    if not settings.anthropic_api_key:
        logger.warning("No ANTHROPIC_API_KEY; falling back to HOLD")
        return AgentDecision(action="HOLD", percentage=0, reasoning="missing API key")

    user_payload = {
        "symbol": symbol,
        "metrics": metrics,
        "regime": {"state": feature.state, "p_next": feature.p_next},
    }

    try:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        message = await client.messages.create(
            model=settings.anthropic_model,
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": json.dumps(user_payload)}],
        )
        raw = message.content[0].text
        try:
            return parse_decision(raw)
        except Exception:
            repair = await client.messages.create(
                model=settings.anthropic_model,
                max_tokens=256,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": json.dumps(user_payload)},
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": "Invalid JSON. Reply with valid JSON only matching the schema."},
                ],
            )
            return parse_decision(repair.content[0].text)
    except Exception as e:
        logger.exception("agent call failed: %s", e)
        return AgentDecision(action="HOLD", percentage=0, reasoning=f"agent error: {e}")


def get_agent_decision_sync(
    metrics: dict,
    feature: RegimeFeature,
    symbol: str,
    settings: Settings,
) -> AgentDecision:
    """Sync path for comparison replay (no running event loop)."""
    if not settings.anthropic_api_key:
        return AgentDecision(action="HOLD", percentage=0, reasoning="missing API key")

    user_payload = {
        "symbol": symbol,
        "metrics": metrics,
        "regime": {"state": feature.state, "p_next": feature.p_next},
    }
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        message = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": json.dumps(user_payload)}],
        )
        raw = message.content[0].text
        try:
            return parse_decision(raw)
        except Exception:
            repair = client.messages.create(
                model=settings.anthropic_model,
                max_tokens=256,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": json.dumps(user_payload)},
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": "Invalid JSON. Reply with valid JSON only matching the schema."},
                ],
            )
            return parse_decision(repair.content[0].text)
    except Exception as e:
        logger.exception("agent sync call failed: %s", e)
        return AgentDecision(action="HOLD", percentage=0, reasoning=f"agent error: {e}")
