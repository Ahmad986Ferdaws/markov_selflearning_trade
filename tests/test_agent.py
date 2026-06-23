import json

from app.schemas.agent import AgentDecision
from app.services.agent import parse_decision


def test_parse_decision_raw_json():
    raw = '{"action": "HOLD", "percentage": 0, "reasoning": "wait"}'
    d = parse_decision(raw)
    assert d.action == "HOLD"


def test_parse_decision_strips_fences():
    raw = '```json\n{"action": "BUY", "percentage": 10, "reasoning": "bull"}\n```'
    d = parse_decision(raw)
    assert d.action == "BUY"
    assert d.percentage == 10


def test_parse_invalid_raises():
    try:
        parse_decision("not json")
        raised = False
    except Exception:
        raised = True
    assert raised
