"""Pins for the non-LLM regime baseline (app/services/baseline.py).

The legacy intraday loop maps a RegimeFeature to a TradeIntent with three
rules. These tests sit on the p_bull >= 0.4 boundary and confirm that the
bear and sideways rules ignore the forecast entirely.
"""

from __future__ import annotations

import pytest

from app.services.baseline import baseline_decision
from app.services.regime import RegimeFeature


def _feature(state: str, p_bull: float) -> RegimeFeature:
    rest = (1.0 - p_bull) / 2
    return RegimeFeature(state=state, p_next={"bull": p_bull, "bear": rest, "sideways": rest})


def test_bull_with_the_forecast_exactly_at_the_threshold_buys_fifteen_percent():
    intent = baseline_decision(_feature("bull", 0.4), "BTC-USD")
    assert (intent.action, intent.percentage, intent.symbol) == ("BUY", 15.0, "BTC-USD")
    assert "p_bull=0.40" in intent.reasoning


def test_bull_with_the_forecast_just_under_the_threshold_holds():
    intent = baseline_decision(_feature("bull", 0.39), "BTC-USD")
    assert (intent.action, intent.percentage) == ("HOLD", 0.0)


def test_bull_with_no_bull_probability_reported_holds():
    feature = RegimeFeature(state="bull", p_next={"bear": 0.5, "sideways": 0.5})
    assert baseline_decision(feature, "BTC-USD").action == "HOLD"


@pytest.mark.parametrize("p_bull", [0.0, 0.4, 0.99])
def test_bear_sells_half_the_position_whatever_the_forecast_says(p_bull):
    intent = baseline_decision(_feature("bear", p_bull), "ETH-USD")
    assert (intent.action, intent.percentage, intent.symbol) == ("SELL", 50.0, "ETH-USD")
    assert "exit bias" in intent.reasoning
