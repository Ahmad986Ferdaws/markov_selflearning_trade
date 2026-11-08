"""CreateRunRequest defaults and literal choices (app/schemas/runs.py), pinned.

These are the values a POST /runs with an empty body receives. The watchlist
default comes from a default_factory, so two requests must not share one
list object; the strategy and provider literals are the only accepted ones.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.runs import CreateRunRequest


def test_an_empty_request_defaults_to_a_baseline_yfinance_run_on_btc():
    body = CreateRunRequest()
    assert body.strategy == "baseline"
    assert body.ingestion_provider == "yfinance"
    assert body.watchlist == ["BTC-USD"]
    assert body.trade_symbol is None


def test_two_requests_do_not_share_the_default_watchlist_object():
    first, second = CreateRunRequest(), CreateRunRequest()
    assert first.watchlist == second.watchlist
    assert first.watchlist is not second.watchlist
    first.watchlist.append("ETH-USD")
    assert second.watchlist == ["BTC-USD"]


@pytest.mark.parametrize("strategy", ["baseline", "agent"])
def test_the_two_known_strategies_are_accepted(strategy):
    assert CreateRunRequest(strategy=strategy).strategy == strategy


def test_an_unknown_strategy_is_a_validation_error():
    with pytest.raises(ValidationError, match="'baseline' or 'agent'"):
        CreateRunRequest(strategy="momentum")


@pytest.mark.parametrize("provider", ["yfinance", "dexscreener"])
def test_the_two_known_providers_are_accepted(provider):
    assert CreateRunRequest(ingestion_provider=provider).ingestion_provider == provider


def test_an_unknown_provider_is_a_validation_error():
    with pytest.raises(ValidationError, match="'yfinance' or 'dexscreener'"):
        CreateRunRequest(ingestion_provider="binance")
