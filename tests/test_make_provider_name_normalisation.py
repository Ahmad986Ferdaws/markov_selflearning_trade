"""make_provider must tolerate surrounding whitespace in LLM_PROVIDER.

A stray space in .env (LLM_PROVIDER=' ollama') used to be lowercased but not
stripped, so it was logged as an unknown provider and silently disabled the
agent. Offline: no provider is ever called, only constructed.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

from app.services.agent_policy import OllamaProvider, make_provider

LOGGER = "app.services.agent_policy"


def test_padded_mixed_case_ollama_still_selects_the_ollama_provider():
    provider = make_provider(SimpleNamespace(llm_provider=" Ollama "))
    assert isinstance(provider, OllamaProvider)


def test_padded_none_disables_the_agent_without_an_unknown_warning(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert make_provider(SimpleNamespace(llm_provider="none ")) is None
    assert "unknown" not in caplog.text


def test_an_unknown_name_is_still_refused_with_the_unknown_warning(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert make_provider(SimpleNamespace(llm_provider="binance")) is None
    assert "unknown llm_provider='binance'; agent disabled" in caplog.text
