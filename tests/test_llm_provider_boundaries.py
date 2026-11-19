"""Pins for the provider layer in app/services/agent_policy.py: the health
probe fails soft, OllamaProvider normalises its host and posts the documented
/api/chat payload, AnthropicProvider builds its client lazily, and
make_provider refuses unknown names and a keyless anthropic with a warning.

Offline: httpx.post is monkeypatched with a recorder; no client is ever built.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import httpx

from app.services.agent_policy import (
    AnthropicProvider,
    LLMProvider,
    OllamaProvider,
    make_provider,
)

LOGGER = "app.services.agent_policy"


# --- health() -------------------------------------------------------------------------
class _Down(LLMProvider):
    id = "down"

    def complete(self, system: str, user: str) -> str:
        raise ConnectionError("refused")


class _Up(LLMProvider):
    id = "up"

    def complete(self, system: str, user: str) -> str:
        return '{"ok": 1}'


def test_health_is_false_and_warns_when_complete_raises(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _Down().health() is False
    assert "down provider health check failed: refused" in caplog.text


def test_health_is_true_when_complete_returns_text(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _Up().health() is True
    assert caplog.text == ""


# --- OllamaProvider ---------------------------------------------------------------------
def test_ollama_strips_the_trailing_slash_from_its_host():
    assert OllamaProvider("m", "http://h:1/").host == "http://h:1"
    assert OllamaProvider("m", "http://h:1").host == "http://h:1"


def test_ollama_posts_a_json_format_non_streaming_chat_at_the_configured_temperature(monkeypatch):
    calls: list[tuple[tuple, dict]] = []

    class _Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"message": {"content": "x"}}

    def recorder(*args, **kwargs):
        calls.append((args, kwargs))
        return _Response()

    monkeypatch.setattr(httpx, "post", recorder)
    provider = OllamaProvider("m", "http://h:1/", temperature=0.25)

    assert provider.complete("SYS", "USER") == "x"

    [(args, kwargs)] = calls
    assert args == ("http://h:1/api/chat",)
    assert kwargs["timeout"] == 60.0
    assert kwargs["json"] == {
        "model": "m",
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.25},
        "messages": [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "USER"},
        ],
    }


# --- AnthropicProvider -------------------------------------------------------------------
def test_anthropic_builds_its_client_lazily():
    provider = AnthropicProvider("m", "not-a-real-key", temperature=0.5)
    assert provider._client is None            # nothing built until the first complete()
    assert (provider.id, provider.model, provider.temperature) == ("anthropic", "m", 0.5)


# --- make_provider -------------------------------------------------------------------------
def test_make_provider_refuses_an_unknown_name_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert make_provider(SimpleNamespace(llm_provider="mystery-llm")) is None
    assert "unknown llm_provider='mystery-llm'; agent disabled" in caplog.text


def test_make_provider_refuses_anthropic_without_a_key_and_names_the_key(caplog):
    settings = SimpleNamespace(llm_provider="anthropic", anthropic_api_key="")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert make_provider(settings) is None
    assert "ANTHROPIC_API_KEY is empty; agent disabled" in caplog.text
    assert "unknown" not in caplog.text
