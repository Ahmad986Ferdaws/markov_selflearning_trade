"""The committed .env.example is the documented quick start (`cp .env.example
.env`). A blank value there must mean "use the default", not "set to empty" —
`ANTHROPIC_MODEL=` used to become anthropic_model == "" and the Anthropic
provider then failed its health check with model="" (reported as unreachable).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.services.agent_policy import make_provider

TEMPLATE = Path(".env.example")


def test_template_blank_model_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    s = Settings(_env_file=TEMPLATE)
    assert s.anthropic_model == Settings.model_fields["anthropic_model"].default
    assert s.anthropic_model  # never empty


def test_blank_numeric_value_in_env_file_keeps_default(tmp_path, monkeypatch):
    monkeypatch.delenv("POLL_INTERVAL_SECONDS", raising=False)
    env = tmp_path / ".env"
    env.write_text("POLL_INTERVAL_SECONDS=\nREGIME_WINDOW=\n")
    s = Settings(_env_file=env)   # used to raise: '' is not a valid integer
    assert (s.poll_interval_seconds, s.regime_window) == (15, 20)


def test_blank_process_env_var_is_ignored_too(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "")
    assert Settings(_env_file=None).anthropic_model


@pytest.mark.parametrize("name", ["ANTHROPIC_MODEL", "OLLAMA_MODEL"])
def test_provider_is_never_built_with_an_empty_model(monkeypatch, name):
    monkeypatch.setenv(name, "")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic" if name == "ANTHROPIC_MODEL" else "ollama")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-for-construction-only")
    provider = make_provider(Settings(_env_file=None))
    assert provider is not None and provider.model


def test_explicit_values_in_template_still_apply():
    s = Settings(_env_file=TEMPLATE)
    assert s.llm_provider == "none"
    assert s.allow_legacy_agent_api is False
