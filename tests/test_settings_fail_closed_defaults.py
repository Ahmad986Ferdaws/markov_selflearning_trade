"""Pins for the fail-closed defaults in app/config.py and how the environment
overrides them. Every Settings here is built with _env_file=None so a
developer's local .env cannot leak into the assertions, and the guard
variables are cleared from the process environment first.

These defaults are the executable form of README 'Safety & scope': the LLM
agent is CLI-only (allow_legacy_agent_api False, llm_provider 'none'), keys
come from the environment only (anthropic_api_key ''), and a run can never fan
out unbounded billed calls (agent_max_calls 64, agent_cache_only False).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import app
from app.config import Settings

ROOT = Path(app.__file__).resolve().parent.parent

GUARD_ENV_VARS = ("ALLOW_LEGACY_AGENT_API", "LLM_PROVIDER", "AGENT_CACHE_ONLY",
                  "AGENT_MAX_CALLS", "ANTHROPIC_API_KEY")


@pytest.fixture
def clean_env(monkeypatch):
    for name in GUARD_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _fresh() -> Settings:
    return Settings(_env_file=None)


def test_the_guards_fail_closed_by_default(clean_env):
    settings = _fresh()
    assert settings.allow_legacy_agent_api is False
    assert settings.llm_provider == "none"
    assert settings.agent_cache_only is False
    assert settings.agent_max_calls == 64
    assert settings.anthropic_api_key == ""


def test_readme_safety_and_scope_states_the_rules_these_defaults_enforce():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = re.search(r"^## Safety & scope\n(.*?)(?=^## )", readme, re.S | re.M)
    assert section, "README lost its 'Safety & scope' section"
    text = section.group(1)
    assert "Simulation only" in text
    assert "LLM agent is CLI-only" in text                 # allow_legacy_agent_api, llm_provider
    assert "Keys come from the environment only" in text   # anthropic_api_key
    documented = re.search(r"`AGENT_MAX_CALLS`, default (\d+)", readme)
    assert documented, "README no longer documents the AGENT_MAX_CALLS default"
    assert int(documented.group(1)) == Settings.model_fields["agent_max_calls"].default


def test_the_environment_overrides_the_guards(clean_env):
    clean_env.setenv("AGENT_MAX_CALLS", "3")
    clean_env.setenv("ALLOW_LEGACY_AGENT_API", "true")
    clean_env.setenv("LLM_PROVIDER", "ollama")
    settings = _fresh()
    assert (settings.agent_max_calls, settings.allow_legacy_agent_api, settings.llm_provider) == (
        3, True, "ollama",
    )


def test_an_unknown_environment_variable_is_ignored(clean_env):
    clean_env.setenv("REGIME_UNKNOWN_KNOB", "1")
    settings = _fresh()
    assert not hasattr(settings, "regime_unknown_knob")
    assert Settings.model_config["extra"] == "ignore"


def test_env_file_none_shuts_out_a_local_dot_env_that_the_default_constructor_reads(clean_env, tmp_path):
    clean_env.chdir(tmp_path)
    (tmp_path / ".env").write_text("ALLOW_LEGACY_AGENT_API=true\n", encoding="utf-8")
    assert Settings().allow_legacy_agent_api is True          # the default constructor honours .env
    assert Settings(_env_file=None).allow_legacy_agent_api is False
