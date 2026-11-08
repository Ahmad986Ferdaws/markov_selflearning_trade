"""Every key in .env.example must be a field of app.config.Settings, so a
renamed or removed setting cannot leave a stale template behind; each key
appears once; and the template ships an empty ANTHROPIC_API_KEY (the SECURITY
policy: keys live only in a local, git-ignored .env). The file is read relative
to app.__file__ like tests/test_scope_tripwires.py does.
"""

from __future__ import annotations

from pathlib import Path

import app
from app.config import Settings

ROOT = Path(app.__file__).resolve().parent.parent
ENV_EXAMPLE = ROOT / ".env.example"


def _template_pairs() -> list[tuple[str, str]]:
    """(KEY, value) for every non-blank, non-comment line, split on the first '='."""
    pairs: list[tuple[str, str]] = []
    for raw in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        assert "=" in line, f".env.example line is not KEY=value: {raw!r}"
        key, _, value = line.partition("=")
        pairs.append((key.strip(), value.strip()))
    return pairs


def test_the_template_parser_sees_keys_values_and_urls_with_equals_free_split():
    pairs = _template_pairs()
    assert pairs, ".env.example has no KEY=value lines?"
    assert ("LLM_PROVIDER", "none") in pairs
    assert ("OLLAMA_HOST", "http://localhost:11434") in pairs   # value kept whole


def test_every_template_key_is_a_settings_field():
    fields = set(Settings.model_fields)
    stale = [key for key, _ in _template_pairs() if key.lower() not in fields]
    assert stale == [], f".env.example names settings that do not exist: {stale}"


def test_every_template_key_appears_once():
    keys = [key for key, _ in _template_pairs()]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    assert duplicates == [], f"duplicate keys in .env.example: {duplicates}"


def test_the_template_ships_an_empty_anthropic_api_key():
    values = dict(_template_pairs())
    assert "ANTHROPIC_API_KEY" in values
    assert values["ANTHROPIC_API_KEY"] == ""
