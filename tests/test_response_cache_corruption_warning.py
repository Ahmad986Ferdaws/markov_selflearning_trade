"""A corrupt on-disk agent response cache is still discarded fail-open, but no
longer silently: ResponseCache logs one WARNING naming the file, so a run whose
every decision became a fresh billed call can be traced to its cause.
Offline: temporary files only, no provider.
"""

from __future__ import annotations

import json
import logging

from app.services.agent_policy import ResponseCache

LOGGER = "app.services.agent_policy"


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_an_unparseable_cache_file_starts_empty_and_warns_once_with_its_path(tmp_path, caplog):
    path = tmp_path / "cache.json"
    path.write_text("not json")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        cache = ResponseCache(path)
    assert cache._d == {}
    assert not cache.contains("x")
    [message] = _warnings(caplog)
    assert str(path) in message
    assert "starting empty" in message


def test_a_cache_mapping_to_a_non_string_value_starts_empty_and_warns_once(tmp_path, caplog):
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({"k": 1}))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        cache = ResponseCache(path)
    assert cache._d == {}
    [message] = _warnings(caplog)
    assert str(path) in message
    assert "must map strings to strings" in message


def test_a_valid_cache_loads_without_any_warning(tmp_path, caplog):
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({"k": "v"}))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        cache = ResponseCache(path)
    assert cache._d == {"k": "v"}
    assert _warnings(caplog) == []


def test_a_missing_cache_file_is_not_corruption_and_does_not_warn(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        cache = ResponseCache(tmp_path / "absent.json")
    assert cache._d == {}
    assert _warnings(caplog) == []
