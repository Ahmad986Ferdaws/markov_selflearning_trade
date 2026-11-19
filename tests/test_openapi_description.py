"""The OpenAPI document says what this web app is (app/main.py).

create_app() used to set only a title and a version, so /docs gave a reader
no hint that this is the quarantined, simulation-only legacy surface, that
the LLM agent is CLI-only here, or that evaluate-cli is the current product.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def info() -> dict:
    from app.main import app

    return TestClient(app).get("/openapi.json").json()["info"]


def test_the_description_states_simulation_only_and_the_cli_only_agent(info):
    assert "simulation" in info["description"]
    assert "CLI-only" in info["description"]
    assert "ALLOW_LEGACY_AGENT_API" in info["description"]
    assert "evaluate-cli" in info["description"]


def test_the_summary_is_a_one_liner_saying_the_same(info):
    assert "simulation-only" in info["summary"]
    assert "legacy" in info["summary"]
    assert "\n" not in info["summary"]


def test_the_license_is_mit_and_links_to_the_repository_file(info):
    assert info["license"]["name"] == "MIT"
    assert info["license"]["url"].endswith("/markov_selflearning_trade/blob/main/LICENSE")


def test_the_title_is_unchanged(info):
    assert info["title"] == "Paper Trading Agent"
