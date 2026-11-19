"""CreateRunRequest (app/schemas/runs.py) refuses what it used to swallow.

An unknown key such as "stratgy" silently created a baseline run; a blank or
duplicated watchlist entry went straight into the Run row. Each is now a
pydantic ValidationError, which the web layer turns into a 422, and symbols
are stripped of surrounding whitespace on the way in.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.models.base import Base
from app.schemas.runs import CreateRunRequest


def test_an_unknown_field_is_rejected_instead_of_ignored():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CreateRunRequest(**{"stratgy": "agent"})


@pytest.mark.parametrize("watchlist", [[""], ["BTC-USD", " "]])
def test_a_blank_watchlist_entry_is_rejected(watchlist):
    with pytest.raises(ValidationError, match="must not be blank"):
        CreateRunRequest(watchlist=watchlist)


def test_a_duplicate_watchlist_entry_is_rejected_even_when_spelled_with_spaces():
    with pytest.raises(ValidationError, match="'BTC-USD' is duplicated"):
        CreateRunRequest(watchlist=["BTC-USD", " BTC-USD "])


def test_symbols_are_stripped_on_the_way_in():
    body = CreateRunRequest(watchlist=[" BTC-USD ", "ETH-USD "], trade_symbol=" BTC-USD ")
    assert body.watchlist == ["BTC-USD", "ETH-USD"]
    assert body.trade_symbol == "BTC-USD"


def test_defaults_are_unchanged_by_the_validators():
    body = CreateRunRequest()
    assert (body.strategy, body.ingestion_provider) == ("baseline", "yfinance")
    assert body.watchlist == ["BTC-USD"]
    assert body.trade_symbol is None


def test_post_runs_with_an_unknown_key_is_a_422():
    from app.dependencies import get_db_dep, get_settings_dep
    from app.main import app

    engine = create_engine("sqlite:///:memory:",
                           connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    app.dependency_overrides[get_db_dep] = lambda: db
    app.dependency_overrides[get_settings_dep] = lambda: Settings()

    resp = TestClient(app).post("/runs", json={"stratgy": "agent", "watchlist": ["BTC-USD"]})

    assert resp.status_code == 422
    assert "stratgy" in resp.text
