"""CreateRunRequest (app/schemas/runs.py) refuses an empty watchlist and a blank trade_symbol.

An accepted ``watchlist=[]`` produced a Run that polled nothing and sat in
"running" forever; a whitespace ``trade_symbol`` was stripped to "" and then
treated by create_run as "not given", skipping the watchlist-membership check.
Both are now pydantic ValidationErrors, which the web layer turns into a 422.
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


def test_an_empty_watchlist_is_a_validation_error():
    with pytest.raises(ValidationError, match="watchlist"):
        CreateRunRequest(watchlist=[])


@pytest.mark.parametrize("trade_symbol", [" ", ""])
def test_a_blank_trade_symbol_is_a_validation_error(trade_symbol):
    with pytest.raises(ValidationError, match="trade_symbol must not be blank"):
        CreateRunRequest(trade_symbol=trade_symbol)


def test_an_absent_trade_symbol_and_a_one_symbol_watchlist_still_validate():
    body = CreateRunRequest(trade_symbol=None, watchlist=["BTC-USD"])
    assert body.trade_symbol is None
    assert body.watchlist == ["BTC-USD"]


def test_post_runs_with_a_blank_trade_symbol_is_a_422(monkeypatch):
    from app.api.routes import runs as runs_module
    from app.dependencies import get_db_dep, get_settings_dep
    from app.main import app

    started: list[int] = []
    monkeypatch.setattr(runs_module, "start_run_task",
                        lambda run_id, settings: started.append(run_id))
    engine = create_engine("sqlite:///:memory:",
                           connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    app.dependency_overrides[get_db_dep] = lambda: db
    app.dependency_overrides[get_settings_dep] = lambda: Settings()

    resp = TestClient(app).post("/runs", json={"trade_symbol": "  "})

    assert resp.status_code == 422
    assert "trade_symbol must not be blank" in resp.text
    assert started == []
