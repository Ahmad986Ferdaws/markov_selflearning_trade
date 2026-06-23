import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.models.base import Base
from app.models.entities import Run, Snapshot
from app.schemas.agent import AgentDecision
from app.services import comparison as comparison_module
from app.services.comparison import run_comparison


def _fake_history(*_args, **_kwargs) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=120, freq="D")
    closes = 100 * np.cumprod(1 + pd.Series(np.random.default_rng(0).normal(0, 0.01, 120), index=idx))
    return pd.DataFrame({"Close": closes})


def test_comparison_on_snapshots(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(comparison_module, "fetch_daily_history", _fake_history)
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    run = Run(id=1, watchlist="BTC-USD", trade_symbol="BTC-USD", cash=1000, status="stopped")
    db.add(run)
    for i, price in enumerate([100.0, 101.0, 102.0, 100.0]):
        db.add(
            Snapshot(
                run_id=1,
                symbol="BTC-USD",
                price=price,
                source="yfinance",
                payload_json=json.dumps({"symbol": "BTC-USD", "price": price}),
                created_at=datetime(2024, 1, 1 + i, tzinfo=timezone.utc),
            )
        )
    db.commit()
    settings = Settings(anthropic_api_key="")
    report = run_comparison(db, 1, settings)
    assert report.snapshot_count == 4
    assert report.baseline.name == "baseline"
    assert report.agent.name == "agent"
    db.close()


def test_comparison_replays_only_trade_symbol_snapshots(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(comparison_module, "fetch_daily_history", _fake_history)
    seen_symbols: list[str] = []

    def fake_agent_decision(metrics, *_args):
        seen_symbols.append(metrics["symbol"])
        return AgentDecision(action="HOLD", percentage=0, reasoning="test")

    monkeypatch.setattr(
        comparison_module.agent_service,
        "get_agent_decision_sync",
        fake_agent_decision,
    )
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    run = Run(
        id=1,
        watchlist="ETH-USD,BTC-USD",
        trade_symbol="BTC-USD",
        cash=1000,
        status="stopped",
    )
    db.add(run)
    rows = [
        ("ETH-USD", 2000.0),
        ("BTC-USD", 100.0),
        ("ETH-USD", 2010.0),
        ("BTC-USD", 101.0),
    ]
    for i, (symbol, price) in enumerate(rows):
        db.add(
            Snapshot(
                run_id=1,
                symbol=symbol,
                price=price,
                source="yfinance",
                payload_json=json.dumps({"symbol": symbol, "price": price}),
                created_at=datetime(2024, 1, 1 + i, tzinfo=timezone.utc),
            )
        )
    db.commit()

    report = run_comparison(db, 1, Settings(anthropic_api_key="test"))

    assert report.snapshot_count == 2
    assert seen_symbols == ["BTC-USD", "BTC-USD"]
    db.close()
