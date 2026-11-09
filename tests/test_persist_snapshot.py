"""What persist_snapshot (app/services/ledger.py) writes, pinned.

It is the one ledger entry point without a test. Every poll in run_loop goes
through it, so the row shape matters: the scalar columns mirror the
NormalizedSnapshot, payload_json is exactly truncated_metrics() (four keys,
timestamp as ISO-8601), an unreported volume is stored as NULL while a
reported 0.0 survives as 0.0 (the rule tests/test_ingestion_provider.py pins
upstream), and the row is committed, not merely flushed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.base import Base
from app.models.entities import Run, Snapshot
from app.schemas.snapshot import NormalizedSnapshot
from app.services.ledger import persist_snapshot

_WHEN = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


@pytest.fixture
def run(db):
    row = Run(cash=1000.0, starting_cash=1000.0, position_qty=0.0,
              trade_symbol="BTC-USD", watchlist="BTC-USD")
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _snap(volume: float | None) -> NormalizedSnapshot:
    return NormalizedSnapshot(symbol="BTC-USD", price=50_000.0, volume=volume,
                              timestamp=_WHEN, source="test")


def test_the_row_mirrors_the_snapshot_and_carries_a_server_set_created_at(db, run):
    snap = _snap(volume=None)
    row = persist_snapshot(db, run.id, snap)

    assert row.id is not None
    assert row.run_id == run.id
    assert (row.symbol, row.price, row.source) == ("BTC-USD", 50_000.0, "test")
    assert row.volume is None                                 # "not reported" is NULL
    assert isinstance(row.created_at, datetime)               # server_default=func.now()


def test_payload_json_is_exactly_truncated_metrics_with_an_iso_timestamp(db, run):
    snap = _snap(volume=None)
    row = persist_snapshot(db, run.id, snap)

    payload = json.loads(row.payload_json)
    assert payload == snap.truncated_metrics()
    assert set(payload) == {"symbol", "price", "volume", "timestamp"}
    assert payload["timestamp"] == "2026-01-02T03:04:05+00:00"
    assert payload["volume"] is None


def test_a_reported_zero_volume_is_stored_as_zero_not_null(db, run):
    row = persist_snapshot(db, run.id, _snap(volume=0.0))
    assert row.volume == 0.0
    assert row.volume is not None
    assert json.loads(row.payload_json)["volume"] == 0.0


def test_each_call_commits_one_row_that_survives_a_rollback(db, run):
    persist_snapshot(db, run.id, _snap(volume=None))
    db.rollback()                           # a flushed-but-uncommitted row would vanish here
    persist_snapshot(db, run.id, _snap(volume=0.0))
    db.rollback()
    assert db.query(Snapshot).count() == 2

    rows = db.query(Snapshot).order_by(Snapshot.id).all()     # fresh load after the rollbacks
    assert [r.volume for r in rows] == [None, 0.0]
    assert all(r.run_id == run.id for r in rows)
