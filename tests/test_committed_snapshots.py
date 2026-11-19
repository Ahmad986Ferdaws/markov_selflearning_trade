"""The committed snapshots are the only data the suite touches. docs/17 records
that a review once claimed the pickles were unreadable; pin the truth instead:
every one loads as a frame of sorted, positive daily closes, each long history
is longer than its three-year snapshot, and the headline snapshot hashes to
the digest that the run record and the re-derivation script both name.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import app
from app.services.data_cache import history_hash
from scripts.independent_rederivation import EXPECTED_HASH

ROOT = Path(app.__file__).resolve().parent.parent
SNAPSHOTS = ROOT / "data" / "snapshots"
PICKLES = sorted(SNAPSHOTS.glob("*.pkl"))


def test_twenty_eight_snapshots_are_committed():
    assert len(PICKLES) == 28


@pytest.mark.parametrize("path", PICKLES, ids=[p.name for p in PICKLES])
def test_snapshot_loads_as_sorted_positive_daily_closes(path: Path):
    frame = pd.read_pickle(path)
    assert isinstance(frame, pd.DataFrame)
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame.index.is_unique
    assert frame.index.is_monotonic_increasing
    assert len(frame) >= 250
    closes = frame["Close"]
    assert closes.dtype == np.float64
    assert np.isfinite(closes).all()
    assert (closes > 0).all()


def test_each_long_history_is_longer_than_its_three_year_snapshot():
    long_histories = [p for p in PICKLES if not p.stem.endswith("_3y")]
    assert len(long_histories) == 8
    for path in long_histories:
        symbol, span = path.stem.rsplit("_", 1)
        assert span in ("10y", "12y", "20y"), path.name
        short = SNAPSHOTS / f"{symbol}_3y.pkl"
        assert short.exists(), path.name
        assert len(pd.read_pickle(path)) > len(pd.read_pickle(short)), path.name


def test_the_headline_snapshot_hashes_to_the_recorded_digest():
    digest = history_hash(pd.read_pickle(SNAPSHOTS / "BTC-USD_3y.pkl"))
    record = json.loads((ROOT / "results" / "BTC-USD_4a150b23.json").read_text(encoding="utf-8"))
    assert digest == record["data_hash"] == EXPECTED_HASH
    assert digest.startswith("4a150b23")          # the record's filename names the data
