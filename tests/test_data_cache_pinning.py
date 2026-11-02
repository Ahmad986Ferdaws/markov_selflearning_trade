"""Pins for app/services/data_cache.py: the hash that names a run's data and
the snapshot cache that makes a run reproducible offline.

No network. fetch_daily_history is monkeypatched to either explode (a pinned
snapshot must be served) or return a fixed frame (the pull must be pinned).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.services import data_cache
from app.services.data_cache import history_hash, load_or_fetch


def _frame(n: int = 5, start: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"Close": start + np.arange(n, dtype=float)}, index=idx)


# --- history_hash --------------------------------------------------------------
def test_hash_ignores_float_noise_below_the_rounding_precision():
    a = _frame()
    b = a.copy()
    b["Close"] += 1e-10
    assert history_hash(a) == history_hash(b)


def test_hash_changes_when_one_close_moves_by_a_millionth():
    a = _frame()
    b = a.copy()
    b.loc[b.index[2], "Close"] += 1e-6
    assert history_hash(a) != history_hash(b)


def test_hash_changes_when_a_date_moves_even_if_the_values_do_not():
    a = _frame()
    b = a.copy()
    b.index = b.index + pd.Timedelta(days=1)
    assert history_hash(a) != history_hash(b)


def test_hash_depends_only_on_the_close_column():
    a = _frame()
    b = a.copy()
    b["Volume"] = 1.0
    assert history_hash(a) == history_hash(b)


def test_a_single_column_frame_without_close_hashes_like_the_close_series():
    a = _frame()
    b = a.rename(columns={"Close": "Price"})
    assert history_hash(a) == history_hash(b)


# --- load_or_fetch: the pinned path ---------------------------------------------
def _explode(*args, **kwargs):
    raise AssertionError("network fetch attempted while a pinned snapshot exists")


def test_an_existing_snapshot_is_served_without_any_fetch(tmp_path, monkeypatch):
    monkeypatch.setattr(data_cache, "fetch_daily_history", _explode)
    frame = _frame()
    frame.to_pickle(tmp_path / "BTC-USD_3y.pkl")
    out, source = load_or_fetch("BTC-USD", years=3, cache_dir=tmp_path)
    pd.testing.assert_frame_equal(out, frame)
    assert source.startswith("cache:")


def test_the_years_argument_selects_a_distinct_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(data_cache, "fetch_daily_history", _explode)
    _frame(3).to_pickle(tmp_path / "BTC-USD_3y.pkl")
    _frame(12).to_pickle(tmp_path / "BTC-USD_12y.pkl")
    three, _ = load_or_fetch("BTC-USD", years=3, cache_dir=tmp_path)
    twelve, _ = load_or_fetch("BTC-USD", years=12, cache_dir=tmp_path)
    assert (len(three), len(twelve)) == (3, 12)
