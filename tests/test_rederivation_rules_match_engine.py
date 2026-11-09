"""scripts/independent_rederivation.py re-implements the data hash and the
label rule with zero app imports, so that a bug in the engine's evaluation
path would show up as a disagreement with the headline. Nothing proved the
two copies agree until now. These tests pin them against each other: the
script's snapshot_hash against app.services.data_cache.history_hash, and the
script's labels_from_prices against app.services.evaluation.causal_states, on
the pinned BTC-USD snapshot and on synthetic series.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import app
from app.services.data_cache import history_hash
from app.services.evaluation import causal_states
from scripts.independent_rederivation import K, WINDOW, labels_from_prices, snapshot_hash

ROOT = Path(app.__file__).resolve().parent.parent
PINNED = ROOT / "data" / "snapshots" / "BTC-USD_3y.pkl"


def _pinned_frame() -> pd.DataFrame:
    return pd.read_pickle(PINNED)


def _synthetic_frame(n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    steps = np.random.default_rng(11).normal(0.0, 0.01, n)
    return pd.DataFrame({"Close": 100.0 * np.exp(np.cumsum(steps)), "Volume": 1.0}, index=idx)


def _engine_returns(closes: pd.Series) -> pd.Series:
    return closes.pct_change(fill_method=None).iloc[1:]     # exactly as evaluation.evaluate does


# --- the hash --------------------------------------------------------------------
def test_script_hash_matches_the_engine_hash_on_the_pinned_frame():
    frame = _pinned_frame()
    assert snapshot_hash(frame["Close"]) == history_hash(frame)


def test_script_hash_matches_the_engine_hash_on_a_synthetic_frame():
    frame = _synthetic_frame()
    assert snapshot_hash(frame["Close"]) == history_hash(frame)
    nudged = frame.copy()
    nudged.loc[nudged.index[5], "Close"] += 1e-6
    assert snapshot_hash(nudged["Close"]) == history_hash(nudged) != history_hash(frame)


# --- the label rule --------------------------------------------------------------
def test_script_labels_match_the_engine_labels_on_the_pinned_frame():
    closes = _pinned_frame()["Close"]
    labels, n = labels_from_prices(closes)
    expected = causal_states(_engine_returns(closes), WINDOW, K)
    assert (WINDOW, K) == (30, 0.2)                 # the run record's chosen config
    assert len(labels) == len(expected) == n
    assert labels == expected                       # elementwise, warm-up Nones included
    assert set(labels) - {None} == {"bull", "bear", "sideways"}


def test_constant_prices_label_sideways_after_the_warmup():
    closes = pd.Series(np.full(100, 42.0), index=pd.date_range("2024-01-01", periods=100, freq="D"))
    labels, n = labels_from_prices(closes)
    # the warm-up is WINDOW price days, i.e. WINDOW - 1 daily returns: the
    # first label that is not None sits at index WINDOW - 1 of the return series
    assert labels[: WINDOW - 1] == [None] * (WINDOW - 1)
    assert labels[WINDOW - 1] == "sideways"
    assert set(labels[WINDOW - 1:]) == {"sideways"}
    assert n == len(closes) - 1 == len(labels)
    assert labels == causal_states(_engine_returns(closes), WINDOW, K)


def test_returned_count_is_one_less_than_the_price_series():
    for length in (31, 64, 500):
        labels, n = labels_from_prices(_synthetic_frame(length)["Close"])
        assert n == length - 1 == len(labels)
