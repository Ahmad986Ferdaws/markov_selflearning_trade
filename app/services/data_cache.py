"""Reproducibility: pin the input data, hash it, persist run records.

yfinance is revisable and a rolling "3y" window shifts every day, so two runs on
"the same" symbol can differ. Snapshotting the pulled history (and hashing it)
makes a run byte-reproducible and lets a result name the exact data it came from.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from app.services.regime import fetch_daily_history

CACHE_DIR = Path("data/snapshots")
RESULTS_DIR = Path("results")


def history_hash(history: pd.DataFrame) -> str:
    """Stable sha256 of the close series (values + dates), float-noise resistant."""
    closes = history["Close"] if "Close" in history.columns else history.squeeze()
    closes = closes.round(8)
    blob = "|".join(f"{d}:{v}" for d, v in zip(closes.index.astype(str), closes.astype(str)))
    return hashlib.sha256(blob.encode()).hexdigest()


def load_or_fetch(
    symbol: str,
    years: int = 3,
    refresh: bool = False,
    cache_dir: Path | str = CACHE_DIR,
) -> tuple[pd.DataFrame, str]:
    """Return (history, source). Uses a local snapshot unless refresh=True.

    Pin once, reuse forever — so the "official" result is reproducible. Pass
    refresh=True (or delete the snapshot) to re-pull fresh data.
    """
    cache_dir = Path(cache_dir)
    path = cache_dir / f"{symbol.replace('/', '_')}_{years}y.pkl"
    if path.exists() and not refresh:
        return pd.read_pickle(path), f"cache:{path}"
    history = fetch_daily_history(symbol, years=years)
    cache_dir.mkdir(parents=True, exist_ok=True)
    history.to_pickle(path)
    return history, f"fetched:{symbol} ({years}y) -> {path}"


def save_run_record(report, results_dir: Path | str = RESULTS_DIR) -> Path:
    """Persist the full report keyed by data hash so re-runs are idempotent."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    data_hash = getattr(report, "data_hash", "") or "nohash"
    path = results_dir / f"{report.symbol.replace('/', '_')}_{data_hash[:8]}.json"
    path.write_text(json.dumps(asdict(report), indent=2, default=str))
    return path
