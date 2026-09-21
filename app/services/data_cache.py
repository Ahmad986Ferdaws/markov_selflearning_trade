"""Reproducibility: pin the input data, hash it, persist run records.

yfinance is revisable and a rolling "3y" window shifts every day, so two runs on
"the same" symbol can differ. Snapshotting the pulled history (and hashing it)
makes a run byte-reproducible and lets a result name the exact data it came from.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
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


def snapshot_path(symbol: str, years: int = 3, cache_dir: Path | str = CACHE_DIR) -> Path:
    """Where the pinned history for (symbol, years) lives — the one naming rule."""
    return Path(cache_dir) / f"{symbol.replace('/', '_')}_{years}y.pkl"


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
    path = snapshot_path(symbol, years, cache_dir)
    if path.exists() and not refresh:
        return pd.read_pickle(path), f"cache:{path}"
    history = fetch_daily_history(symbol, years=years)
    cache_dir.mkdir(parents=True, exist_ok=True)
    history.to_pickle(path)
    return history, f"fetched:{symbol} ({years}y) -> {path}"


def save_run_record(report, results_dir: Path | str = RESULTS_DIR) -> Path:
    """Persist by complete report content, preserving distinct runs on the same data.

    Existing historical data-hash filenames are never overwritten. Identical
    reports reuse one path; changed metrics/policies produce separate evidence.
    """
    payload = json.dumps(asdict(report), indent=2, sort_keys=True, allow_nan=False)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    symbol = re.sub(r"[^A-Za-z0-9_-]", "_", report.symbol)[:80] or "symbol"
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{symbol}_{digest}.json"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise ValueError(f"existing content-addressed record differs: {path}")
        # diagnose-cli selects the latest run by mtime, including replays.
        path.touch()
        return path
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=results_dir,
                                         prefix=".record-", delete=False) as f:
            temporary = Path(f.name)
            f.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
