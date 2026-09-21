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


def file_mode_from_umask() -> int:
    """The mode a plainly created file would get (0o666 masked by the umask).

    NamedTemporaryFile creates its file 0o600; renaming it into place keeps
    that mode, so records and caches came out owner-only while every other
    committed file is 0o644. Callers chmod the temporary to this before the
    rename.
    """
    umask = os.umask(0)
    os.umask(umask)
    return 0o666 & ~umask


def _legacy_record_path(results_dir: Path, report) -> Path | None:
    """The pre-content-addressing name `{symbol}_{data_hash[:8]}.json`, if it
    stays inside results_dir (the old naming did not sanitize the symbol)."""
    data_hash = getattr(report, "data_hash", "") or "nohash"
    candidate = results_dir / f"{report.symbol.replace('/', '_')}_{data_hash[:8]}.json"
    if candidate.resolve().parent != results_dir.resolve():
        return None
    return candidate


def save_run_record(report, results_dir: Path | str = RESULTS_DIR) -> Path:
    """Persist by complete report content, preserving distinct runs on the same data.

    Existing historical data-hash filenames are never overwritten. Identical
    reports reuse one path; changed metrics/policies produce separate evidence.
    A legacy `{symbol}_{data_hash[:8]}.json` record whose JSON is value-identical
    to the report is reused (touched) rather than duplicated under the new
    name — so `make eval` on a fresh clone no longer dirties results/ with a
    second copy of the committed receipt.
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
    legacy = _legacy_record_path(results_dir, report)
    if legacy is not None and legacy != path and legacy.exists():
        try:
            same = json.loads(legacy.read_text(encoding="utf-8")) == json.loads(payload)
        except (OSError, ValueError):
            same = False   # unreadable or not JSON: leave it alone, write the new record
        if same:
            legacy.touch()
            return legacy
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=results_dir,
                                         prefix=".record-", delete=False) as f:
            temporary = Path(f.name)
            f.write(payload)
        os.chmod(temporary, file_mode_from_umask())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
