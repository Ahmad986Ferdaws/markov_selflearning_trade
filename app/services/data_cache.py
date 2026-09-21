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
import uuid
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


def atomic_write_text(path: Path | str, text: str) -> Path:
    """Write `text` to `path` atomically, with a plainly created file's mode.

    The temporary is created with os.open(..., 0o666) so the kernel applies
    the process umask at creation (0o644 under the usual 022) — not
    NamedTemporaryFile's owner-only 0o600, which survives the rename, and
    without touching the process-wide umask (an os.umask(0)/restore pair is
    a race for every other thread). The rename is atomic; a failed write
    leaves the destination untouched.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(64):
        temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
            break
        except FileExistsError:
            continue
    else:  # pragma: no cover - 64 uuid collisions
        raise OSError(f"could not create a temporary file next to {path}")
    try:
        try:
            f = os.fdopen(fd, "w", encoding="utf-8")
        except BaseException:
            os.close(fd)
            raise
        with f:
            f.write(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _legacy_record_path(results_dir: Path, report) -> Path | None:
    """The pre-content-addressing name `{symbol}_{data_hash[:8]}.json`, if it
    stays inside results_dir (the old naming did not sanitize the symbol)."""
    data_hash = getattr(report, "data_hash", "") or "nohash"
    candidate = results_dir / f"{report.symbol.replace('/', '_')}_{data_hash[:8]}.json"
    try:
        # a symlinked legacy file pointing outside is not ours to touch; a NUL
        # byte or an over-long name raises here (ValueError / ENAMETOOLONG) and
        # simply means "no legacy record"
        if candidate.resolve().parent != results_dir.resolve() or not candidate.exists():
            return None
    except (OSError, ValueError):
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
    if legacy is not None and legacy != path:
        try:
            # strict: the legacy JSON re-serialised canonically must equal the
            # new payload byte for byte (so 1 vs 1.0 or true vs 1 do not match)
            canonical = json.dumps(json.loads(legacy.read_text(encoding="utf-8")),
                                   indent=2, sort_keys=True, allow_nan=False)
            same = canonical == payload
        except (OSError, ValueError):
            same = False   # unreadable or not JSON: leave it alone, write the new record
        if same:
            legacy.touch()
            return legacy
    return atomic_write_text(path, payload)
