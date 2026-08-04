"""Data layer: validation, adapters, and seeded synthetic generators.

Validation is fail-loud: unsorted, duplicated, non-positive, or misaligned
price data is REJECTED before it can reach a filter. Missing interior values
are surfaced explicitly (the filter handles them as predict-only steps) —
never forward/backward filled here, and backward filling is banned everywhere.

Real data enters through `load_pair_from_snapshots` (this repo's pinned
yfinance pickles — reproducible offline) or `load_pair_csv` (documented
adapter: two CSV/Parquet files with a datetime index and Open/Close columns).

Synthetic generators are SEEDED and return their ground truth so tests can
verify recovery instead of eyeballing it. Synthetic results are never
presented as empirical performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLS = ("Open", "Close")


@dataclass
class PairData:
    """Validated, aligned pair. close/open are level prices; the model layer
    decides level vs log representation explicitly."""

    p1: pd.DataFrame     # columns Open, Close — instrument 1 (the y series)
    p2: pd.DataFrame     # columns Open, Close — instrument 2 (the hedge)
    name1: str
    name2: str

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.p1.index

    def __len__(self) -> int:
        return len(self.p1)


def _validate_frame(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{name}: index must be a DatetimeIndex")
    if df.index.has_duplicates:
        raise ValueError(f"{name}: duplicate timestamps")
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{name}: timestamps not sorted ascending")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    sub = df[list(REQUIRED_COLS)].astype(float)
    if (sub.dropna() <= 0).any().any():
        raise ValueError(f"{name}: non-positive prices present")
    return sub


def align_pair(p1: pd.DataFrame, p2: pd.DataFrame, name1: str, name2: str,
               min_overlap: int = 250) -> PairData:
    """Validate both legs and inner-join on timestamps. Interior NaNs survive
    (explicitly handled downstream); rows missing from one leg entirely are
    dropped by the join — that is calendar alignment, not imputation."""
    a = _validate_frame(p1, name1)
    b = _validate_frame(p2, name2)
    idx = a.index.intersection(b.index)
    if len(idx) < min_overlap:
        raise ValueError(f"pair overlap too short: {len(idx)} < {min_overlap}")
    return PairData(a.loc[idx], b.loc[idx], name1, name2)


def load_pair_from_snapshots(sym1: str, sym2: str, years: int = 20,
                             snap_dir: str | Path = "data/snapshots") -> PairData:
    """Load a pair from this repo's pinned snapshots (offline, reproducible)."""
    d = Path(snap_dir)
    f1, f2 = d / f"{sym1}_{years}y.pkl", d / f"{sym2}_{years}y.pkl"
    for f in (f1, f2):
        if not f.exists():
            raise FileNotFoundError(f"pinned snapshot missing: {f}")
    return align_pair(pd.read_pickle(f1), pd.read_pickle(f2), sym1, sym2)


def load_pair_csv(path1: str | Path, path2: str | Path,
                  name1: str = "P1", name2: str = "P2") -> PairData:
    """CSV/Parquet adapter. Files need a datetime index column named 'Date'
    (or the first column) and Open/Close columns."""
    def _read(p: str | Path) -> pd.DataFrame:
        p = Path(p)
        df = pd.read_parquet(p) if p.suffix in (".parquet", ".pq") else pd.read_csv(p)
        if not isinstance(df.index, pd.DatetimeIndex):
            date_col = "Date" if "Date" in df.columns else df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
        return df
    return align_pair(_read(path1), _read(path2), name1, name2)


# --------------------------------------------------------------------------- #
# Seeded synthetic generators (ground truth returned for tests)
# --------------------------------------------------------------------------- #
def synthetic_pair(n: int = 1500, seed: int = 7,
                   beta0: float = 1.5, beta_drift_sd: float = 0.002,
                   alpha0: float = 0.5, alpha_drift_sd: float = 0.0005,
                   obs_noise_sd: float = 0.08) -> tuple[PairData, pd.DataFrame]:
    """Cointegrated-by-construction pair with a KNOWN drifting beta.

    p2 is a positive random walk (log space); p1 = beta_t*p2 + alpha_t + noise
    in LEVEL space. p2's volatility is deliberately non-trivial: when p2 barely
    moves, [p2, 1] is nearly collinear and alpha/beta are weakly identified —
    a real Kalman limitation the recovery test would otherwise mislabel a bug.
    Returns (PairData, truth) with truth columns beta/alpha.
    """
    rng = np.random.default_rng(seed)
    lp2 = np.cumsum(rng.normal(0.0003, 0.02, n)) + np.log(50.0)
    p2 = np.exp(lp2)
    beta = beta0 + np.cumsum(rng.normal(0.0, beta_drift_sd, n))
    alpha = alpha0 + np.cumsum(rng.normal(0.0, alpha_drift_sd, n))
    p1 = beta * p2 + alpha + rng.normal(0.0, obs_noise_sd, n)
    p1 = np.maximum(p1, 1e-3)
    idx = pd.bdate_range("2015-01-01", periods=n)
    mk = lambda px: pd.DataFrame(
        {"Open": px * (1 + rng.normal(0, 0.0005, n)), "Close": px}, index=idx)
    pair = align_pair(mk(p1), mk(p2), "SYN1", "SYN2", min_overlap=10)
    truth = pd.DataFrame({"beta": beta, "alpha": alpha}, index=idx)
    return pair, truth


def synthetic_constant_pair(n: int = 800, seed: int = 11,
                            beta: float = 2.0, alpha: float = 1.0,
                            obs_noise_sd: float = 0.1) -> tuple[PairData, pd.DataFrame]:
    """Constant-state pair: the filter must converge and stay stable."""
    return synthetic_pair(n=n, seed=seed, beta0=beta, beta_drift_sd=1e-12,
                          alpha0=alpha, alpha_drift_sd=1e-12,
                          obs_noise_sd=obs_noise_sd)


def synthetic_trend(n: int = 600, seed: int = 5, level0: float = 100.0,
                    vel0: float = 0.05, vel_sd: float = 0.01,
                    obs_sd: float = 0.5) -> tuple[pd.Series, pd.DataFrame]:
    """Local-linear-trend series with known level/velocity paths."""
    rng = np.random.default_rng(seed)
    vel = vel0 + np.cumsum(rng.normal(0, vel_sd, n))
    level = level0 + np.cumsum(vel)
    y = level + rng.normal(0, obs_sd, n)
    idx = pd.bdate_range("2018-01-01", periods=n)
    return (pd.Series(y, index=idx, name="y"),
            pd.DataFrame({"level": level, "velocity": vel}, index=idx))
