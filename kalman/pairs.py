"""Pair qualification — TRAINING-DATA-ONLY diagnostics, never proof.

A Kalman filter does not create cointegration and cannot conjure mean
reversion; it only tracks a relationship that must already exist. These
checks describe the training window and are reported as diagnostics. Passing
them is not evidence of future profitability, and when many pairs are
screened the caller must apply multiple-testing discipline (report the count
screened; the CLI prints it).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from .data import PairData


@dataclass
class PairDiagnostics:
    n_train: int
    ols_beta: float
    ols_alpha: float
    eg_adf_stat: float          # ADF on the OLS residual (Engle-Granger step 2)
    eg_adf_pvalue: float
    half_life_days: float       # residual AR(1) half-life
    resid_autocorr_1: float
    beta_sign_stable: bool      # sign agreement across two half-window OLS fits
    beta_split_drift: float     # |beta_first_half - beta_second_half|
    overlap_ok: bool
    verdict: str                # descriptive only

    def summary(self) -> str:
        return (
            f"n={self.n_train}  beta={self.ols_beta:.3f}  "
            f"EG-ADF p={self.eg_adf_pvalue:.3f}  half-life={self.half_life_days:.1f}d  "
            f"rho1={self.resid_autocorr_1:.2f}  sign-stable={self.beta_sign_stable}  "
            f"split-drift={self.beta_split_drift:.3f}\n  -> {self.verdict}"
        )


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float, np.ndarray]:
    A = np.column_stack([x, np.ones(len(x))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0]), float(coef[1]), y - A @ coef


def qualify_pair(pair: PairData, train_end: int, use_log: bool = True,
                 min_overlap: int = 500) -> PairDiagnostics:
    """Diagnostics computed strictly on bars [0, train_end)."""
    c1 = pair.p1["Close"].to_numpy(float)[:train_end]
    c2 = pair.p2["Close"].to_numpy(float)[:train_end]
    mask = np.isfinite(c1) & np.isfinite(c2)
    c1, c2 = c1[mask], c2[mask]
    if use_log:
        c1, c2 = np.log(c1), np.log(c2)
    n = len(c1)

    beta, alpha, resid = _ols(c1, c2)
    adf_stat, adf_p, *_ = adfuller(resid, autolag="AIC")

    # AR(1) half-life of the residual
    r_lag, r_now = resid[:-1], resid[1:]
    phi = float(np.dot(r_lag - r_lag.mean(), r_now - r_now.mean())
                / max(np.dot(r_lag - r_lag.mean(), r_lag - r_lag.mean()), 1e-12))
    half_life = float(np.log(0.5) / np.log(abs(phi))) if 0 < abs(phi) < 1 else float("inf")

    rho1 = phi
    b1, _, _ = _ols(c1[: n // 2], c2[: n // 2])
    b2, _, _ = _ols(c1[n // 2:], c2[n // 2:])
    sign_stable = np.sign(b1) == np.sign(b2) != 0
    drift = abs(b1 - b2)

    notes = []
    if adf_p > 0.05:
        notes.append(f"EG-ADF p={adf_p:.2f}: no stationarity evidence at 5%")
    if not np.isfinite(half_life) or half_life > n / 4:
        notes.append("half-life long relative to window")
    if not sign_stable:
        notes.append("hedge-ratio SIGN flipped between halves")
    verdict = "; ".join(notes) if notes else "diagnostics unremarkable (NOT proof of edge)"
    return PairDiagnostics(n, beta, alpha, float(adf_stat), float(adf_p),
                           half_life, rho1, bool(sign_stable), drift,
                           n >= min_overlap, verdict)
