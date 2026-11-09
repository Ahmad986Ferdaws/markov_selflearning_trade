"""`qualify_pair` reports whether the hedge ratio keeps the same non-zero sign
in both halves of the training window; a flip is named in the verdict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kalman.data import align_pair, synthetic_pair
from kalman.pairs import qualify_pair


def _pair(p1: np.ndarray, p2: np.ndarray):
    idx = pd.bdate_range("2020-01-01", periods=len(p1))
    f1 = pd.DataFrame({"Open": p1, "Close": p1}, index=idx)
    f2 = pd.DataFrame({"Open": p2, "Close": p2}, index=idx)
    return align_pair(f1, f2, "A", "B", min_overlap=10)


def _hedge_leg(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.random.Generator]:
    rng = np.random.default_rng(seed)
    p2 = np.clip(50.0 + np.cumsum(rng.normal(0, 0.5, n)), 20.0, None)
    return p2, rng


def test_clearly_positive_beta_is_sign_stable():
    pair, _ = synthetic_pair(n=600, seed=3)             # beta0 = 1.5 by construction
    d = qualify_pair(pair, train_end=600, min_overlap=100)
    assert d.ols_beta > 0
    assert d.beta_sign_stable is True
    assert "SIGN flipped" not in d.verdict


def test_sign_flip_between_halves_is_flagged():
    p2, rng = _hedge_leg()
    n, half = len(p2), len(p2) // 2
    p1 = np.empty(n)
    p1[:half] = 2.0 * p2[:half] + rng.normal(0, 0.5, half)               # p1 = +2 * p2
    p1[half:] = 400.0 - 2.0 * p2[half:] + rng.normal(0, 0.5, n - half)   # p1 = -2 * p2 (+ offset: prices stay > 0)
    assert (p1 > 0).all()
    d = qualify_pair(_pair(p1, p2), train_end=n, use_log=False, min_overlap=10)
    assert d.beta_sign_stable is False
    assert "hedge-ratio SIGN flipped between halves" in d.verdict
    assert d.beta_split_drift > 3.0                      # |(+2) - (-2)| up to noise


def test_same_negative_sign_in_both_halves_is_stable():
    # the check is sign AGREEMENT, not positivity: a consistently inverse pair passes
    p2, rng = _hedge_leg(seed=1)
    p1 = 400.0 - 2.0 * p2 + rng.normal(0, 0.5, len(p2))
    assert (p1 > 0).all()
    d = qualify_pair(_pair(p1, p2), train_end=len(p2), use_log=False, min_overlap=10)
    assert d.ols_beta < 0
    assert d.beta_sign_stable is True
    assert "SIGN flipped" not in d.verdict
