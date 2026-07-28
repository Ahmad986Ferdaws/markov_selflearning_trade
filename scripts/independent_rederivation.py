"""Independent re-derivation — ZERO app imports, pure pandas/numpy.

The replication of the headline numbers so far shares the engine code with the
studies it checks. This script closes that gap: it re-implements the claim
chain FROM THE CHOSEN CONFIG (window=30, k=0.2) ONWARD from the documented
rules alone — rolling z-score labels, MLE transition counts, causal
walk-forward argmax — and recomputes the BTC-USD headline numbers from the raw
pinned snapshot. If a bug lived in the engine's evaluation path, this file
would disagree with it. (Scope note: the train-only hyperparameter SELECTION
loop is not re-derived here; the identity claim this checks is conditional on
the recorded config.)

The snapshot is hash-verified against the run record before scoring, so a
refreshed pickle produces a distinct SNAPSHOT CHANGED verdict — never a
misleading MISMATCH.

Expected (from results/BTC-USD_4a150b23.json and docs/19):
  328 held-out predictions · 30 switch days · 0 switch attempts ·
  298 hits (90.85%) · model hit-rate == persistence hit-rate exactly.

Run:  python scripts/independent_rederivation.py   (exit 0 iff all reproduce)
"""

from __future__ import annotations

import hashlib
import sys

import numpy as np
import pandas as pd

WINDOW, K = 30, 0.2          # the chosen config from the run record
TRAIN_FRAC = 0.7
STATES = ("bull", "bear", "sideways")
IDX = {s: i for i, s in enumerate(STATES)}
# results/BTC-USD_4a150b23.json — the exact snapshot the headline was computed on
EXPECTED_HASH = "4a150b23b35efd8182d83afdd50817f5a7476155332b549f568e8b3620e7aaa8"


def snapshot_hash(closes: pd.Series) -> str:
    """Re-implemented (no app imports): sha256 over rounded closes + dates,
    matching app/services/data_cache.history_hash's documented recipe."""
    closes = closes.round(8)
    blob = "|".join(f"{d}:{v}" for d, v in zip(closes.index.astype(str), closes.astype(str)))
    return hashlib.sha256(blob.encode()).hexdigest()


def labels_from_prices(closes: pd.Series) -> list:
    """The documented rule, re-implemented: trailing 30d mean/std of daily
    returns; bull if mean > k*std, bear if mean < -k*std, else sideways."""
    rets = closes.pct_change().dropna()
    mean = rets.rolling(WINDOW, min_periods=WINDOW).mean()
    std = rets.rolling(WINDOW, min_periods=WINDOW).std()
    out = []
    for m, s in zip(mean, std):
        if np.isnan(m) or np.isnan(s):
            out.append(None)
        elif s <= 0:
            out.append("sideways")
        elif m > K * s:
            out.append("bull")
        elif m < -K * s:
            out.append("bear")
        else:
            out.append("sideways")
    return out, len(rets)


def main() -> int:
    hist = pd.read_pickle("data/snapshots/BTC-USD_3y.pkl")
    closes = hist["Close"] if "Close" in hist.columns else hist.squeeze()
    if snapshot_hash(closes) != EXPECTED_HASH:
        print("SNAPSHOT CHANGED — the pickle on disk is not the pinned data the "
              "headline was computed on; re-pin before re-deriving.")
        return 2
    states, n = labels_from_prices(closes)
    split = int(n * TRAIN_FRAC)

    preds = switch_days = attempts = hits = persist_hits = 0
    for t in range(split, n - 1):
        cur, nxt = states[t], states[t + 1]
        if cur is None or nxt is None:
            continue
        # MLE transition matrix from the compressed labeled sequence up to t
        obs = [s for s in states[: t + 1] if s is not None]
        if len(obs) < 2:
            continue
        counts = np.zeros((3, 3))
        for a, b in zip(obs, obs[1:]):
            counts[IDX[a], IDX[b]] += 1
        row = counts[IDX[cur]]
        dist = row / row.sum() if row.sum() > 0 else np.ones(3) / 3
        pred = int(np.argmax(dist))
        preds += 1
        hits += int(pred == IDX[nxt])
        persist_hits += int(cur == nxt)
        switch_days += int(cur != nxt)
        attempts += int(pred != IDX[cur])

    print(f"predictions={preds}  switch_days={switch_days}  switch_attempts={attempts}")
    print(f"model hits={hits} ({hits/preds:.2%})   persistence hits={persist_hits} ({persist_hits/preds:.2%})")
    ok = (preds == 328 and switch_days == 30 and attempts == 0
          and hits == 298 and hits == persist_hits)
    print("REPRODUCED" if ok else "MISMATCH — investigate before trusting the headline")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
