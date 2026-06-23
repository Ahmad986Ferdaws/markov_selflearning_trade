"""INDEPENDENT verification of the docs/16 claim:
   model argmax == current state on every held-out test day => 0 divergences / 5425.
Re-implements the selection + walk-forward path and counts pointwise divergences
between argmax(model_dist) and cur_idx, day by day, from PINNED snapshots only.
"""
import numpy as np
import pandas as pd
from app.services.data_cache import load_or_fetch
from app.services.regime import STATE_ORDER, STATE_TO_IDX, estimate_transition_matrix, label_state

GRID_W = (10, 20, 30)
GRID_K = (0.2, 0.35, 0.5, 0.75)
MIN_TRAIN = 60

def causal_states(returns, window, k):
    mean = returns.rolling(window=window, min_periods=window).mean()
    std = returns.rolling(window=window, min_periods=window).std()
    return [label_state(m, s, k) for m, s in zip(mean, std)]

def walk(returns, states, start, end):
    r = returns.to_numpy(); n = len(r)
    preds, actual, cur = [], [], []
    for t in range(start, min(end, n - 1)):
        c = states[t]; nx = states[t + 1]
        if c is None or nx is None or c not in STATE_TO_IDX or nx not in STATE_TO_IDX:
            continue
        observed = [s for s in states[: t + 1] if s in STATE_TO_IDX]
        if len(observed) < 2:
            continue
        p = estimate_transition_matrix(pd.Series(observed))
        dist = p[STATE_TO_IDX[c]]
        preds.append(int(np.argmax(dist)))
        actual.append(STATE_TO_IDX[nx])
        cur.append(STATE_TO_IDX[c])
    return preds, actual, cur

def train_freq(states, start, end):
    counts = np.zeros(len(STATE_ORDER))
    for s in states[start:end]:
        if s in STATE_TO_IDX:
            counts[STATE_TO_IDX[s]] += 1
    tot = counts.sum()
    return counts / tot if tot > 0 else np.ones(len(STATE_ORDER)) / len(STATE_ORDER)

def bal_acc(pred, actual):
    n_states = len(STATE_ORDER)
    tot = np.zeros(n_states); cor = np.zeros(n_states)
    for p, a in zip(pred, actual):
        tot[a] += 1
        if p == a: cor[a] += 1
    rec = [cor[c]/tot[c] for c in range(n_states) if tot[c] > 0]
    return float(np.mean(rec)) if rec else 0.0

def select_and_test(returns):
    n = len(returns); split = int(n * 0.7)
    cands = []
    for w in GRID_W:
        for k in GRID_K:
            st = causal_states(returns, w, k)
            pr, ac, _ = walk(returns, st, MIN_TRAIN, split - 1)
            if not ac: continue
            tf = train_freq(st, MIN_TRAIN, split)
            cands.append({"w": w, "k": k, "score": bal_acc(pr, ac),
                          "max_share": float(tf.max()), "nm": int((tf >= 0.10).sum())})
    diverse = [c for c in cands if c["nm"] >= 2]
    best = max(diverse, key=lambda c: c["score"]) if diverse else min(cands, key=lambda c: c["max_share"])
    w, k = best["w"], best["k"]
    st = causal_states(returns, w, k)
    pred, actual, cur = walk(returns, st, split, n)
    div = sum(1 for p, c in zip(pred, cur) if p != c)  # model-argmax vs persistence(current)
    return w, k, len(pred), div, [i for i,(p,c) in enumerate(zip(pred,cur)) if p!=c]

SYMS = ["BTC-USD","ETH-USD","SOL-USD","XRP-USD","DOGE-USD","ADA-USD","LTC-USD","AVAX-USD","LINK-USD",
        "AAPL","NVDA","TSLA","MSFT","AMZN","SPY","QQQ","GLD","TLT","USO","EURUSD=X"]

total_n = 0; total_div = 0
print(f"{'sym':<10}{'w':>3}{'k':>6}{'n':>6}{'div':>6}")
for s in SYMS:
    hist, _ = load_or_fetch(s, years=3)
    closes = hist["Close"] if "Close" in hist.columns else hist.squeeze()
    rets = closes.pct_change().dropna()
    w,k,n,div,_ = select_and_test(rets)
    total_n += n; total_div += div
    print(f"{s:<10}{w:>3}{k:>6}{n:>6}{div:>6}")
print(f"{'TOTAL':<10}{'':>3}{'':>6}{total_n:>6}{total_div:>6}")

# BTC early split
hist,_ = load_or_fetch("BTC-USD", years=3)
closes = hist["Close"] if "Close" in hist.columns else hist.squeeze()
mid = len(hist)//2
early = hist.iloc[:mid]
ce = early["Close"] if "Close" in early.columns else early.squeeze()
re = ce.pct_change().dropna()
w,k,n,div,didx = select_and_test(re)
print(f"\nBTC-early: w={w} k={k} n={n} divergences={div} at day-indices={didx}")
