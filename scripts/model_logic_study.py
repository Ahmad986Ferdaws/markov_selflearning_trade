"""Model-logic deep dive: WHY the edge is zero, and whether duration rescues it.

Three questions, answered on every studied asset (20 x 3y robustness + 8 x long
history), using each run's already-chosen (window, k):

1. THEOREM CHECK. The argmax forecast equals persistence whenever every state's
   self-transition probability exceeds 1/2 (off-diagonals then sum to < 1/2, so
   no single one can win). Measure the minimum diagonal across all estimated
   matrices — the margin by which the identity is *forced*, not incidental.

2. HAZARD CURVES. A first-order chain cannot time switches; a duration-aware
   (semi-Markov) model could — IF the empirical hazard h(d) = P(switch tomorrow |
   d days in state) rises above 1/2 somewhere. Measure h(d) per state (train
   region only).

3. THE OBVIOUS NEXT MODEL, TESTED HONESTLY. A causal duration-augmented
   predictor: at day t, estimate hazards and successor frequencies from
   states[:t+1] ONLY (incremental counts, no lookahead); predict a switch when
   the hazard estimate at the current duration exceeds 1/2 (with >= MIN_OBS
   observations), else persist. Scored on the same held-out region as the
   headline results. NOTE: this is an exploratory analysis (one extra model,
   reported whatever it says) — not a tuned search.

Writes results/model_logic/summary.json.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.data_cache import history_hash, load_or_fetch
from app.services.evaluation import causal_states
from app.services.regime import STATE_ORDER, STATE_TO_IDX, estimate_transition_matrix

OUT = Path("results/model_logic")
OUT.mkdir(parents=True, exist_ok=True)
MIN_OBS = 15          # minimum observations of (state, duration) before the
                      # hazard estimate is allowed to trigger a switch call
TRAIN_FRAC = 0.7


def _configs() -> list[tuple[str, int, int, float, str]]:
    """(symbol, years, window, k, data_hash) for every clean per-asset run record.
    The hash pins the analysis to the exact snapshot the (window, k) was chosen
    on — a refreshed snapshot fails loudly instead of silently mismatching."""
    out = []
    for p in sorted(Path("results/robustness").glob("*.json")):
        if p.name == "summary.json":
            continue
        d = json.loads(p.read_text())
        sym = d.get("symbol", "")
        if "error" in d or ":" in sym:      # skip failed + sub-period slices
            continue
        out.append((sym, 3, int(d["chosen_window"]), float(d["chosen_k"]),
                    str(d.get("data_hash", ""))))
    for p in sorted(Path("results/long_history").glob("*.json")):
        if p.name == "summary.json":
            continue
        d = json.loads(p.read_text())
        s, r = d.get("summary", {}), d.get("report", {})
        if "error" in s or not r:
            continue
        out.append((s["symbol"], int(s["years_requested"]),
                    int(r["chosen_window"]), float(r["chosen_k"]),
                    str(r.get("data_hash", ""))))
    return out


def analyze(symbol: str, years: int, window: int, k: float,
            expected_hash: str = "") -> dict:
    history, _ = load_or_fetch(symbol, years=years)
    if expected_hash and history_hash(history) != expected_hash:
        raise ValueError(
            f"{symbol}: snapshot hash != run-record hash — the pinned data has "
            "changed since (window, k) was chosen; refusing to mix them."
        )
    closes = history["Close"] if "Close" in history.columns else history.squeeze()
    returns = closes.pct_change().dropna()
    states = causal_states(returns, window, k)
    labeled = [(i, s) for i, s in enumerate(states) if s in STATE_TO_IDX]
    n = len(returns)
    split = int(n * TRAIN_FRAC)

    # --- 1. theorem check: final estimated matrix diagonal dominance ---------
    # Only VISITED states matter: a state with zero occurrences gets a uniform
    # fallback row (1/3 everywhere), which is trivially non-dominant — but an
    # unvisited state can never be "today", so its row can never break the
    # identity. (This is exactly the NVDA k=0.35 case: zero bear days in 3y.)
    observed = [s for s in states if s in STATE_TO_IDX]
    P = estimate_transition_matrix(pd.Series(observed))
    diag = [float(P[i][i]) for i in range(len(STATE_ORDER))]
    occupancy = {s: observed.count(s) for s in STATE_ORDER}
    visited = [s for s in STATE_ORDER if occupancy[s] > 0]
    min_diag_visited = min(diag[STATE_TO_IDX[s]] for s in visited)
    dominant = all(int(np.argmax(P[STATE_TO_IDX[s]])) == STATE_TO_IDX[s] for s in visited)

    # --- 2. train-region hazard curves h(d) per state -------------------------
    haz_exit: dict = defaultdict(lambda: defaultdict(int))
    haz_surv: dict = defaultdict(lambda: defaultdict(int))

    def _runs(pairs):
        """Yield (state, duration_at_t, switched) for consecutive labeled days."""
        d = 0
        for (i0, s0), (i1, s1) in zip(pairs, pairs[1:]):
            if i1 != i0 + 1:          # gap (warmup None) — reset the run
                d = 0
                continue
            d = d + 1 if d else 1
            yield s0, d, (s1 != s0)
            if s1 != s0:
                d = 0

    for s, d, switched in _runs([(i, s) for i, s in labeled if i < split]):
        (haz_exit if switched else haz_surv)[s][d] += 1

    def hazard(s, d):
        e, v = haz_exit[s].get(d, 0), haz_surv[s].get(d, 0)
        return (e / (e + v), e + v) if e + v else (0.0, 0)

    haz_table = {
        s: {d: {"h": round(hazard(s, d)[0], 3), "n": hazard(s, d)[1]}
            for d in (1, 2, 3, 5, 10, 20) if hazard(s, d)[1] > 0}
        for s in STATE_ORDER
    }
    max_h = 0.0
    for s in STATE_ORDER:
        for d in set(haz_exit[s]) | set(haz_surv[s]):
            h, cnt = hazard(s, d)
            if cnt >= MIN_OBS:
                max_h = max(max_h, h)

    # --- 3. causal duration-augmented predictor on the held-out region -------
    ex: dict = defaultdict(lambda: defaultdict(int))     # incremental, causal
    sv: dict = defaultdict(lambda: defaultdict(int))
    succ: dict = defaultdict(lambda: defaultdict(int))
    dur = 0
    prev_i = None
    stats = {"n": 0, "dur_hits": 0, "per_hits": 0, "attempts": 0,
             "switch_days": 0, "dur_switch_hits": 0}
    for (i0, s0), (i1, s1) in zip(labeled, labeled[1:]):
        if i1 != i0 + 1:
            dur = 0
            prev_i = i1
            continue
        dur = dur + 1 if prev_i == i0 else 1
        prev_i = i1
        # predict s1 BEFORE seeing it (counts so far are strictly past)
        if i0 >= split:
            e, v = ex[s0].get(dur, 0), sv[s0].get(dur, 0)
            fire = (e + v) >= MIN_OBS and e / (e + v) > 0.5
            # `fire` implies e > 0 implies succ[s0] non-empty; tie-break on
            # successor counts is made order-independent (lexicographic) so the
            # prediction never depends on observation insertion order.
            pred = (max(sorted(succ[s0]), key=succ[s0].get) if fire else s0)
            stats["n"] += 1
            stats["attempts"] += int(pred != s0)
            stats["dur_hits"] += int(pred == s1)
            stats["per_hits"] += int(s0 == s1)
            if s1 != s0:
                stats["switch_days"] += 1
                stats["dur_switch_hits"] += int(pred == s1)
        # then observe and update
        if s1 != s0:
            ex[s0][dur] += 1
            succ[s0][s1] += 1
            dur = 0
        else:
            sv[s0][dur] += 1

    return {
        "symbol": symbol, "years": years, "window": window, "k": k,
        "matrix_diag": [round(x, 4) for x in diag],
        "occupancy": occupancy,
        "unvisited_states": [s for s in STATE_ORDER if occupancy[s] == 0],
        "min_diag_visited": round(min_diag_visited, 4),
        "diag_dominant_visited": bool(dominant),
        "hazard_at_duration": haz_table,
        "max_hazard_with_support": round(max_h, 4),
        "duration_model": {
            **stats,
            "dur_hit_rate": round(stats["dur_hits"] / stats["n"], 6) if stats["n"] else None,
            "per_hit_rate": round(stats["per_hits"] / stats["n"], 6) if stats["n"] else None,
            "edge": round((stats["dur_hits"] - stats["per_hits"]) / stats["n"], 6) if stats["n"] else None,
        },
    }


def main() -> None:
    cfgs = _configs()
    print(f"=== Model-logic study: {len(cfgs)} asset runs ===", flush=True)
    rows = []
    for i, (sym, yrs, w, k, h) in enumerate(cfgs, 1):
        r = analyze(sym, yrs, w, k, expected_hash=h)
        rows.append(r)
        dm = r["duration_model"]
        edge_s = f"{dm['edge']:+.4f}" if dm["edge"] is not None else "n/a"
        print(
            f"[{i:>2}/{len(cfgs)}] {sym:<10} ({yrs}y w={w} k={k})  "
            f"min_diag(visited)={r['min_diag_visited']:.3f} dominant={r['diag_dominant_visited']}"
            f"{' unvisited=' + ','.join(r['unvisited_states']) if r['unvisited_states'] else ''}  "
            f"max_hazard={r['max_hazard_with_support']:.3f}  "
            f"dur_model: attempts={dm['attempts']} edge={edge_s}",
            flush=True,
        )

    agg = {
        "n_runs": len(rows),
        "all_visited_rows_dominant": all(r["diag_dominant_visited"] for r in rows),
        "global_min_visited_diagonal": min(r["min_diag_visited"] for r in rows),
        "runs_with_unvisited_state": sum(bool(r["unvisited_states"]) for r in rows),
        "global_max_hazard_with_support": max(r["max_hazard_with_support"] for r in rows),
        "runs_where_duration_model_fired": sum(r["duration_model"]["attempts"] > 0 for r in rows),
        "total_duration_attempts": sum(r["duration_model"]["attempts"] for r in rows),
        "runs_with_nonzero_duration_edge": sum(
            abs(r["duration_model"]["edge"] or 0) > 1e-12 for r in rows
        ),
        "worst_duration_edge": min((r["duration_model"]["edge"] or 0) for r in rows),
        "best_duration_edge": max((r["duration_model"]["edge"] or 0) for r in rows),
    }
    (OUT / "summary.json").write_text(json.dumps({"aggregate": agg, "runs": rows}, indent=2))
    print("\n=== AGGREGATE ===", flush=True)
    for kk, v in agg.items():
        print(f"  {kk}: {v}", flush=True)
    print(f"\n[saved] {OUT/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
