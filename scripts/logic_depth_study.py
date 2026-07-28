"""Logic depth study — three questions the argmax identity does NOT answer.

The identity (argmax forecast == persistence, 0 divergences) says the model's
DECISIONS add nothing. Three sharper questions remain, answered here for every
studied asset run, all strictly causal:

1. HOW TIGHT IS THE BOUND? Recompute divergences per run with an incremental
   estimator that exactly replicates `estimate_transition_matrix` (MLE counts,
   uniform fallback), pool them, and apply the exact zero-events upper bound
   (the "rule of three"): with 0 divergences in N paired predictions, the
   one-sided 95% bound on the true divergence rate is 1 - 0.05^(1/N).

2. MEMORY-2. Does conditioning on the LAST TWO states (a second-order chain)
   ever produce a switch-dominant context with real support — e.g. right after
   a fresh switch, is switching back more likely than staying? If yes, score it
   walk-forward vs persistence on the held-out region (predict-then-update,
   MIN_OBS support, no tuning).

3. PROBABILISTIC SKILL. Identical argmax does not mean identical FORECASTS.
   Score the model's causal probability vector against a causal "sticky
   baseline" (global stay-rate on today's state, remainder split evenly) with
   log-loss and Brier on the same held-out days. If the Markov row scores
   better, the matrix carries real probabilistic information despite the
   decision tie; if not, the null deepens to the probability level.

Self-check: for EVERY run, the incremental estimator is asserted equal to the
engine's `estimate_transition_matrix` at one train-region checkpoint plus up to
four test-region checkpoints (a hostile differential review additionally
verified equality on every scored BTC day, max abs diff 0.0). The labeled
sequence is asserted gap-free (warmup prefix only), which is the adjacency
precondition the incremental counting relies on.

Writes results/logic_depth/summary.json.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.data_cache import history_hash, load_or_fetch
from app.services.evaluation import causal_states
from app.services.regime import STATE_ORDER, STATE_TO_IDX, estimate_transition_matrix
from scripts.model_logic_study import _configs

OUT = Path("results/logic_depth")
OUT.mkdir(parents=True, exist_ok=True)
MIN_OBS = 15
TRAIN_FRAC = 0.7
_EPS = 1e-12
N_STATES = len(STATE_ORDER)


def _row(counts: np.ndarray, i: int) -> np.ndarray:
    s = counts[i].sum()
    return counts[i] / s if s > 0 else np.ones(N_STATES) / N_STATES


def analyze(symbol: str, years: int, window: int, k: float,
            expected_hash: str = "") -> dict:
    history, _ = load_or_fetch(symbol, years=years)
    if expected_hash and history_hash(history) != expected_hash:
        raise ValueError(f"{symbol}: snapshot hash != run-record hash — refusing to mix.")
    closes = history["Close"] if "Close" in history.columns else history.squeeze()
    returns = closes.pct_change().dropna()
    states = causal_states(returns, window, k)
    n = len(returns)
    split = int(n * TRAIN_FRAC)

    # Adjacency precondition: the incremental counts assume labels are gap-free
    # after warmup (the engine compresses across gaps; we count adjacent pairs).
    # With the current labeler None only occurs in the warmup prefix — assert it
    # so a future labeler change fails loudly instead of silently drifting.
    labeled_idx = [i for i, s in enumerate(states) if s in STATE_TO_IDX]
    assert labeled_idx == list(range(labeled_idx[0], labeled_idx[-1] + 1)), (
        f"{symbol}: mid-series label gap — incremental counting no longer matches "
        "the engine's compressed-pair semantics; fix before trusting results."
    )

    counts = np.zeros((N_STATES, N_STATES))        # first-order, incremental MLE
    counts2: dict = defaultdict(lambda: np.zeros(N_STATES))  # (prev,cur) contexts
    stay_transitions = 0                            # for the sticky baselines
    total_transitions = 0
    exit_into = np.zeros(N_STATES)                  # global switch-target counts
    last_obs: int | None = None                     # engine semantics: compressed
    prev_adjacent: int | None = None                # memory-2: strictly consecutive

    res = {
        "n_scored": 0, "divergences": 0, "min_prefix_self_prob": 1.0,
        "markov_ll": 0.0, "sticky_ll": 0.0, "sticky2_ll": 0.0,
        "markov_brier": 0.0, "sticky_brier": 0.0, "sticky2_brier": 0.0,
        "m2_scored": 0, "m2_attempts": 0, "m2_hits": 0, "m2_persist_hits": 0,
        "switchy_contexts_seen": 0,
    }
    checked = 0

    for t in range(n - 1):
        cur_s, nxt_s = states[t], states[t + 1]
        cur = STATE_TO_IDX.get(cur_s) if cur_s in STATE_TO_IDX else None
        nxt = STATE_TO_IDX.get(nxt_s) if nxt_s in STATE_TO_IDX else None

        # ---- estimator self-check: one train-region + several test-region ----
        if cur is not None and last_obs is not None and (
            t == split // 2 or (t >= split and checked < 5 and t % 97 == 0)
        ):
            observed = [s for s in states[: t + 1] if s in STATE_TO_IDX]
            ref = estimate_transition_matrix(pd.Series(observed))[cur]
            assert np.allclose(_row(counts, cur), ref), f"{symbol}: estimator drift at t={t}"
            checked += 1

        # ---- score day t (predict t+1) BEFORE updating any counts ----------
        if cur is not None and nxt is not None and last_obs is not None and t >= split:
            dist = _row(counts, cur)
            res["n_scored"] += 1
            if int(np.argmax(dist)) != cur:
                res["divergences"] += 1
            # per-STEP dominance: the self-probability of the row ACTUALLY used
            # at this decision (causal prefix matrix) — the quantity that forces
            # the identity day by day, not a full-sample summary
            res["min_prefix_self_prob"] = min(res["min_prefix_self_prob"], float(dist[cur]))
            # probabilistic skill vs causal sticky baseline
            s_hat = (stay_transitions / total_transitions) if total_transitions else 1.0 / N_STATES
            sticky = np.full(N_STATES, (1.0 - s_hat) / (N_STATES - 1))
            sticky[cur] = s_hat
            # stronger baseline: exit mass follows the causal GLOBAL switch-target
            # distribution (state-agnostic) instead of a uniform split
            sticky2 = np.zeros(N_STATES)
            sticky2[cur] = s_hat
            others = [j for j in range(N_STATES) if j != cur]
            w_ = np.array([exit_into[j] for j in others], dtype=float)
            w_ = w_ / w_.sum() if w_.sum() > 0 else np.ones(len(others)) / len(others)
            for j, wj in zip(others, w_):
                sticky2[j] = (1.0 - s_hat) * wj
            y = np.zeros(N_STATES)
            y[nxt] = 1.0
            res["markov_ll"] += -math.log(max(float(dist[nxt]), _EPS))
            res["sticky_ll"] += -math.log(max(float(sticky[nxt]), _EPS))
            res["sticky2_ll"] += -math.log(max(float(sticky2[nxt]), _EPS))
            res["markov_brier"] += float(((dist - y) ** 2).sum())
            res["sticky_brier"] += float(((sticky - y) ** 2).sum())
            res["sticky2_brier"] += float(((sticky2 - y) ** 2).sum())
            # memory-2 (only when yesterday is adjacent + labeled)
            if prev_adjacent is not None:
                ctx = (prev_adjacent, cur)
                c2 = counts2[ctx]
                if c2.sum() >= MIN_OBS:
                    res["m2_scored"] += 1
                    pred2 = int(np.argmax(c2))
                    if pred2 != cur:
                        res["m2_attempts"] += 1
                    res["m2_hits"] += int(pred2 == nxt)
                    res["m2_persist_hits"] += int(cur == nxt)

        # ---- update counts with the (t -> t+1) observation -----------------
        # (adjacent-pair counting; the gap-free assertion above guarantees this
        # equals the engine's compressed-pair semantics)
        if cur is not None:
            if nxt is not None:
                counts[cur][nxt] += 1
                total_transitions += 1
                stay_transitions += int(nxt == cur)
                if nxt != cur:
                    exit_into[nxt] += 1
                if prev_adjacent is not None:
                    counts2[(prev_adjacent, cur)][nxt] += 1
            last_obs = cur
        prev_adjacent = cur if (cur is not None) else None

    # switch-dominant contexts with support, over the WHOLE history (descriptive)
    for (p_, c_), c2 in counts2.items():
        if c2.sum() >= MIN_OBS and int(np.argmax(c2)) != c_:
            res["switchy_contexts_seen"] += 1

    ns = res["n_scored"]
    out = {
        "symbol": symbol, "years": years, "window": window, "k": k,
        "n_scored": ns, "divergences": res["divergences"],
        "min_prefix_self_prob": round(res["min_prefix_self_prob"], 6),
        "markov_log_loss": round(res["markov_ll"] / ns, 6) if ns else None,
        "sticky_log_loss": round(res["sticky_ll"] / ns, 6) if ns else None,
        "sticky2_log_loss": round(res["sticky2_ll"] / ns, 6) if ns else None,
        "ll_improvement": round((res["sticky_ll"] - res["markov_ll"]) / ns, 6) if ns else None,
        "ll_improvement_vs_sticky2": round((res["sticky2_ll"] - res["markov_ll"]) / ns, 6) if ns else None,
        "markov_brier": round(res["markov_brier"] / ns, 6) if ns else None,
        "sticky_brier": round(res["sticky_brier"] / ns, 6) if ns else None,
        "sticky2_brier": round(res["sticky2_brier"] / ns, 6) if ns else None,
        "brier_improvement": round((res["sticky_brier"] - res["markov_brier"]) / ns, 6) if ns else None,
        "brier_improvement_vs_sticky2": round((res["sticky2_brier"] - res["markov_brier"]) / ns, 6) if ns else None,
        "m2": {
            "scored": res["m2_scored"], "attempts": res["m2_attempts"],
            "switchy_contexts": res["switchy_contexts_seen"],
            "hit_rate": round(res["m2_hits"] / res["m2_scored"], 6) if res["m2_scored"] else None,
            "persist_hit_rate": round(res["m2_persist_hits"] / res["m2_scored"], 6) if res["m2_scored"] else None,
            "edge": round((res["m2_hits"] - res["m2_persist_hits"]) / res["m2_scored"], 6) if res["m2_scored"] else None,
        },
    }
    return out


def main() -> None:
    cfgs = _configs()
    print(f"=== Logic depth study: {len(cfgs)} asset runs ===", flush=True)
    rows = []
    for i, (sym, yrs, w, k, h) in enumerate(cfgs, 1):
        r = analyze(sym, yrs, w, k, expected_hash=h)
        rows.append(r)
        print(
            f"[{i:>2}/{len(cfgs)}] {sym:<10} ({yrs}y)  n={r['n_scored']:>5}  div={r['divergences']}  "
            f"minSelfP={r['min_prefix_self_prob']:.3f}  "
            f"dLL={r['ll_improvement']:+.4f}  dBrier={r['brier_improvement']:+.4f}  "
            f"m2: switchy_ctx={r['m2']['switchy_contexts']} attempts={r['m2']['attempts']} "
            f"edge={r['m2']['edge'] if r['m2']['edge'] is not None else 0:+.4f}",
            flush=True,
        )

    pooled_n = sum(r["n_scored"] for r in rows)
    pooled_div = sum(r["divergences"] for r in rows)
    ub95 = 1.0 - 0.05 ** (1.0 / pooled_n) if pooled_div == 0 else None
    agg = {
        "pooled_predictions": pooled_n,
        "pooled_divergences": pooled_div,
        "divergence_rate_95pct_upper_bound": ub95,
        "global_min_prefix_self_prob": min(r["min_prefix_self_prob"] for r in rows),
        "runs_prefix_dominant_throughout": sum(r["min_prefix_self_prob"] > 0.5 for r in rows),
        "mean_ll_improvement": round(float(np.mean([r["ll_improvement"] for r in rows])), 6),
        "runs_markov_better_ll": sum(r["ll_improvement"] > 0 for r in rows),
        "mean_ll_improvement_vs_sticky2": round(float(np.mean([r["ll_improvement_vs_sticky2"] for r in rows])), 6),
        "runs_markov_better_ll_vs_sticky2": sum(r["ll_improvement_vs_sticky2"] > 0 for r in rows),
        "mean_brier_improvement": round(float(np.mean([r["brier_improvement"] for r in rows])), 6),
        "runs_markov_better_brier": sum(r["brier_improvement"] > 0 for r in rows),
        "runs_markov_better_brier_vs_sticky2": sum(r["brier_improvement_vs_sticky2"] > 0 for r in rows),
        "total_switchy_m2_contexts": sum(r["m2"]["switchy_contexts"] for r in rows),
        "total_m2_attempts": sum(r["m2"]["attempts"] for r in rows),
        # edge range over FIRING runs only — an edge from a run that never
        # attempted a switch is definitionally 0 and would dilute the range
        "worst_m2_edge": min((r["m2"]["edge"] for r in rows
                              if r["m2"]["attempts"] > 0 and r["m2"]["edge"] is not None), default=None),
        "best_m2_edge": max((r["m2"]["edge"] for r in rows
                             if r["m2"]["attempts"] > 0 and r["m2"]["edge"] is not None), default=None),
    }
    (OUT / "summary.json").write_text(json.dumps({"aggregate": agg, "runs": rows}, indent=2))
    print("\n=== AGGREGATE ===", flush=True)
    for kk, v in agg.items():
        print(f"  {kk}: {v}", flush=True)
    print(f"\n[saved] {OUT/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
