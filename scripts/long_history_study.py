"""Long-history study — attack the project's #1 stated limitation.

docs/16 closes with: "one ~3-year window per asset; a longer multi-cycle history
is the next test." This runs the EXACT same honest evaluation (same grid, same
costs, same 70/30 split — protocol identical to scripts/robustness_study.py) on
the longest history Yahoo will give per asset: ~11y of BTC, ~20y of SPY/GLD/TLT
(which spans the 2008 crisis, the 2010s bull, COVID, and the 2022 bear).

If the null result — zero edge over persistence — survives multiple full market
cycles, the finding graduates from "one window" to "robust across regimes."

Also verifies the structural identity POINTWISE on the held-out region: counts
the days where argmax(P[today]) != today's state. Zero divergences == the model
IS persistence, not merely tied with it.

Data: yfinance only. Simulation only. Snapshots pinned to data/snapshots/,
outputs to results/long_history/.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.data_cache import load_or_fetch
from app.services.evaluation import _walk_forward, causal_states, evaluate
from app.services.regime import STATE_TO_IDX

OUT = Path("results/long_history")
OUT.mkdir(parents=True, exist_ok=True)

# (symbol, years requested) — years capped by Yahoo's actual coverage per asset.
SYMBOLS = [
    ("BTC-USD", 12),   # ~2014+ (Yahoo's full BTC history): two halvings, 2018 + 2022 bears
    ("ETH-USD", 10),   # ~2017+
    ("SPY", 20),       # ~2006+: includes the 2008 crisis, COVID, 2022
    ("QQQ", 20),
    ("AAPL", 20),
    ("GLD", 20),
    ("TLT", 20),       # a full rate cycle
    ("EURUSD=X", 20),
]

# IDENTICAL protocol to scripts/robustness_study.py — comparability is the point.
GRID_WINDOWS = (10, 20, 30)
GRID_K = (0.2, 0.35, 0.5, 0.75)


def _count_divergences(history: pd.DataFrame, window: int, k: float) -> dict:
    """Pointwise identity check on the held-out region with the chosen config:
    how often does argmax(P[today]) differ from today's state?"""
    closes = history["Close"] if "Close" in history.columns else history.squeeze()
    returns = closes.pct_change().dropna()
    n = len(returns)
    split = int(n * 0.7)
    states = causal_states(returns, window, k)
    wf = _walk_forward(returns, states, split, n, 0.3, 0.5)
    div = sum(
        1 for dist, cur in zip(wf["pred_probs"], wf["cur_idx"])
        if int(np.argmax(dist)) != cur
    )
    return {"n_predictions": len(wf["pred_probs"]), "divergences": div}


def run_one(symbol: str, years: int) -> dict:
    try:
        history, src = load_or_fetch(symbol, years=years)
        span = f"{history.index[0].date()} -> {history.index[-1].date()}"
        rep = evaluate(
            history, symbol=symbol, train_frac=0.7,
            grid_windows=GRID_WINDOWS, grid_k=GRID_K,
            fee_pct=0.3, slippage_pct=0.5,
        )
        ident = _count_divergences(history, rep.chosen_window, rep.chosen_k)
        acc = rep.accuracy
        pol = {p.name: p for p in rep.policies}
        base, hold = pol.get("baseline"), pol.get("buy_hold")
        row = {
            "symbol": symbol,
            "years_requested": years,
            "span": span,
            "n_days": len(history),
            "n_test": acc.n,
            "chosen_window": rep.chosen_window,
            "chosen_k": rep.chosen_k,
            "model_hit": acc.hit_rate,
            "model_bal": acc.balanced_accuracy,
            "persist_hit": acc.persistence_hit_rate,
            "persist_bal": acc.persistence_balanced,
            "edge_vs_persist_bal": acc.balanced_accuracy - acc.persistence_balanced,
            "edge_vs_persist_hit": acc.hit_rate - acc.persistence_hit_rate,
            "identity_check": ident,
            "baseline_return": base.total_return if base else None,
            "buyhold_return": hold.total_return if hold else None,
            "baseline_trades": base.num_trades if base else None,
            "data_hash": rep.data_hash[:12],
            "warnings": rep.warnings,
        }
        (OUT / f"{symbol.replace('/', '_').replace('=', '_')}.json").write_text(
            json.dumps({"summary": row, "report": asdict(rep)}, indent=2, default=str)
        )
        return row
    except Exception as e:  # noqa: BLE001
        return {"symbol": symbol, "error": str(e), "trace": traceback.format_exc()[-400:]}


def main() -> None:
    rows = []
    print(f"=== Long-history study: {len(SYMBOLS)} assets, max available span ===", flush=True)
    for i, (sym, yrs) in enumerate(SYMBOLS, 1):
        print(f"[{i}/{len(SYMBOLS)}] {sym} ({yrs}y) ...", flush=True)
        r = run_one(sym, yrs)
        rows.append(r)
        if "error" in r:
            print(f"    [FAIL] {r['error']}", flush=True)
        else:
            ic = r["identity_check"]
            print(
                f"    {r['span']}  n_test={r['n_test']}  "
                f"edge_bal={r['edge_vs_persist_bal']:+.4f}  edge_hit={r['edge_vs_persist_hit']:+.4f}  "
                f"divergences={ic['divergences']}/{ic['n_predictions']}  "
                f"base={r['baseline_return']:+.1%}  hold={r['buyhold_return']:+.1%}",
                flush=True,
            )

    ok = [r for r in rows if "error" not in r]
    agg = {
        "n_assets": len(ok),
        "n_failed": len(rows) - len(ok),
        "assets_with_nonzero_hit_edge": sum(abs(r["edge_vs_persist_hit"]) > 1e-12 for r in ok),
        "total_predictions": sum(r["identity_check"]["n_predictions"] for r in ok),
        "total_divergences": sum(r["identity_check"]["divergences"] for r in ok),
        "max_abs_edge_bal": max((abs(r["edge_vs_persist_bal"]) for r in ok), default=None),
        "assets_baseline_beats_hold": sum(
            (r["baseline_return"] or 0) > (r["buyhold_return"] or 0) for r in ok
        ),
    }
    (OUT / "summary.json").write_text(json.dumps({"aggregate": agg, "assets": rows}, indent=2, default=str))
    print("\n=== AGGREGATE ===", flush=True)
    for k, v in agg.items():
        print(f"  {k}: {v}", flush=True)
    print(f"\n[saved] {OUT/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
