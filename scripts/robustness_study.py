"""Cross-asset / cross-period robustness study.

Re-runs the EXACT honest evaluation (app/services/evaluation.evaluate) across a
basket of liquid markets and a few sub-periods, to test whether the headline
finding — "the Markov regime model has no edge over persistence, and neither
trading policy beats buy-and-hold net of costs" — generalizes beyond one asset,
one cycle (the project's #1 caveat).

Data: yfinance (Yahoo) only. No exchange APIs. Simulation only. Pins every pull
to data/snapshots/ for reproducibility and writes per-asset + summary JSON to
results/robustness/.
"""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

from app.services.data_cache import load_or_fetch
from app.services.evaluation import evaluate

OUT = Path("results/robustness")
OUT.mkdir(parents=True, exist_ok=True)

# Liquid, Yahoo-available. Crypto + equities + indices + macro ETFs + FX, so the
# test spans very different regime structures. (Binance/exchange APIs excluded by
# policy; these are Yahoo price series, not exchange feeds.)
SYMBOLS = [
    # crypto
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD",
    "LTC-USD", "AVAX-USD", "LINK-USD",
    # equities
    "AAPL", "NVDA", "TSLA", "MSFT", "AMZN",
    # indices / macro ETFs
    "SPY", "QQQ", "GLD", "TLT", "USO",
    # fx
    "EURUSD=X",
]

GRID_WINDOWS = (10, 20, 30)
GRID_K = (0.2, 0.35, 0.5, 0.75)


def _summarize(symbol: str, rep) -> dict:
    acc = rep.accuracy
    pol = {p.name: p for p in rep.policies}
    base = pol.get("baseline")
    hold = pol.get("buy_hold")
    edge_bal = acc.balanced_accuracy - acc.persistence_balanced
    edge_hit = acc.hit_rate - acc.persistence_hit_rate
    return {
        "symbol": symbol,
        "n_test": acc.n,
        "chosen_window": rep.chosen_window,
        "chosen_k": rep.chosen_k,
        "regime_mix": rep.regime_mix,
        "model_hit": acc.hit_rate,
        "model_bal": acc.balanced_accuracy,
        "persist_hit": acc.persistence_hit_rate,
        "persist_bal": acc.persistence_balanced,
        "naive_hit": acc.naive_hit_rate,
        "naive_bal": acc.naive_balanced,
        "edge_vs_persist_bal": edge_bal,     # >0 means model adds skill over "tomorrow=today"
        "edge_vs_persist_hit": edge_hit,
        "baseline_return": base.total_return if base else None,
        "buyhold_return": hold.total_return if hold else None,
        "baseline_sharpe": base.sharpe if base else None,
        "buyhold_sharpe": hold.sharpe if hold else None,
        "baseline_trades": base.num_trades if base else None,
        "beats_persistence": bool(edge_bal > 0.02),          # real skill claim
        "beats_buyhold": bool(base and hold and base.total_return > hold.total_return),
        "any_policy_positive": bool(
            base and hold and max(base.total_return, hold.total_return) > 0
        ),
        "data_hash": rep.data_hash[:12],
        "warnings": rep.warnings,
    }


def run_one(symbol: str, history=None) -> dict | None:
    try:
        if history is None:
            history, src = load_or_fetch(symbol, years=3)
        rep = evaluate(
            history, symbol=symbol, train_frac=0.7,
            grid_windows=GRID_WINDOWS, grid_k=GRID_K,
            fee_pct=0.3, slippage_pct=0.5,
        )
        (OUT / f"{symbol.replace('/', '_').replace('=', '_')}.json").write_text(
            json.dumps(asdict(rep), indent=2, default=str)
        )
        return _summarize(symbol, rep)
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {symbol}: {e}", flush=True)
        return {"symbol": symbol, "error": str(e), "trace": traceback.format_exc()[-400:]}


def main() -> None:
    rows = []
    print(f"=== Cross-asset robustness: {len(SYMBOLS)} symbols ===", flush=True)
    for i, sym in enumerate(SYMBOLS, 1):
        print(f"[{i}/{len(SYMBOLS)}] {sym} ...", flush=True)
        r = run_one(sym)
        if r:
            rows.append(r)
            if "error" not in r:
                print(
                    f"    n={r['n_test']:>4}  model_bal={r['model_bal']:.3f}  "
                    f"persist_bal={r['persist_bal']:.3f}  edge={r['edge_vs_persist_bal']:+.3f}  "
                    f"base={r['baseline_return']:+.1%}  hold={r['buyhold_return']:+.1%}  "
                    f"beats_hold={r['beats_buyhold']}", flush=True
                )

    # --- temporal robustness: split BTC + SPY into early/late halves ---
    temporal = []
    for sym in ("BTC-USD", "SPY"):
        try:
            hist, _ = load_or_fetch(sym, years=3)
            mid = len(hist) // 2
            for label, sl in (("early", hist.iloc[:mid]), ("late", hist.iloc[mid:])):
                r = run_one(f"{sym}:{label}", history=sl)
                if r and "error" not in r:
                    temporal.append(r)
                    print(
                        f"    [temporal] {sym}:{label}  edge={r['edge_vs_persist_bal']:+.3f}  "
                        f"base={r['baseline_return']:+.1%}  hold={r['buyhold_return']:+.1%}", flush=True
                    )
        except Exception as e:  # noqa: BLE001
            print(f"  [temporal FAIL] {sym}: {e}", flush=True)

    ok = [r for r in rows if "error" not in r]
    agg = {
        "n_assets": len(ok),
        "n_failed": len(rows) - len(ok),
        "assets_model_beats_persistence": sum(r["beats_persistence"] for r in ok),
        "assets_baseline_beats_buyhold": sum(r["beats_buyhold"] for r in ok),
        "assets_any_policy_positive": sum(r["any_policy_positive"] for r in ok),
        "mean_edge_vs_persist_bal": (sum(r["edge_vs_persist_bal"] for r in ok) / len(ok)) if ok else None,
        "max_edge_vs_persist_bal": max((r["edge_vs_persist_bal"] for r in ok), default=None),
    }
    summary = {"aggregate": agg, "assets": rows, "temporal": temporal}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print("\n=== AGGREGATE ===", flush=True)
    for k, v in agg.items():
        print(f"  {k}: {v}", flush=True)
    print(f"\n[saved] {OUT/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
