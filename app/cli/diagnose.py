"""Offline run-record diagnosis — no model, no network, just the receipt.

Usage:
  diagnose-cli                       # newest record in results/
  diagnose-cli results/BTC-USD_4a150b23.json

Reads a deterministic run record written by evaluate-cli and narrates what it
shows: the edge (or absence of one) vs the persistence bar, the trading
outcomes net of costs, and every honesty-gate warning the engine raised. This
is the article-inspired "diagnose" mode kept honest: it only ever READS the
already-computed record — it cannot re-derive market state, so it is
lookahead-safe by construction.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS_DIR = Path("results")


def _latest_record(results_dir: Path = RESULTS_DIR) -> Path | None:
    records = sorted(
        (p for p in results_dir.glob("*.json")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return records[0] if records else None


def format_diagnosis(rec: dict) -> str:
    """Pure formatter: run-record dict -> human diagnosis. Testable offline."""
    acc = rec.get("accuracy", {})
    hit = float(acc.get("hit_rate", 0.0))
    per = float(acc.get("persistence_hit_rate", 0.0))
    bal = float(acc.get("balanced_accuracy", 0.0))
    pbal = float(acc.get("persistence_balanced", 0.0))
    n = int(acc.get("n", 0))
    edge_hit = hit - per
    edge_bal = bal - pbal

    lines = [
        f"=== Diagnosis: {rec.get('symbol', '?')} "
        f"(data {str(rec.get('data_hash', ''))[:12]}, n={n} held-out days) ===",
        "",
        "-- The persistence bar --",
        f"  model hit-rate       {hit:7.1%}",
        f"  persistence hit-rate {per:7.1%}   (\"tomorrow = today\")",
        f"  edge (hit)           {edge_hit:+8.4%}",
        f"  edge (balanced)      {edge_bal:+8.4%}",
    ]
    if abs(edge_hit) < 1e-9 and abs(edge_bal) < 1e-9:
        lines.append("  -> IDENTICAL to persistence. The transition matrix re-derives")
        lines.append("     autocorrelation; the high hit-rate is persistence, not skill.")
    elif edge_bal > 0.02:
        lines.append("  -> Positive edge over persistence — verify pointwise before believing it.")
    else:
        lines.append("  -> No meaningful edge over persistence (within noise).")

    pols = rec.get("policies", [])
    if pols:
        lines += ["", "-- Trading, net of costs --"]
        for p in pols:
            lines.append(
                f"  {p.get('name', '?'):<12} return {float(p.get('total_return', 0)):+8.1%}   "
                f"sharpe {float(p.get('sharpe', 0)):+6.2f}   trades {p.get('num_trades', 0)}"
            )
        rets = [float(p.get("total_return", 0)) for p in pols]
        if rets and max(rets) < 0:
            lines.append("  -> Every policy lost money net of costs on this window.")

    warns = rec.get("warnings", [])
    lines += ["", f"-- Honesty gates ({len(warns)} fired) --"]
    lines += [f"  ! {w}" for w in warns] if warns else ["  (none)"]

    lines += [
        "",
        "Verdict: read the edge lines above, not the hit-rate. A 90% hit-rate that",
        "ties persistence is a 0% improvement. This record is the receipt.",
    ]
    return "\n".join(lines)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    path = Path(args[0]) if args else _latest_record()
    if path is None or not path.exists():
        print("diagnose-cli: no run record found (run evaluate-cli first, "
              "or pass a path to a results/*.json).")
        raise SystemExit(1)
    rec = json.loads(path.read_text())
    print(f"[record] {path}")
    print(format_diagnosis(rec))


if __name__ == "__main__":
    main()
