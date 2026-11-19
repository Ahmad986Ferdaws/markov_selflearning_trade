"""Offline run-record diagnosis — no model, no network, just the receipt.

Usage:
  diagnose-cli                       # newest record in results/
  diagnose-cli results/BTC-USD_4a150b23.json
  diagnose-cli results/long_history/SPY.json   # study envelope {"summary", "report"}

Reads a deterministic run record written by evaluate-cli and narrates what it
shows: the edge (or absence of one) vs the persistence bar, the trading
outcomes net of costs, and every honesty-gate warning the engine raised. This
is the article-inspired "diagnose" mode kept honest: it only ever READS the
already-computed record — it cannot re-derive market state, so it is
lookahead-safe by construction.

It is equally honest about its input: anything that is not a run record with
held-out predictions is refused with a one-line message naming the file and
exit status 1 (usage errors exit 2, as argparse does).
It used to default every missing field to zero and narrate "IDENTICAL to
persistence" over an empty dict, a study summary, or a JSON array.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

RESULTS_DIR = Path("results")
_REQUIRED_ACCURACY = ("hit_rate", "persistence_hit_rate", "balanced_accuracy", "persistence_balanced")


class NotARunRecord(ValueError):
    """The input is not an evaluate-cli run record (or has nothing to diagnose)."""


def _latest_record(results_dir: Path = RESULTS_DIR) -> Path | None:
    records = sorted(
        (p for p in results_dir.glob("*.json")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return records[0] if records else None


def as_run_record(rec) -> dict:
    """Return the run record inside `rec`, or raise NotARunRecord.

    Accepts a raw evaluate-cli record, or the `{"summary": ..., "report": ...}`
    envelope the robustness / long-history studies write (the report inside is
    a run record). Refuses anything without an `accuracy` block, without the
    four edge fields, or with no held-out predictions — a diagnosis over zeros
    is a fabricated verdict, not a receipt.
    """
    if isinstance(rec, dict) and "accuracy" not in rec and isinstance(rec.get("report"), dict):
        rec = rec["report"]
    if not isinstance(rec, dict) or not isinstance(rec.get("accuracy"), dict):
        raise NotARunRecord("not an evaluate-cli run record (no 'accuracy' block)")
    acc = rec["accuracy"]
    missing = [k for k in _REQUIRED_ACCURACY if k not in acc]
    if missing:
        raise NotARunRecord(f"run record is missing accuracy field(s): {', '.join(missing)}")
    for key in _REQUIRED_ACCURACY:
        if not _is_finite_number(acc[key]):
            raise NotARunRecord(f"run record accuracy.{key} must be a finite number, got {acc[key]!r}")
    n = acc.get("n", 0)
    # a genuine count: JSON int, not a float to truncate, not a bool, not "328"
    if isinstance(n, bool) or not isinstance(n, int):
        raise NotARunRecord(f"run record 'accuracy.n' must be an integer count, got {n!r}")
    if n <= 0:
        raise NotARunRecord("run record has no held-out predictions (accuracy.n = 0); nothing to diagnose")
    if "symbol" in rec and not isinstance(rec["symbol"], str):
        raise NotARunRecord("run record 'symbol' must be a string")
    pols = rec.get("policies", [])
    if not isinstance(pols, list) or any(not isinstance(p, dict) for p in pols):
        raise NotARunRecord("run record 'policies' must be a list of objects")
    for p in pols:
        if "name" in p and not isinstance(p["name"], str):
            raise NotARunRecord("run record policy 'name' must be a string")
        for key in ("total_return", "sharpe"):
            if key in p and not _is_finite_number(p[key]):
                raise NotARunRecord(f"run record policy {p.get('name', '?')!r}.{key} must be a finite number")
    warns = rec.get("warnings", [])
    if not isinstance(warns, list) or any(not isinstance(w, str) for w in warns):
        raise NotARunRecord("run record 'warnings' must be a list of strings")
    return rec


def _is_finite_number(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:          # a JSON integer too large for a float
        return False


def load_run_record(path: Path) -> dict:
    """Read + validate a record file; every failure becomes a NotARunRecord.

    Messages do not repeat the path; `main` prefixes it once for every refusal.
    """
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as error:
        raise NotARunRecord(f"cannot read file ({error.strerror or error})") from error
    except UnicodeDecodeError as error:
        raise NotARunRecord("not a UTF-8 text file") from error
    except json.JSONDecodeError as error:
        raise NotARunRecord(f"not valid JSON ({error.msg} at line {error.lineno})") from error
    return as_run_record(raw)


def format_diagnosis(rec: dict) -> str:
    """Pure formatter: run-record dict -> human diagnosis. Testable offline.

    Raises NotARunRecord (a ValueError) instead of narrating over missing data.
    """
    rec = as_run_record(rec)
    acc = rec["accuracy"]
    hit = float(acc["hit_rate"])
    per = float(acc["persistence_hit_rate"])
    bal = float(acc["balanced_accuracy"])
    pbal = float(acc["persistence_balanced"])
    n = int(acc["n"])
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
    elif edge_bal < -0.02:
        # a large negative edge is not "noise": the model diverged from the
        # trivial bar and lost (docs/16 records early-BTC at -0.167 this way)
        lines.append("  -> WORSE than persistence: the model diverged from the trivial bar and lost")
        lines.append("     accuracy doing so. Verify pointwise; this is a real negative result.")
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="diagnose-cli",
        description="Narrate an evaluate-cli run record offline: edge vs persistence, "
                    "trading outcomes, honesty gates. Reads JSON only; no model, no network.",
    )
    parser.add_argument(
        "record", nargs="?", type=Path,
        help="path to a results/*.json run record (default: newest file in results/)",
    )
    args = parser.parse_args(argv)
    path = args.record if args.record is not None else _latest_record(RESULTS_DIR)
    if path is None or not path.exists():
        print("diagnose-cli: no run record found (run evaluate-cli first, "
              "or pass a path to a results/*.json).", file=sys.stderr)
        raise SystemExit(1)
    try:
        rec = load_run_record(path)
    except NotARunRecord as error:
        print(f"diagnose-cli: {path}: {error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"[record] {path}")
    print(format_diagnosis(rec))


if __name__ == "__main__":
    main()
