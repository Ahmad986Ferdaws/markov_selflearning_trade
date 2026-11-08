"""Phase 0 CLI — the standalone regime report on the pinned snapshot.

Usage:
  regime-cli                          # settings from .env (REGIME_SYMBOL, REGIME_WINDOW, REGIME_K, ...)
  regime-cli --symbol ETH-USD         # one-off override; .env is untouched
  regime-cli --window 30 --k 0.2      # the headline's chosen config
  regime-cli --symbol SPY --years 20  # the 20-year pinned snapshot

Every flag overrides the matching setting for this invocation only. The report
reads the committed snapshot in data/snapshots/ when one exists.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence

from app.config import get_settings
from app.services.regime import run_phase0_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="regime-cli",
        description="Phase 0 Markov regime report. Simulation only; reads the pinned snapshot.",
        allow_abbrev=False,
    )
    parser.add_argument("--symbol", help="ticker to report on (default: REGIME_SYMBOL)")
    parser.add_argument("--window", type=int, help="trailing window in days (default: REGIME_WINDOW)")
    parser.add_argument("--k", type=float, help="z-score multiple for bull/bear (default: REGIME_K)")
    parser.add_argument("--fee-pct", type=float, help="one-way fee in percent (default: FEE_PCT)")
    parser.add_argument("--slippage-pct", type=float,
                        help="one-way slippage in percent (default: SLIPPAGE_PCT)")
    parser.add_argument("--years", type=int,
                        help="history length in years; selects the matching pinned snapshot "
                             "(default: 3)")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    # argparse's float() happily accepts "nan" and "inf"; neither survives a
    # comparison, so each would slip past a plain range check and reach the
    # labelling rule, where NaN makes every day sideways without a word.
    if args.window is not None and args.window < 2:
        parser.error("--window must be at least 2 (a rolling mean needs two bars)")
    if args.k is not None and not (math.isfinite(args.k) and args.k >= 0):
        parser.error("--k must be a finite, non-negative number (it multiplies a standard deviation)")
    for name in ("fee_pct", "slippage_pct"):
        value = getattr(args, name)
        if value is not None and not (math.isfinite(value) and 0 <= value < 100):
            parser.error(f"--{name.replace('_', '-')} must be a finite percentage in [0, 100)")
    if args.years is not None and args.years < 1:
        parser.error("--years must be at least 1 (it names the pinned snapshot's history length)")
    s = get_settings()
    # Only forwarded when given, so the report keeps its own default (3 years).
    years = {} if args.years is None else {"years": args.years}
    print(
        run_phase0_report(
            symbol=args.symbol or s.regime_symbol,
            window=s.regime_window if args.window is None else args.window,
            k=s.regime_k if args.k is None else args.k,
            bull_thresh=s.regime_bull_thresh,
            bear_thresh=s.regime_bear_thresh,
            fee_pct=s.fee_pct if args.fee_pct is None else args.fee_pct,
            slippage_pct=s.slippage_pct if args.slippage_pct is None else args.slippage_pct,
            **years,
        )
    )


if __name__ == "__main__":
    main()
