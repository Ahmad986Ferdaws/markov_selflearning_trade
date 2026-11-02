"""kalman-cli — research commands. SIMULATION ONLY; no broker, no live orders.

  kalman-cli demo         seeded synthetic: known-beta recovery + constant-state
  kalman-cli pairs        qualification diagnostics (default SPY/QQQ, pinned 20y)
  kalman-cli walkforward  full walk-forward with all variants + report
  kalman-cli trend        local-linear-trend filter demo on synthetic data
  kalman-cli replay       one-observation-at-a-time replay == batch (paper mode)
"""

from __future__ import annotations

import argparse

import numpy as np

from .core import PairFilter, TrendFilter, run_pair_filter
from .data import (load_pair_from_snapshots, synthetic_constant_pair,
                   synthetic_pair, synthetic_trend)
from .ledger import LedgerConfig
from .pairs import qualify_pair
from .research import WalkForwardConfig, format_report, walk_forward


def demo() -> None:
    pair, truth = synthetic_pair()
    c1 = pair.p1["Close"].to_numpy(float)
    c2 = pair.p2["Close"].to_numpy(float)
    # correct-spec parameters (the generator's own q/r) — the demo shows what
    # the filter does when the model is right; tests also cover misspecification
    f = PairFilter.diffuse(q_beta=0.002 ** 2, q_alpha=0.0005 ** 2, r=0.08 ** 2)
    recs = run_pair_filter(c1, c2, f)
    beta_hat = np.array([r.x_post[0] for r in recs])
    err = np.abs(beta_hat[200:] - truth["beta"].to_numpy()[200:])
    print(f"[synthetic drifting beta] mean |beta_hat - beta_true| after warmup: "
          f"{err.mean():.4f} (max {err.max():.4f})")

    cpair, ctruth = synthetic_constant_pair()
    cc1 = cpair.p1["Close"].to_numpy(float)
    cc2 = cpair.p2["Close"].to_numpy(float)
    fc = PairFilter.diffuse(1e-8, 1e-9, 0.05)
    crecs = run_pair_filter(cc1, cc2, fc)
    tail = np.array([r.x_post[0] for r in crecs[-100:]])
    print(f"[constant state] beta true={ctruth['beta'].iloc[0]:.3f}  "
          f"filter tail mean={tail.mean():.3f}  sd={tail.std():.4f} (converged+stable)")
    print("NOTE: synthetic correctness demo — not empirical performance.")


def pairs(s1: str = "SPY", s2: str = "QQQ") -> None:
    pair = load_pair_from_snapshots(s1, s2)
    d = qualify_pair(pair, train_end=int(len(pair) * 0.5))
    print(f"=== Pair qualification (TRAIN half only): {s1}/{s2} ===")
    print(d.summary())
    print("(1 pair screened; diagnostics, not proof of profitability)")


def walkforward(s1: str = "SPY", s2: str = "QQQ") -> None:
    pair = load_pair_from_snapshots(s1, s2)
    res = walk_forward(pair, WalkForwardConfig(), LedgerConfig())
    print(format_report(res, f"{s1}/{s2}"))


def trend() -> None:
    y, truth = synthetic_trend()
    f = TrendFilter(1e-4, 1e-5, 0.25)
    vel_err = []
    for t, val in enumerate(y.to_numpy()):
        rec = f.step(float(val), dt=1.0)
        if t > 100:
            vel_err.append(abs(rec.x_post[1] - truth["velocity"].iloc[t]))
    print(f"[trend] mean |velocity_hat - velocity_true| after warmup: "
          f"{np.mean(vel_err):.4f}")
    print("Velocity crossings are a FEATURE to test against MA baselines, "
          "not an assumed edge.")


def replay() -> None:
    pair, _ = synthetic_pair(n=400)
    c1 = pair.p1["Close"].to_numpy(float)
    c2 = pair.p2["Close"].to_numpy(float)
    batch = run_pair_filter(c1, c2, PairFilter.diffuse(1e-5, 1e-6, 0.05))
    f2 = PairFilter.diffuse(1e-5, 1e-6, 0.05)
    rets = np.diff(c1, prepend=c1[0])
    max_diff = 0.0
    for t in range(len(c1)):
        rec = f2.step(float(c1[t]), float(c2[t]), past_returns=rets[:t])
        max_diff = max(max_diff, float(np.max(np.abs(rec.x_post - batch[t].x_post))))
    print(f"[replay] one-at-a-time vs batch max |state diff| = {max_diff:.2e} "
          f"({'AGREE' if max_diff < 1e-12 else 'DISAGREE'})")


COMMANDS = {"demo": demo, "pairs": pairs, "walkforward": walkforward,
            "trend": trend, "replay": replay}
PAIR_COMMANDS = ("pairs", "walkforward")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kalman-cli", description=__doc__, allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("command", nargs="?", default="demo", choices=sorted(COMMANDS),
                   help="research command (default: demo)")
    p.add_argument("symbols", nargs="*", metavar="SYMBOL",
                   help="pair for `pairs` / `walkforward` (default: SPY QQQ); "
                        "loaded from the pinned data/snapshots, never fetched live")
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in PAIR_COMMANDS:
        if len(args.symbols) > 2:
            parser.error(f"`{args.command}` takes at most two symbols, got {args.symbols}")
        try:
            COMMANDS[args.command](*args.symbols)
        except FileNotFoundError as error:   # pairs are read from pinned snapshots only
            parser.error(f"no pinned snapshot for that pair ({error}); nothing is fetched live")
    else:
        if args.symbols:
            parser.error(f"`{args.command}` takes no symbols, got {args.symbols}")
        COMMANDS[args.command]()


if __name__ == "__main__":
    main()
