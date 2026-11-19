"""A test z-std of exactly 0.0 is a number, not a missing value.

Two truthiness checks in kalman/research.py used to drop it: the aggregate's
`if m.get("z_std_test")` turned a fold-wide 0.0 into `mean_z_std = None`, and
`format_report` printed 'n/a' for it. A zero z-std is a degenerate signal the
report should show (and warn about), not hide.
"""

from __future__ import annotations

import numpy as np

from kalman import research
from kalman.data import synthetic_pair
from kalman.ledger import LedgerConfig
from kalman.research import WalkForwardConfig, format_report, walk_forward


def _result(mean_z_std):
    # Sharpe 0.37 and a real CI so that '0.00' / 'n/a' can only come from the z-std column.
    agg = {"folds": 1, "mean_test_sharpe": 0.37, "sharpe_ci90": (-0.1, 0.2),
           "total_test_return": 0.0, "total_costs": 0.0, "trades": 0,
           "gated_bars": 0, "mean_z_std": mean_z_std}
    return {"folds": [], "aggregate": {"static_ols": agg}}


def _row(report: str, variant: str) -> str:
    return next(ln for ln in report.splitlines() if ln.startswith(variant))


def _z_std_column(row: str) -> str:
    # variant folds sharpe CI90 total-ret trades gated z-std costs
    return row.split()[7]


def test_zero_z_std_prints_as_a_number():
    row = _row(format_report(_result(0.0), "SYN"), "static_ols")
    assert _z_std_column(row) == "0.00"
    assert "n/a" not in row


def test_missing_z_std_still_prints_na():
    row = _row(format_report(_result(None), "SYN"), "static_ols")
    assert _z_std_column(row) == "n/a"
    assert "0.00" not in row


def test_walk_forward_aggregate_keeps_a_zero_z_std(monkeypatch):
    def constant_signal(name, y, x, *args, **kwargs):
        n = len(y)
        return np.zeros(n), np.ones(n), None          # z identically 0 -> z_std_test == 0.0

    monkeypatch.setattr(research, "_variant_signals", constant_signal)
    pair, _ = synthetic_pair(n=640, seed=4)
    res = walk_forward(pair, WalkForwardConfig(train=400, validation=120, test=120),
                       LedgerConfig())
    for v in research.VARIANTS:
        assert res["by_variant"][v][0]["z_std_test"] == 0.0
        assert res["aggregate"][v]["mean_z_std"] == 0.0     # was None before the fix

    report = format_report(res, "SYN")
    for v in research.VARIANTS:
        assert _z_std_column(_row(report, v)) == "0.00"
    # the scale caveat now fires on the degenerate signal instead of being skipped on None
    assert "kalman_fixed innovation z-std is 0.00" in report
