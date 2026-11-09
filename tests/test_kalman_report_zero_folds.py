"""The walk-forward report must say when nothing was evaluated.

A pair shorter than train + validation + test fits no fold, so `walk_forward`
returns `folds == []` and `aggregate == {}`. `format_report` used to fall
through to the 'test Sharpe is positive' verdict on that empty table.
"""

from __future__ import annotations

from kalman.data import synthetic_pair
from kalman.ledger import LedgerConfig
from kalman.research import WalkForwardConfig, format_report, walk_forward

WF = WalkForwardConfig(train=400, validation=120, test=120)   # one fold needs 640 bars


def test_short_history_yields_no_fold_and_an_empty_aggregate():
    pair, _ = synthetic_pair(n=300)
    res = walk_forward(pair, WF, LedgerConfig())
    assert res["folds"] == []
    assert res["aggregate"] == {}


def test_report_on_zero_folds_says_nothing_was_evaluated():
    pair, _ = synthetic_pair(n=300)
    report = format_report(walk_forward(pair, WF, LedgerConfig()), "SYN")
    assert "folds: 0" in report
    assert "no complete train/validation/test fold" in report
    assert "nothing was evaluated" in report
    assert "positive" not in report
    assert "no robust edge" not in report
    assert "variant" not in report                 # the empty table is skipped too
    assert report.count("VERDICT") == 1


def test_one_bar_short_of_a_fold_is_the_zero_fold_boundary():
    short, _ = synthetic_pair(n=639, seed=4)
    exact, _ = synthetic_pair(n=640, seed=4)
    res_short = walk_forward(short, WF, LedgerConfig())
    res_exact = walk_forward(exact, WF, LedgerConfig())
    assert res_short["aggregate"] == {}
    assert len(res_exact["folds"]) == 1

    short_report = format_report(res_short, "SYN")
    exact_report = format_report(res_exact, "SYN")
    assert "nothing was evaluated" in short_report
    # one complete fold -> the ordinary table and verdict, untouched by the new branch
    assert "nothing was evaluated" not in exact_report
    assert "variant" in exact_report and "kalman_fixed" in exact_report
    assert "VERDICT" in exact_report
