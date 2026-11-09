"""walk_forward refuses a pair with a missing close instead of silently
flattening the baselines.

kalman/data.py lets interior NaNs survive alignment because the Kalman filter
handles a missing observation as a predict-only step. The static-OLS and EW
baselines do not: one NaN inside the training window makes their z-stream
all-NaN, so the stale-data gate holds them flat for the whole test period and
the report shows a cash-like baseline row with no warning. walk_forward now
raises before any fold is built.

tests/test_kalman_report_zero_folds.py pins that a clean 300-bar pair yields
no fold under the 400/120/120 layout; that path is re-asserted here so the new
check is the only thing that changed for a gappy pair of the same length.
"""

from __future__ import annotations

import numpy as np
import pytest

from kalman.data import synthetic_pair
from kalman.ledger import LedgerConfig
from kalman.research import WalkForwardConfig, walk_forward

WF = WalkForwardConfig(train=400, validation=120, test=120)


def test_one_missing_close_on_one_leg_is_refused_before_any_fold():
    pair, _ = synthetic_pair(n=300)
    pair.p1.loc[pair.index[3], "Close"] = np.nan
    with pytest.raises(ValueError, match="missing close") as exc:
        walk_forward(pair, WF, LedgerConfig())
    assert "1 bar(s)" in str(exc.value)


def test_missing_closes_on_both_legs_are_counted_once_per_bar():
    pair, _ = synthetic_pair(n=300)
    pair.p1.loc[pair.index[3], "Close"] = np.nan
    pair.p2.loc[pair.index[3], "Close"] = np.nan       # same bar on the other leg
    pair.p2.loc[pair.index[7], "Close"] = np.nan
    with pytest.raises(ValueError, match=r"walk_forward: 2 bar\(s\) have a missing close"):
        walk_forward(pair, WF, LedgerConfig())


def test_a_clean_short_pair_still_yields_no_fold():
    pair, _ = synthetic_pair(n=300)
    res = walk_forward(pair, WF, LedgerConfig())
    assert res["folds"] == []
    assert res["aggregate"] == {}
