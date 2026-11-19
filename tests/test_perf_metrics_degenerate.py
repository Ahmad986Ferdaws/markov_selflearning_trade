"""Pins for kalman.research.perf_metrics and sharpe_ci_block_bootstrap on
degenerate inputs: too few bars, a non-positive starting equity, a flat equity
curve, the entry counter on executed gross, and the bootstrap's minimum-length
and seeding contracts.

Sortino is deliberately not pinned here: its definition is under separate
review and this file must not freeze it.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from kalman.research import perf_metrics, sharpe_ci_block_bootstrap


def _ledger(equity, gross=None) -> pd.DataFrame:
    """The columns perf_metrics reads, built the way run_ledger builds them."""
    eq = pd.Series(np.asarray(equity, dtype=float))
    n = len(eq)
    return pd.DataFrame({
        "equity": eq,
        "ret": eq.pct_change().fillna(0.0),
        "gross": np.zeros(n) if gross is None else np.asarray(gross, dtype=float),
        "turnover": np.zeros(n),
        "costs": np.zeros(n),
    })


@pytest.mark.parametrize("n", [0, 1, 2])
def test_fewer_than_three_bars_returns_only_the_count(n):
    assert perf_metrics(_ledger([100.0] * n)) == {"n": n}


@pytest.mark.parametrize("eq0", [0.0, -1.0])
def test_non_positive_starting_equity_returns_only_the_count(eq0):
    assert perf_metrics(_ledger([eq0, 1.0, 2.0, 3.0])) == {"n": 4}


def test_flat_equity_curve_has_zero_sharpe_drawdown_return_and_trades():
    m = perf_metrics(_ledger([100_000.0] * 10))
    assert m["n"] == 10
    assert m["sharpe"] == 0.0                        # zero std: defined as 0, not nan
    assert m["max_drawdown"] == 0.0
    assert m["trades"] == 0
    assert m["total_return"] == 0.0
    assert m["cagr"] == 0.0 and m["vol"] == 0.0
    assert math.isnan(m["calmar"])                   # no drawdown: undefined


def test_trades_count_flat_to_invested_transitions_in_executed_gross():
    assert perf_metrics(_ledger([100.0] * 6, gross=[0, 0, 1, 1, 0, 1]))["trades"] == 2
    # the first bar compares against a flat start, so gross > 0 there is an entry
    assert perf_metrics(_ledger([100.0] * 3, gross=[1, 1, 1]))["trades"] == 1
    # a re-size while invested is not a new entry
    assert perf_metrics(_ledger([100.0] * 4, gross=[0, 1, 2, 1]))["trades"] == 1


def test_bootstrap_needs_at_least_two_blocks():
    rets = np.random.default_rng(0).normal(0.0, 0.01, 40)
    short = sharpe_ci_block_bootstrap(rets[:39], block=20)
    assert math.isnan(short[0]) and math.isnan(short[1])
    lo, hi = sharpe_ci_block_bootstrap(rets[:40], n_boot=50, block=20)
    assert math.isfinite(lo) and math.isfinite(hi)


def test_seeded_bootstrap_is_reproducible_and_ordered():
    rets = np.random.default_rng(0).normal(0.0, 0.01, 200)
    a = sharpe_ci_block_bootstrap(rets, n_boot=200, block=20, seed=1)
    b = sharpe_ci_block_bootstrap(rets, n_boot=200, block=20, seed=1)
    assert a == b
    lo, hi = a
    assert type(lo) is float and type(hi) is float
    assert math.isfinite(lo) and math.isfinite(hi) and lo <= hi
