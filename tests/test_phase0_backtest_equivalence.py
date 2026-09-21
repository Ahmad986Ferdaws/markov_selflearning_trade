"""walk_forward_backtest now labels the series ONCE and reads the causal label
positionally. It must equal the original per-step prefix re-labelling exactly
(the oracle below is that loop kept verbatim): same equity curve to the bit,
same state per day, same warm-up skipping, same sparse-cell warnings.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from app.services.data_cache import load_or_fetch
from app.services.regime import (
    _position_from_state,
    define_states,
    sparse_cell_warnings,
    walk_forward_backtest,
)


def _prefix_relabel_oracle(history, window, k, bull_thresh, bear_thresh,
                           fee_pct, slippage_pct, min_train):
    """The pre-optimisation loop: re-label returns[:t] from scratch every step."""
    returns = history["Close"].pct_change().dropna()
    cost_rate = (fee_pct + slippage_pct) / 100.0
    equity, prev = 1.0, 0.0
    equities, idx, states = [], [], []
    for t in range(min_train, len(returns)):
        train_states = define_states(returns.iloc[:t], window=window, k=k,
                                     bull_thresh=bull_thresh, bear_thresh=bear_thresh)
        if len(train_states) < 10:
            continue
        cur = train_states.iloc[-1]
        pos = _position_from_state(cur)
        cost = abs(pos - prev) * cost_rate
        equity *= 1.0 + (pos * float(returns.iloc[t]) - cost)
        equities.append(equity)
        idx.append(returns.index[t])
        states.append(cur)
        prev = pos
    return pd.Series(equities, index=idx), pd.Series(states, index=idx)


def _random_walk(periods: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-01", periods=periods, freq="D")
    closes = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, periods))
    return pd.DataFrame({"Close": closes}, index=idx)


def _assert_matches_oracle(history, **kw):
    bt = walk_forward_backtest(history, **kw)
    eq, st = _prefix_relabel_oracle(history, **kw)
    pd.testing.assert_series_equal(bt.equity_curve, eq, check_exact=True)
    pd.testing.assert_series_equal(bt.states, st, check_exact=True)
    return bt


def test_matches_prefix_relabel_oracle_on_synthetic():
    _assert_matches_oracle(_random_walk(400, seed=0), window=10, k=0.5,
                           bull_thresh=0.02, bear_thresh=-0.02,
                           fee_pct=0.3, slippage_pct=0.5, min_train=50)


def test_warmup_skipping_matches_when_min_train_precedes_labels():
    # min_train < window: the first steps have < 10 labelled days and are
    # skipped exactly as before; the first scored day is the same.
    bt = _assert_matches_oracle(_random_walk(300, seed=1), window=20, k=0.5,
                                bull_thresh=0.02, bear_thresh=-0.02,
                                fee_pct=0.0, slippage_pct=0.0, min_train=5)
    assert len(bt.returns) == 300 - 1 - (20 - 1) - 10   # returns - warm-up - 10 labelled


def test_pinned_btc_snapshot_matches_oracle_and_sparse_warnings():
    history, _ = load_or_fetch("BTC-USD", years=3)
    kw = dict(window=20, k=0.5, bull_thresh=0.02, bear_thresh=-0.02,
              fee_pct=0.3, slippage_pct=0.5, min_train=60)
    bt = _assert_matches_oracle(history, **kw)
    returns = history["Close"].pct_change().dropna()
    assert bt.sparse_warnings == sparse_cell_warnings(define_states(returns, window=20, k=0.5))
    assert bt.regime_mix == bt.states.value_counts(normalize=True).to_dict()


def test_inert_threshold_warning_fires_once_not_per_step(caplog):
    history = _random_walk(200, seed=2)
    with caplog.at_level(logging.WARNING, logger="app.services.regime"):
        walk_forward_backtest(history, window=10, bull_thresh=0.05, bear_thresh=-0.05, min_train=30)
    fired = [r for r in caplog.records if "IGNORED in zscore mode" in r.getMessage()]
    # the series is labelled once and reused for the sparse-cell pass; the old
    # loop re-labelled every step, so this warning fired ~n times per backtest
    assert len(fired) == 1
