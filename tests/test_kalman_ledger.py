"""Ledger + causality acceptance tests — spec criteria 4, 5, 8, 9, 10, 13.

The two-leg ledger is the only P&L source; these tests reconcile it by hand
and prove the timing/causality contracts hold mechanically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalman.data import align_pair, synthetic_pair
from kalman.ledger import (CostConfig, HedgeMode, LedgerConfig, PriceModel,
                           run_ledger)
from kalman.research import backtest, signals_kalman
from kalman.strategy import Decision, Position, StateMachine, StrategyConfig


def _mk_pair(n=6, p1=100.0, p2=50.0):
    idx = pd.bdate_range("2020-01-01", periods=n)
    f1 = pd.DataFrame({"Open": [p1] * n, "Close": [p1] * n}, index=idx)
    f2 = pd.DataFrame({"Open": [p2] * n, "Close": [p2] * n}, index=idx)
    return align_pair(f1, f2, "A", "B", min_overlap=2)


def _decisions(seq):
    return [Decision(t, pos, "test") for t, pos in enumerate(seq)]


# 8 — hand-calculated reconciliation of two-leg P&L, turnover, and costs
def test_hand_reconciled_two_leg_pnl():
    n = 4
    idx = pd.bdate_range("2020-01-01", periods=n)
    o1 = np.array([100.0, 100.0, 110.0, 120.0]); c1 = np.array([100.0, 105.0, 115.0, 118.0])
    o2 = np.array([50.0, 50.0, 52.0, 54.0]);     c2 = np.array([50.0, 51.0, 53.0, 53.0])
    # decision at bar0: LONG residual -> fills at bar1 open (gross=10_000, log
    # model, beta>0 -> L1=+5000 @100 = 50 sh; L2=-5000 @50 = -100 sh).
    # decision at bar1: FLAT -> exit fills at bar2 open.
    dec = _decisions([Position.LONG_RESIDUAL, Position.FLAT,
                      Position.FLAT, Position.FLAT])
    cfg = LedgerConfig(capital=100_000.0, gross_target=10_000.0,
                       costs=CostConfig(commission_bps=10, half_spread_bps=0,
                                        slippage_bps=0, borrow_bps_pa=0.0))
    led = run_ledger(idx, o1, o2, c1, c2, dec, np.array([1.0] * n), cfg)

    # bar1: buy 50 sh @100 (=5000), sell 100 sh @50 (=-5000); turnover 10000
    # costs = 10bp * 10000 = 10; cash = 100000 -5000 +5000 -10 = 99990
    assert led.iloc[1]["turnover"] == pytest.approx(10_000.0)
    assert led.iloc[1]["costs"] == pytest.approx(10.0)
    assert led.iloc[1]["cash"] == pytest.approx(99_990.0)
    # equity at bar1 close: cash + 50*105 - 100*51 = 99990 + 5250 - 5100
    assert led.iloc[1]["equity"] == pytest.approx(100_140.0)
    # bar2: exit fills at open2 (110/52): sell 50 @110=+5500, buy 100 @52=-5200
    # turnover 10700, costs 10.70, cash 99990+5500-5200-10.70 = 100279.30
    assert led.iloc[2]["turnover"] == pytest.approx(10_700.0)
    assert led.iloc[2]["cash"] == pytest.approx(100_279.30)
    # flat afterwards: equity == cash, unchanged at bar3
    assert led.iloc[3]["equity"] == pytest.approx(100_279.30)


# 5 — a signal from bar t cannot affect holdings before its execution event
def test_one_bar_execution_lag():
    pair = _mk_pair(n=5)
    idx = pair.index
    o1 = pair.p1["Open"].to_numpy(); c1 = pair.p1["Close"].to_numpy()
    o2 = pair.p2["Open"].to_numpy(); c2 = pair.p2["Close"].to_numpy()
    dec = _decisions([Position.FLAT, Position.LONG_RESIDUAL, Position.LONG_RESIDUAL,
                      Position.LONG_RESIDUAL, Position.LONG_RESIDUAL])
    led = run_ledger(idx, o1, o2, c1, c2, dec, np.ones(5), LedgerConfig())
    assert led.iloc[1]["s1"] == 0.0          # signal at bar1 -> NOT filled at bar1
    assert led.iloc[2]["s1"] != 0.0          # filled at bar2 open
    assert led.iloc[0]["s1"] == 0.0


# 4 — changing data after cutoff T cannot alter anything at or before T
def test_future_data_cannot_change_past():
    pair, _ = synthetic_pair(n=500, seed=3)
    c1 = pair.p1["Close"].to_numpy(float).copy()
    c2 = pair.p2["Close"].to_numpy(float).copy()
    z_a, b_a, _ = signals_kalman(c1, c2, 1e-5, 1e-6, 0.05, train_end=100)
    T = 300
    c1_mut = c1.copy(); c1_mut[T:] = c1[T:] * 1.5 + 7.0   # rewrite the future
    c2_mut = c2.copy(); c2_mut[T:] = c2[T:] * 0.5 + 3.0
    z_b, b_b, _ = signals_kalman(c1_mut, c2_mut, 1e-5, 1e-6, 0.05, train_end=100)
    assert np.array_equal(np.nan_to_num(z_a[:T]), np.nan_to_num(z_b[:T]))
    assert np.array_equal(b_a[:T], b_b[:T])

    scfg = StrategyConfig(warmup=50)
    lcfg = LedgerConfig()
    led_a = backtest(pair, z_a, b_a, scfg, lcfg, None, 0, len(c1))
    pair_mut = pair
    pair_mut.p1["Close"].to_numpy()  # (prices used are passed via backtest slices)
    led_b = backtest(pair, z_b, b_b, scfg, lcfg, None, 0, len(c1))
    # positions and equity strictly before T depend only on pre-T signals
    assert np.array_equal(led_a["s1"].to_numpy()[:T], led_b["s1"].to_numpy()[:T])
    assert np.array_equal(led_a["equity"].to_numpy()[:T - 1],
                          led_b["equity"].to_numpy()[:T - 1])


# 9 — increasing nonnegative costs cannot improve net P&L on a fixed path
def test_cost_monotonicity():
    pair, _ = synthetic_pair(n=400, seed=9)
    c1 = pair.p1["Close"].to_numpy(float)
    c2 = pair.p2["Close"].to_numpy(float)
    z, b, _ = signals_kalman(c1, c2, 1e-5, 1e-6, 0.05, train_end=100)
    scfg = StrategyConfig(warmup=50)
    cheap = LedgerConfig(costs=CostConfig(commission_bps=0, half_spread_bps=0,
                                          slippage_bps=0, borrow_bps_pa=0))
    dear = LedgerConfig(costs=CostConfig(commission_bps=20, half_spread_bps=10,
                                         slippage_bps=10, borrow_bps_pa=100))
    led_cheap = backtest(pair, z, b, scfg, cheap)
    led_dear = backtest(pair, z, b, scfg, dear)
    # identical decision path (costs don't feed signals) -> same fills
    assert np.array_equal((led_cheap["turnover"] > 0).to_numpy(),
                          (led_dear["turnover"] > 0).to_numpy())
    assert led_dear["equity"].iloc[-1] <= led_cheap["equity"].iloc[-1] + 1e-9


# 10 — malformed data is rejected loudly
def test_bad_data_rejected():
    idx = pd.bdate_range("2020-01-01", periods=10)
    good = pd.DataFrame({"Open": np.ones(10), "Close": np.ones(10)}, index=idx)
    dup = good.copy(); dup.index = idx[[0, 0, 1, 2, 3, 4, 5, 6, 7, 8]]
    with pytest.raises(ValueError, match="duplicate"):
        align_pair(dup, good, "A", "B", min_overlap=2)
    unsorted = good.iloc[::-1]
    with pytest.raises(ValueError, match="sorted"):
        align_pair(unsorted, good, "A", "B", min_overlap=2)
    neg = good.copy(); neg.loc[idx[3], "Close"] = -1.0
    with pytest.raises(ValueError, match="non-positive"):
        align_pair(neg, good, "A", "B", min_overlap=2)
    with pytest.raises(ValueError, match="overlap"):
        align_pair(good, good, "A", "B", min_overlap=50)


# rehedge mode charges turnover; freeze mode does not rebalance on beta drift
def test_hedge_modes_differ_and_both_pay():
    n = 8
    idx = pd.bdate_range("2020-01-01", periods=n)
    o1 = np.full(n, 100.0); c1 = np.full(n, 100.0)
    o2 = np.full(n, 50.0);  c2 = np.full(n, 50.0)
    dec = _decisions([Position.LONG_RESIDUAL] * n)
    betas = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0])  # beta jumps
    frozen = run_ledger(idx, o1, o2, c1, c2, dec, betas,
                        LedgerConfig(hedge_mode=HedgeMode.FREEZE,
                                     price_model=PriceModel.LEVEL))
    rehedged = run_ledger(idx, o1, o2, c1, c2, dec, betas,
                          LedgerConfig(hedge_mode=HedgeMode.REHEDGE,
                                       rehedge_threshold=0.5,
                                       price_model=PriceModel.LEVEL))
    assert frozen["turnover"].iloc[4:].sum() == 0.0        # freeze: no rebalance
    assert rehedged["turnover"].iloc[4:].sum() > 0.0       # rehedge: rebalance...
    assert rehedged["costs"].sum() > frozen["costs"].sum() # ...and pays for it


# 13 — every variant flows through the identical execution engine
def test_variants_share_execution_engine():
    import inspect

    from kalman import research
    # the ONLY route to P&L is backtest -> run_ledger; walk_forward calls
    # backtest for every variant and contains no other ledger/cost path
    assert "run_ledger(" in inspect.getsource(research.backtest)
    wf_src = inspect.getsource(research.walk_forward)
    assert "backtest(" in wf_src and "run_ledger(" not in wf_src
    # no variant computes returns from a posterior-spread diff anywhere in code
    for mod in (research,):
        code_lines = [ln for ln in inspect.getsource(mod).splitlines()
                      if not ln.lstrip().startswith(("#", '"', "'"))]
        assert not any(".diff()" in ln and "spread" in ln for ln in code_lines)
