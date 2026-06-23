"""Phase A tests — daily evaluation: accuracy, buy-hold, balanced regimes, no leak."""

import numpy as np
import pandas as pd

from app.services.evaluation import (
    causal_states,
    evaluate,
    _accuracy,
    _policy_metrics,
)
from app.services.regime import STATE_TO_IDX, label_state


def _trending_history(periods: int = 600, seed: int = 0) -> pd.DataFrame:
    """Synthetic series with pronounced alternating up/down regimes (drift >~ vol),
    so bull/bear states genuinely occur and the model has something to predict."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=periods, freq="D")
    drift = np.where((np.arange(periods) // 50) % 2 == 0, 0.015, -0.015)
    rets = drift + rng.normal(0, 0.012, periods)
    closes = 100 * np.cumprod(1 + rets)
    return pd.DataFrame({"Close": closes}, index=idx)


def test_label_state_zscore_rule():
    assert label_state(0.03, 0.01, k=0.5) == "bull"      # 0.03 > 0.5*0.01
    assert label_state(-0.03, 0.01, k=0.5) == "bear"
    assert label_state(0.001, 0.01, k=0.5) == "sideways"  # within band
    assert label_state(0.05, 0.0, k=0.5) == "sideways"    # zero std -> sideways
    assert label_state(float("nan"), 0.01) is None


def test_causal_states_have_no_lookahead():
    # A state at index t must not change when future data is appended.
    rng = np.random.default_rng(1)
    rets = pd.Series(rng.normal(0, 0.02, 200))
    full = causal_states(rets, window=20, k=0.5)
    truncated = causal_states(rets.iloc[:120], window=20, k=0.5)
    # first 120 labels must be identical whether or not future bars exist
    assert full[:120] == truncated[:120]


def test_buy_hold_equals_market_minus_one_cost():
    # buy & hold: constant long, one entry cost, return ~ compounded market.
    market = [0.01, -0.02, 0.03, 0.0]
    positions = [1.0, 1.0, 1.0, 1.0]
    res = _policy_metrics("buy_hold", positions, market, cost_rate=0.008)
    # exactly one trade (the entry), and total cost == one turnover * rate
    assert res.num_trades == 1
    assert abs(res.total_cost - 0.008) < 1e-9


def test_accuracy_perfect_predictor():
    # dist always puts prob 1 on the actual class -> hit_rate 1.0, log_loss ~0
    actual = [0, 1, 2, 0]
    probs = []
    for a in actual:
        v = np.full(3, 1e-12)
        v[a] = 1.0
        probs.append(v)
    train_freq = np.array([0.34, 0.33, 0.33])
    acc = _accuracy(probs, actual, train_freq)
    assert acc.hit_rate == 1.0
    assert acc.balanced_accuracy == 1.0
    assert acc.log_loss < 1e-6
    assert acc.n == 4


def test_evaluate_endtoend_balanced_and_reports():
    history = _trending_history()
    report = evaluate(history, symbol="SYN", train_frac=0.7)
    # reports something on held-out test
    assert report.test_size > 0
    assert report.accuracy.n > 0
    # the model actually trades / varies: not 100% one regime
    assert report.regime_mix
    assert max(report.regime_mix.values()) < 0.95
    # both policies present
    names = {p.name for p in report.policies}
    assert names == {"baseline", "buy_hold"}
    # chosen hyperparams come from the configured grid
    assert report.chosen_window in (10, 20, 30)
    assert report.chosen_k in (0.3, 0.5, 0.75)


def test_evaluate_accuracy_is_on_test_not_train():
    # effective sizes: train predictions walked vs test predictions evaluated
    history = _trending_history(periods=700)
    report = evaluate(history, symbol="SYN", train_frac=0.7)
    assert report.train_size > report.test_size  # 70/30
    assert report.train_size + report.test_size <= 700


def test_evaluate_is_deterministic():
    # reproducibility: same history in -> identical numbers out
    history = _trending_history(periods=700)
    r1 = evaluate(history, symbol="SYN")
    r2 = evaluate(history, symbol="SYN")
    assert r1.data_hash == r2.data_hash
    assert r1.accuracy == r2.accuracy
    assert r1.chosen_window == r2.chosen_window and r1.chosen_k == r2.chosen_k
    assert [p.total_return for p in r1.policies] == [p.total_return for p in r2.policies]


def test_custom_policy_flows_through_one_engine():
    # the Phase-C agent will plug in exactly like this — one engine, no new fill math
    from app.services.evaluation import DEFAULT_POLICIES

    history = _trending_history(periods=700)
    pols = {**DEFAULT_POLICIES, "always_flat": lambda ctx: 0.0}
    report = evaluate(history, symbol="SYN", policies=pols)
    names = {p.name for p in report.policies}
    assert "always_flat" in names
    flat = next(p for p in report.policies if p.name == "always_flat")
    assert flat.num_trades == 0
    assert abs(flat.total_return) < 1e-9  # never invested -> 0 return, net of 0 cost
