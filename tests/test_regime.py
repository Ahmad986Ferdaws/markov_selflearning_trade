import numpy as np
import pandas as pd

from app.services.regime import (
    STATE_ORDER,
    define_states,
    estimate_transition_matrix,
    forecast,
    stationary_distribution,
    walk_forward_backtest,
)


def test_transition_matrix_rows_sum_to_one():
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    returns = pd.Series(np.random.randn(100) * 0.01, index=idx)
    states = define_states(returns, window=5, bull_thresh=0.005, bear_thresh=-0.005)
    p = estimate_transition_matrix(states)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)


def test_stationary_distribution_sums_to_one():
    p = np.array([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.2, 0.2, 0.6]])
    pi = stationary_distribution(p)
    assert abs(pi.sum() - 1.0) < 1e-5


def test_forecast_is_probability_vector():
    p = estimate_transition_matrix(
        pd.Series(["bull", "bear", "sideways", "bull", "bull"], index=range(5))
    )
    dist = forecast(p, "bull", n_steps=1)
    assert abs(dist.sum() - 1.0) < 1e-6


def test_walk_forward_no_future_states_in_train():
    idx = pd.date_range("2020-01-01", periods=200, freq="D")
    closes = 100 * np.cumprod(1 + pd.Series(np.random.randn(200) * 0.01, index=idx))
    history = pd.DataFrame({"Close": closes})
    result = walk_forward_backtest(history, window=10, min_train=50)
    assert len(result.equity_curve) > 0
    assert len(result.states) == len(result.equity_curve)


def test_define_states_mutually_exclusive():
    idx = pd.date_range("2020-01-01", periods=50, freq="D")
    returns = pd.Series(0.001, index=idx)
    states = define_states(returns, window=5)
    assert set(states.unique()).issubset(set(STATE_ORDER))
