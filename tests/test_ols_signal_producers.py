"""Pins for the three OLS signal producers in kalman.research: causality,
warm-up NaNs and the static beta.

signals_static_ols, signals_rolling_ols and signals_ew_ols are the baselines
every Kalman verdict is compared against; until now the only test touching
them monkeypatched _variant_signals away. Each producer must (a) fit on data
strictly before bar t (the static one on the training window only), (b) hold
NaN until its windows are full, and (c) leave every earlier output untouched
when a later observation is perturbed. The dispatcher's unknown-name and
'cash' branches are pinned at the end.
"""

from __future__ import annotations

import numpy as np
import pytest

from kalman.research import (
    _variant_signals,
    signals_ew_ols,
    signals_rolling_ols,
    signals_static_ols,
)

N, TRAIN_END, WINDOW, Z_WINDOW = 300, 200, 120, 60


def _pair(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0.0, 0.01, N)) + 4.0
    y = 2.0 * x + 1.0 + rng.normal(0.0, 0.01, N)
    return y, x


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return np.array_equal(a, b, equal_nan=True)


# --- static OLS -----------------------------------------------------------------------


def test_static_beta_is_one_value_equal_to_the_training_window_fit():
    y, x = _pair()
    z, beta = signals_static_ols(y, x, train_end=TRAIN_END)
    assert len(z) == len(beta) == N
    assert np.unique(beta).size == 1
    assert beta[0] == pytest.approx(np.polyfit(x[:TRAIN_END], y[:TRAIN_END], 1)[0], rel=1e-9)


def test_static_z_is_nan_for_the_standardisation_window_then_finite():
    y, x = _pair()
    z, _ = signals_static_ols(y, x, train_end=TRAIN_END)
    assert np.isnan(z[:Z_WINDOW]).all()
    assert np.isfinite(z[Z_WINDOW:]).all()


def test_static_z_is_standardised_by_the_trailing_window_ending_at_t_minus_one():
    y, x = _pair()
    z, _ = signals_static_ols(y, x, train_end=TRAIN_END)
    slope, intercept = np.polyfit(x[:TRAIN_END], y[:TRAIN_END], 1)
    resid = y - (slope * x + intercept)
    past = resid[:Z_WINDOW]                        # bars 0..59, never bar 60 itself
    expected = (resid[Z_WINDOW] - past.mean()) / past.std(ddof=1)
    assert z[Z_WINDOW] == pytest.approx(expected, rel=1e-9)


def test_static_fit_is_training_only_so_later_bars_cannot_move_it():
    y, x = _pair()
    z, beta = signals_static_ols(y, x, train_end=TRAIN_END)
    y2 = y.copy()
    y2[TRAIN_END:] += 10.0                        # every post-training bar shifts
    z2, beta2 = signals_static_ols(y2, x, train_end=TRAIN_END)
    assert np.array_equal(beta, beta2)
    assert _same(z[:TRAIN_END], z2[:TRAIN_END])
    assert z2[TRAIN_END] != z[TRAIN_END]          # the shifted bars themselves do differ


def test_static_train_start_moves_the_fit_window():
    y, x = _pair()
    _, full = signals_static_ols(y, x, train_end=TRAIN_END)
    _, late = signals_static_ols(y, x, train_end=TRAIN_END, train_start=100)
    assert late[0] == pytest.approx(np.polyfit(x[100:TRAIN_END], y[100:TRAIN_END], 1)[0], rel=1e-9)
    assert late[0] != full[0]


# --- rolling OLS ----------------------------------------------------------------------


def test_rolling_beta_is_nan_for_exactly_one_window_then_finite():
    y, x = _pair()
    z, beta = signals_rolling_ols(y, x, window=WINDOW)
    assert np.isnan(beta[:WINDOW]).all()
    assert np.isfinite(beta[WINDOW:]).all()
    # the residual starts at bar `window`; the z-window adds another 60 NaN bars
    assert np.isnan(z[:WINDOW + Z_WINDOW]).all()
    assert np.isfinite(z[WINDOW + Z_WINDOW:]).all()


def test_rolling_fit_window_ends_at_t_minus_one():
    y, x = _pair()
    z, beta = signals_rolling_ols(y, x, window=WINDOW)
    yp = y.copy()
    yp[250] += 5.0
    zp, betap = signals_rolling_ols(yp, x, window=WINDOW)
    assert _same(z[:250], zp[:250])
    assert _same(beta[:251], betap[:251])         # bar 250's own fit does not see y[250]
    assert betap[251] != beta[251]                # ...but the next bar's fit does
    assert zp[250] != z[250]                      # the residual at 250 moves with y[250]


def test_rolling_beta_matches_a_direct_lagged_fit():
    y, x = _pair()
    _, beta = signals_rolling_ols(y, x, window=WINDOW)
    t = 200
    direct = np.polyfit(x[t - WINDOW:t], y[t - WINDOW:t], 1)[0]
    assert beta[t] == pytest.approx(direct, rel=1e-9)


# --- exponentially weighted RLS -------------------------------------------------------


def test_ew_starts_from_a_zero_state_and_warms_up_the_z_window():
    y, x = _pair()
    z, beta = signals_ew_ols(y, x)
    assert beta[0] == 0.0
    assert np.isfinite(beta).all()
    assert np.isnan(z[:Z_WINDOW]).all()
    assert np.isfinite(z[Z_WINDOW:]).all()


def test_ew_residual_uses_the_state_fitted_through_t_minus_one():
    y, x = _pair()
    z, beta = signals_ew_ols(y, x)
    yp = y.copy()
    yp[250] += 5.0
    zp, betap = signals_ew_ols(yp, x)
    assert _same(z[:250], zp[:250])
    assert _same(beta[:251], betap[:251])
    assert betap[251] != beta[251]
    assert zp[250] != z[250]


def test_ew_converges_toward_the_true_slope():
    y, x = _pair()
    _, beta = signals_ew_ols(y, x)
    assert abs(beta[-1] - 2.0) < 0.1


# --- dispatcher -----------------------------------------------------------------------


def test_unknown_variant_raises_its_name():
    y, x = _pair()
    with pytest.raises(ValueError) as excinfo:
        _variant_signals("bogus", y, x, TRAIN_END, 1e-6, 1e-7, 1e-3, 0.01)
    assert str(excinfo.value) == "bogus"


def test_cash_variant_is_always_stale_and_flat():
    y, x = _pair()
    z, beta, recs = _variant_signals("cash", y, x, TRAIN_END, 1e-6, 1e-7, 1e-3, 0.01)
    assert len(z) == N and np.isnan(z).all()
    assert np.array_equal(beta, np.zeros(N))
    assert recs is None


def test_dispatcher_routes_the_three_ols_names_to_their_producers():
    y, x = _pair()
    for name, direct in (("static_ols", signals_static_ols(y, x, TRAIN_END)),
                         ("rolling_ols", signals_rolling_ols(y, x)),
                         ("ew_ols", signals_ew_ols(y, x))):
        z, beta, recs = _variant_signals(name, y, x, TRAIN_END, 1e-6, 1e-7, 1e-3, 0.01)
        assert _same(z, direct[0]) and _same(beta, direct[1]) and recs is None
