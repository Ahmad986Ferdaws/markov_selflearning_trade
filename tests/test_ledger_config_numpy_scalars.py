"""kalman.ledger's config validation accepts numpy integer and float scalars.

Cost and sizing parameters usually arrive from numpy grids (np.int64,
np.float64), and kalman.core already accepts them (_finite_real, AdaptiveQ's
window). The ledger used to reject them as 'must be a real number' /
'must be a positive integer'. bool and np.bool_ stay rejected, non-finite
values stay rejected, and bars_per_year is normalised to a plain int exactly
as AdaptiveQ normalises `window`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalman.ledger import CostConfig, LedgerConfig, run_ledger
from kalman.strategy import Decision, Position


def test_numpy_int64_bars_per_year_is_accepted_and_normalised_to_int():
    cfg = CostConfig(bars_per_year=np.int64(252))
    assert cfg.bars_per_year == 252
    assert type(cfg.bars_per_year) is int


def test_plain_int_bars_per_year_is_unchanged():
    cfg = CostConfig(bars_per_year=52)
    assert cfg.bars_per_year == 52 and type(cfg.bars_per_year) is int


def test_numpy_int32_commission_sums_into_per_side_bps():
    cfg = CostConfig(commission_bps=np.int32(1))
    assert cfg.per_side_bps == 5.0            # 1 + 2.0 + 2.0 defaults


def test_numpy_float64_cost_fields_are_accepted():
    cfg = CostConfig(half_spread_bps=np.float64(2.0), borrow_bps_pa=np.float32(25.0))
    assert cfg.per_side_bps == 5.0


def test_numpy_int64_capital_constructs_a_ledger_config():
    cfg = LedgerConfig(capital=np.int64(100_000))
    assert cfg.capital == 100_000


@pytest.mark.parametrize("kwargs", [
    {"bars_per_year": np.bool_(True)},
    {"bars_per_year": True},
    {"bars_per_year": np.int64(0)},
    {"bars_per_year": np.float64(252.0)},        # a float is still not an integer
    {"commission_bps": np.float64("inf")},
    {"commission_bps": np.float64("nan")},
    {"commission_bps": np.int64(-1)},
    {"commission_bps": np.bool_(True)},
])
def test_numpy_scalars_that_are_not_real_or_not_finite_still_raise(kwargs):
    with pytest.raises(ValueError):
        CostConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"capital": np.int64(0)}, {"max_gross": np.float64("-inf")},
    {"gross_target": np.float64("nan")}, {"capital": np.bool_(True)},
])
def test_numpy_scalars_still_hit_the_ledger_config_rejections(kwargs):
    with pytest.raises(ValueError):
        LedgerConfig(**kwargs)


def _frame(cfg: LedgerConfig) -> pd.DataFrame:
    n = 6
    idx = pd.bdate_range("2020-01-01", periods=n)
    o1 = np.array([100.0, 100.0, 110.0, 120.0, 118.0, 121.0])
    c1 = np.array([100.0, 105.0, 115.0, 118.0, 120.0, 119.0])
    o2 = np.array([50.0, 50.0, 52.0, 54.0, 53.0, 55.0])
    c2 = np.array([50.0, 51.0, 53.0, 53.0, 54.0, 54.5])
    dec = [Decision(t, pos, "test") for t, pos in enumerate(
        [Position.LONG_RESIDUAL, Position.LONG_RESIDUAL, Position.FLAT,
         Position.SHORT_RESIDUAL, Position.SHORT_RESIDUAL, Position.FLAT])]
    return run_ledger(idx, o1, o2, c1, c2, dec, np.full(n, 1.5), cfg)


def test_numpy_built_config_accounts_identically_to_the_default_config():
    numpy_cfg = LedgerConfig(
        capital=np.int64(100_000), gross_target=np.int64(100_000),
        max_gross=np.int64(200_000), rehedge_threshold=np.float64(0.10),
        costs=CostConfig(commission_bps=np.int64(1), half_spread_bps=np.float64(2.0),
                         slippage_bps=np.int32(2), borrow_bps_pa=np.int64(25),
                         financing_bps_pa=np.int64(0), bars_per_year=np.int64(252)))
    pd.testing.assert_frame_equal(_frame(numpy_cfg), _frame(LedgerConfig()))
    assert type(numpy_cfg.costs.bars_per_year) is int
