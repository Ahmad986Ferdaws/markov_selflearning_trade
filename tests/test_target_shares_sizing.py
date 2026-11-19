"""Pins for kalman.ledger._target_shares: the two-leg sizing identities of the
LOG (elasticity) and LEVEL (share-ratio) price models at one worked example.

tests/test_kalman_ledger.py::test_log_model_sizing_uses_beta_magnitude pins
that |beta| matters in the LOG model; this file pins the exact share counts,
the max_gross cap, the SHORT_RESIDUAL sign flip, FLAT, the LEVEL model's share
ratio, and the LEVEL model's zero-price guard.
"""

from __future__ import annotations

import pytest

from kalman.ledger import LedgerConfig, PriceModel, _target_shares
from kalman.strategy import Position

BETA, O1, O2 = 1.5, 50.0, 20.0


def _log(gross_target: float = 100_000.0, **kw) -> LedgerConfig:
    return LedgerConfig(gross_target=gross_target, price_model=PriceModel.LOG, **kw)


def _level(gross_target: float = 100_000.0, **kw) -> LedgerConfig:
    return LedgerConfig(gross_target=gross_target, price_model=PriceModel.LEVEL, **kw)


def test_log_model_splits_gross_by_elasticity_and_hedges_by_beta():
    s1, s2 = _target_shares(Position.LONG_RESIDUAL, BETA, O1, O2, _log())
    # L1 = 100k / (1 + 1.5) = 40k -> 800 sh @ 50;  L2 = -1.5 * 40k = -60k -> -3000 sh @ 20
    assert (s1, s2) == (800.0, -3000.0)
    assert abs(s1 * O1) + abs(s2 * O2) == pytest.approx(100_000.0)
    assert s2 * O2 == pytest.approx(-BETA * s1 * O1)


@pytest.mark.parametrize("make", [_log, _level])
def test_gross_target_above_max_gross_is_capped_at_max_gross(make):
    cfg = make(gross_target=500_000.0, max_gross=200_000.0)
    s1, s2 = _target_shares(Position.LONG_RESIDUAL, BETA, O1, O2, cfg)
    assert abs(s1 * O1) + abs(s2 * O2) == pytest.approx(200_000.0)


@pytest.mark.parametrize("make", [_log, _level])
def test_short_residual_flips_both_legs_exactly(make):
    long_ = _target_shares(Position.LONG_RESIDUAL, BETA, O1, O2, make())
    short = _target_shares(Position.SHORT_RESIDUAL, BETA, O1, O2, make())
    assert short == (-long_[0], -long_[1])
    assert short[0] < 0.0 < short[1]                 # short the y-leg, long the hedge


@pytest.mark.parametrize("make", [_log, _level])
def test_flat_is_zero_shares_on_both_legs(make):
    assert _target_shares(Position.FLAT, BETA, O1, O2, make()) == (0.0, 0.0)


def test_level_model_uses_beta_as_a_share_ratio_at_the_gross_target():
    s1, s2 = _target_shares(Position.LONG_RESIDUAL, BETA, O1, O2, _level())
    # unit basket (1, -1.5) costs 50 + 30 = 80 -> k = 100k / 80 = 1250 -> (1250, -1875)
    assert (s1, s2) == (1250.0, -1875.0)
    assert s2 / s1 == pytest.approx(-BETA)
    assert abs(s1 * O1) + abs(s2 * O2) == pytest.approx(min(100_000.0, _level().max_gross))


def test_level_model_with_zero_fill_prices_returns_zeros_instead_of_dividing():
    assert _target_shares(Position.LONG_RESIDUAL, BETA, 0.0, 0.0, _level()) == (0.0, 0.0)
    assert _target_shares(Position.SHORT_RESIDUAL, BETA, 0.0, 0.0, _level()) == (0.0, 0.0)
