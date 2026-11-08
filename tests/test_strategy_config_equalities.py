"""Pins for StrategyConfig's inclusive and exclusive edges.

tests/test_strategy_signal_boundaries.py::test_invalid_strategy_configuration
rejects a scattered list of bad values; this file pins each threshold from
both sides, with its message, so a later relaxation or tightening of
`0 <= exit_z < entry_z <= stop_z` or of the counter checks shows up here.
"""

from __future__ import annotations

import numpy as np
import pytest

from kalman.strategy import StrategyConfig

_ORDER = "need 0 <= exit_z < entry_z <= stop_z"


def test_exit_z_equal_to_entry_z_is_rejected_and_one_ulp_below_is_accepted():
    with pytest.raises(ValueError, match=_ORDER):
        StrategyConfig(entry_z=2.0, exit_z=2.0)
    cfg = StrategyConfig(entry_z=2.0, exit_z=np.nextafter(2.0, 0.0))
    assert cfg.exit_z < cfg.entry_z


def test_entry_z_equal_to_stop_z_is_accepted_and_above_it_is_rejected():
    cfg = StrategyConfig(entry_z=2.0, exit_z=0.5, stop_z=2.0)
    assert cfg.entry_z == cfg.stop_z == 2.0
    with pytest.raises(ValueError, match=_ORDER):
        StrategyConfig(entry_z=2.0, exit_z=0.5, stop_z=np.nextafter(2.0, 0.0))


def test_exit_z_zero_is_accepted_and_any_negative_is_rejected():
    assert StrategyConfig(exit_z=0.0).exit_z == 0.0
    for bad in (-0.1, np.nextafter(0.0, -1.0)):
        with pytest.raises(ValueError, match=_ORDER):
            StrategyConfig(exit_z=bad)


@pytest.mark.parametrize("name", ["warmup", "max_holding", "cooldown"])
def test_counters_accept_zero_and_reject_negative_bool_and_float(name):
    assert getattr(StrategyConfig(**{name: 0}), name) == 0
    for bad in (-1, True, 1.0):
        with pytest.raises(ValueError, match=f"{name} must be a nonnegative integer"):
            StrategyConfig(**{name: bad})


def test_equal_beta_bounds_are_accepted_and_one_ulp_crossed_is_rejected():
    cfg = StrategyConfig(beta_min=1.0, beta_max=1.0)
    assert cfg.beta_min == cfg.beta_max == 1.0
    with pytest.raises(ValueError, match="beta_min must not exceed beta_max"):
        StrategyConfig(beta_min=np.nextafter(1.0, 2.0), beta_max=1.0)


def test_numpy_float64_entry_z_is_accepted_as_a_real_number():
    cfg = StrategyConfig(entry_z=np.float64(2.0))
    assert cfg.entry_z == 2.0
    assert isinstance(cfg.entry_z, float)            # np.float64 subclasses float


@pytest.mark.parametrize("bad", [float("inf"), -float("inf")])
def test_infinite_stop_z_is_rejected(bad):
    with pytest.raises(ValueError, match="stop_z must be a finite real number"):
        StrategyConfig(stop_z=bad)
