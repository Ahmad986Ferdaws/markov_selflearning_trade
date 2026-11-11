"""Pins for sharpe_ci_block_bootstrap's argument validation and its output.

A zero block divided by zero, a negative block crashed inside numpy with
'negative dimensions are not allowed', and n_boot=0 produced an empty
percentile array and an IndexError. Each is now a ValueError naming the
argument, raised before the generator draws anything, so the CI90 column
of every report is unchanged: the two floats below were computed at the
commit before this check was added and are asserted exactly.

tests/test_perf_metrics_degenerate.py pins the minimum-length and seeding
contracts; the argument floor and the exact value are pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from kalman.research import sharpe_ci_block_bootstrap

RETS = np.random.default_rng(0).normal(0.0, 0.01, 200)


@pytest.mark.parametrize("block", [0, -1])
def test_a_block_below_one_is_rejected_by_name(block):
    with pytest.raises(ValueError, match="block must be an int >= 1"):
        sharpe_ci_block_bootstrap(RETS, block=block)


def test_a_bool_block_is_rejected_even_though_bool_is_an_int():
    with pytest.raises(ValueError, match="block must be an int >= 1"):
        sharpe_ci_block_bootstrap(RETS, block=True)


def test_a_float_block_is_rejected():
    with pytest.raises(ValueError, match="block must be an int >= 1"):
        sharpe_ci_block_bootstrap(RETS, block=20.0)


@pytest.mark.parametrize("n_boot", [0, -5, True])
def test_n_boot_below_one_or_bool_is_rejected_by_name(n_boot):
    with pytest.raises(ValueError, match="n_boot must be an int >= 1"):
        sharpe_ci_block_bootstrap(RETS, n_boot=n_boot)


def test_validation_runs_before_the_minimum_length_check():
    # 39 bars is below 2 * 20, which used to return (nan, nan) for any block;
    # an invalid block is now refused first
    with pytest.raises(ValueError, match="block must be an int >= 1"):
        sharpe_ci_block_bootstrap(RETS[:39], block=0)


def test_seeded_interval_is_bit_identical_to_the_pre_validation_value():
    lo, hi = sharpe_ci_block_bootstrap(RETS, n_boot=200, block=20, seed=1)
    assert lo == pytest.approx(-1.6029919237879828, abs=0)
    assert hi == pytest.approx(2.008563669204777, abs=0)
