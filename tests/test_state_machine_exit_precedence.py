"""Pins for kalman.strategy.StateMachine: which exit wins when several apply,
the max-holding counter's boundary, the stale-bar flatten while invested, the
bars_held reset on both exit paths, and the inclusive entry thresholds.

tests/test_kalman_research.py::test_state_machine_hysteresis_and_gates covers
the ordinary hysteresis path with values well inside each band; this file pins
the equalities and the precedence order stop > max-holding > exit-band.
"""

from __future__ import annotations

import numpy as np
import pytest

from kalman.strategy import Position, StateMachine, StrategyConfig


def _sm(**overrides) -> StateMachine:
    base = dict(entry_z=2.0, exit_z=0.5, stop_z=4.0, warmup=0, cooldown=0, max_holding=0)
    return StateMachine(StrategyConfig(**{**base, **overrides}))


def _enter_long(sm: StateMachine, t: int = 0) -> None:
    d = sm.decide(t, -2.5, 1.0)
    assert d.target is Position.LONG_RESIDUAL and d.reason == "enter-long-residual"
    assert sm.bars_held == 0


def test_max_holding_fires_on_the_decision_where_bars_held_reaches_the_limit():
    sm = _sm(max_holding=3)
    _enter_long(sm)
    assert sm.decide(1, -1.0, 1.0).reason == "hold" and sm.bars_held == 1
    assert sm.decide(2, -1.0, 1.0).reason == "hold" and sm.bars_held == 2
    d = sm.decide(3, -1.0, 1.0)                      # bars_held reaches 3 here
    assert d.target is Position.FLAT and d.reason == "max-holding" and not d.gated
    assert sm.pos is Position.FLAT and sm.bars_held == 0


def test_max_holding_zero_never_fires():
    sm = _sm(max_holding=0)
    _enter_long(sm)
    for t in range(1, 201):
        d = sm.decide(t, -1.0, 1.0)                  # in band: exit_z < |z| < stop_z
        assert d.target is Position.LONG_RESIDUAL and d.reason == "hold"
    assert sm.bars_held == 200


@pytest.mark.parametrize("z", [-4.0, 4.0])
def test_z_exactly_at_stop_z_is_an_emergency_stop(z):
    sm = _sm()
    _enter_long(sm)
    assert abs(z) > sm.cfg.exit_z                    # not an exit-band bar
    d = sm.decide(1, z, 1.0)
    assert d.target is Position.FLAT and d.reason == "emergency-stop"


@pytest.mark.parametrize("z", [-0.5, 0.5])
def test_z_exactly_at_exit_z_leaves_through_the_exit_band(z):
    sm = _sm()
    _enter_long(sm)
    d = sm.decide(1, z, 1.0)
    assert d.target is Position.FLAT and d.reason == "exit-band"
    # one ulp outside the band is still a hold
    sm2 = _sm()
    _enter_long(sm2)
    assert sm2.decide(1, np.nextafter(z, 10.0 * z), 1.0).reason == "hold"


def test_exit_precedence_is_stop_then_max_holding_then_exit_band():
    # stop and max-holding both apply on the same bar: the stop wins
    sm = _sm(max_holding=1)
    _enter_long(sm)
    assert sm.decide(1, -4.0, 1.0).reason == "emergency-stop"
    # max-holding and exit-band both apply: max-holding wins
    sm = _sm(max_holding=1)
    _enter_long(sm)
    assert sm.decide(1, -0.1, 1.0).reason == "max-holding"


def test_stale_bar_while_invested_flattens_gated_and_arms_the_cooldown():
    sm = _sm(cooldown=3)
    _enter_long(sm)
    assert sm.decide(1, -1.0, 1.0).reason == "hold"
    d = sm.decide(2, None, 1.0)
    assert d.target is Position.FLAT and d.reason == "gate:stale-data" and d.gated
    assert sm.pos is Position.FLAT
    assert sm.cooldown_left == sm.cfg.cooldown == 3
    assert sm.bars_held == 0


def test_exit_and_flatten_both_reset_bars_held():
    via_exit = _sm()
    _enter_long(via_exit)
    via_exit.decide(1, -1.0, 1.0)
    via_exit.decide(2, -1.0, 1.0)
    assert via_exit.bars_held == 2
    assert via_exit.decide(3, 0.0, 1.0).reason == "exit-band"              # _exit
    assert via_exit.bars_held == 0

    via_gate = _sm()
    _enter_long(via_gate)
    via_gate.decide(1, -1.0, 1.0)
    via_gate.decide(2, -1.0, 1.0)
    assert via_gate.bars_held == 2
    assert via_gate.decide(3, -1.0, 1.0, healthy=False).reason == "gate:health"  # _flatten
    assert via_gate.bars_held == 0


def test_entry_thresholds_are_inclusive():
    assert _sm().decide(0, -2.0, 1.0).target is Position.LONG_RESIDUAL
    assert _sm().decide(0, 2.0, 1.0).target is Position.SHORT_RESIDUAL
    # one ulp inside the band: no signal
    assert _sm().decide(0, np.nextafter(-2.0, 0.0), 1.0).reason == "no-signal"
    assert _sm().decide(0, np.nextafter(2.0, 0.0), 1.0).reason == "no-signal"
