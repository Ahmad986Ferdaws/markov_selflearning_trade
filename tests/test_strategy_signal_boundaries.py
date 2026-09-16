import pytest

from kalman.strategy import Position, StateMachine, StrategyConfig


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('invested', [False, True])
def test_nonfinite_signal_forces_flat(bad, invested):
    sm = StateMachine(StrategyConfig(warmup=0, cooldown=2))
    if invested:
        sm.decide(0, -3., 1.)
    d = sm.decide(1, bad, 1.)
    assert d.target is Position.FLAT and d.gated
    assert sm.pos is Position.FLAT
    if invested:
        assert sm.cooldown_left == 2


@pytest.mark.parametrize('kwargs', [{'warmup': -1}, {'cooldown': -1},
    {'max_holding': .5}, {'warmup': True}, {'beta_min': float('nan')},
    {'beta_min': 5, 'beta_max': -5}, {'stop_z': float('inf')}])
def test_invalid_strategy_configuration(kwargs):
    with pytest.raises(ValueError):
        StrategyConfig(**kwargs)


def test_finite_entry_exit_and_cooldown_preserved():
    sm = StateMachine(StrategyConfig(warmup=0, cooldown=1))
    assert sm.decide(0, -3., 1.).target is Position.LONG_RESIDUAL
    assert sm.decide(1, .1, 1.).target is Position.FLAT
    assert sm.decide(2, -3., 1.).reason == 'cooldown'
    assert sm.decide(3, -3., 1.).target is Position.LONG_RESIDUAL
