import numpy as np
import pandas as pd
import pytest

from app.services.evaluation import _walk_forward


def walk(policies):
    return _walk_forward(pd.Series([0., .1, .2, -.1]), ['bull'] * 4,
                         1, 4, 0, 0, policies)


def test_policy_mutation_cannot_change_accuracy_or_peer_inputs():
    def mutate(ctx):
        ctx.p_next[:] = [0, 1, 0]
        return 0.
    def observer(ctx):
        return float(ctx.p_next[0])
    control = walk({'observer': observer})
    changed = walk({'mutator': mutate, 'observer': observer})
    np.testing.assert_array_equal(changed['pred_probs'], control['pred_probs'])
    assert changed['positions']['observer'] == control['positions']['observer']


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -.1, 1.1, True, 1j])
def test_invalid_policy_position_fails_with_policy_name(value):
    with pytest.raises(ValueError, match='broken'):
        walk({'broken': lambda ctx: value})


@pytest.mark.parametrize('value', [0., .5, 1.])
def test_valid_exposures_preserved(value):
    assert walk({'valid': lambda ctx: value})['positions']['valid'] == [value, value]
