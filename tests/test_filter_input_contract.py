import numpy as np
import pytest

from kalman.core import AdaptiveQ, PairFilter, TrendFilter


@pytest.mark.parametrize('factory', [PairFilter, TrendFilter])
@pytest.mark.parametrize('bad', [np.nan, np.inf, True, 1j])
def test_nonfinite_or_nonreal_noise_rejected(factory, bad):
    with pytest.raises(ValueError):
        factory(bad, .01, .1)


@pytest.mark.parametrize('factory', [PairFilter, TrendFilter])
@pytest.mark.parametrize('kwargs', [{'x0': [1.]}, {'x0': [np.nan, 0]},
    {'x0': [1j, 0]}, {'P0': np.ones(2)}, {'P0': [[1, 2], [0, 1]]},
    {'P0': [[1, 0], [0, -1]]}, {'P0': [[np.inf, 0], [0, 1]]}])
def test_invalid_prior_rejected_at_construction(factory, kwargs):
    with pytest.raises(ValueError):
        factory(.01, .01, .1, **kwargs)


@pytest.mark.parametrize('kwargs', [{'sigma_ref': np.nan}, {'m_min': -1},
    {'m_min': 3, 'm_max': 2}, {'gamma': np.inf}, {'window': 1}, {'window': True}])
def test_invalid_adaptive_controls(kwargs):
    with pytest.raises(ValueError):
        AdaptiveQ(**{'sigma_ref': .1, **kwargs})


@pytest.mark.parametrize('dt', [np.nan, np.inf, 0., -1., True])
def test_rejected_dt_does_not_advance_filter(dt):
    f = TrendFilter(.01, .01, .1)
    with pytest.raises(ValueError):
        f.step(1., dt=dt)
    assert f.step(1.).t == 0


def test_valid_prior_is_owned_and_zero_psd_prior_allowed():
    x = np.array([1., 0.]); p = np.zeros((2, 2))
    f = PairFilter(.01, .01, .1, x0=x, P0=p)
    x[:] = np.nan; p[:] = np.nan
    assert np.isfinite(f.step(2., 2.).x_post).all()


def test_numpy_integer_adaptive_window_matches_python_window():
    native = AdaptiveQ(.1, window=3)
    grid = AdaptiveQ(.1, window=np.int64(3))
    assert grid.multiplier(np.array([.1, -.2, .1])) == native.multiplier(np.array([.1, -.2, .1]))
    assert type(grid.window) is int
