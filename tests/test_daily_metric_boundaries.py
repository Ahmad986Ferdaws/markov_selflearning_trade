import numpy as np
import pytest

from app.services.evaluation import _accuracy, _policy_metrics


@pytest.mark.parametrize("market,cost,expected", [([-.2], 0, -.2), ([0.], .01, -.01),
                                                ([-.2, .1], 0, -.2), ([.2, -.25], 0, -.25)])
def test_drawdown_includes_initial_capital(market, cost, expected):
    result = _policy_metrics("long", [1.] * len(market), market, cost)
    assert result.max_drawdown == pytest.approx(expected)


@pytest.mark.parametrize("actual,expected", [([1, 1], 0.), ([1, 2], 0.), ([0, 1], .5)])
def test_majority_recall_only_credits_a_present_class(actual, expected):
    result = _accuracy([np.array([1., 0., 0.])] * len(actual), actual,
                       np.array([1., 0., 0.]))
    assert result.naive_balanced == expected
