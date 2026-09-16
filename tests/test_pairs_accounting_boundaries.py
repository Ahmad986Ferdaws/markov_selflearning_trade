import numpy as np
import pandas as pd
import pytest

from kalman.ledger import CostConfig, LedgerConfig, _target_shares, run_ledger
from kalman.strategy import Decision, Position


@pytest.mark.parametrize('beta', [0., -.5, .5, 2.])
def test_log_hedge_preserves_signed_elasticity_including_zero(beta):
    cfg = LedgerConfig(gross_target=10000.)
    s1, s2 = _target_shares(Position.LONG_RESIDUAL, beta, 100., 50., cfg)
    assert s2 * 50 == pytest.approx(-beta * s1 * 100)
    assert abs(s1 * 100) + abs(s2 * 50) == pytest.approx(10000.)


@pytest.mark.parametrize('kwargs', [{'commission_bps': -1}, {'borrow_bps_pa': np.nan},
    {'slippage_bps': np.inf}, {'bars_per_year': 0}, {'bars_per_year': True}])
def test_invalid_costs_rejected(kwargs):
    with pytest.raises(ValueError):
        CostConfig(**kwargs)


@pytest.mark.parametrize('kwargs', [{'capital': 0}, {'gross_target': np.nan},
    {'max_gross': -1}, {'rehedge_threshold': -1}, {'price_model': 'log'}])
def test_invalid_ledger_configuration(kwargs):
    with pytest.raises(ValueError):
        LedgerConfig(**kwargs)


@pytest.mark.parametrize('kind', ['nan-close', 'zero-close', 'bad-beta', 'empty'])
def test_undefined_accounting_rejected(kind):
    idx = pd.date_range('2020-01-01', periods=3)
    prices = np.full(3, 100.)
    c = prices.copy(); b = np.ones(3)
    if kind == 'nan-close': c[1] = np.nan
    if kind == 'zero-close': c[1] = 0
    if kind == 'bad-beta': b[0] = np.nan
    if kind == 'empty': idx = idx[:0]; prices = prices[:0]; c = c[:0]; b = b[:0]
    decisions = [Decision(t, Position.LONG_RESIDUAL, 'test') for t in range(len(idx))]
    with pytest.raises(ValueError):
        run_ledger(idx, prices, prices, c, prices, decisions, b, LedgerConfig())
