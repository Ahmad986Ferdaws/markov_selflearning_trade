import numpy as np
import pandas as pd
import pytest

from app.services.evaluation import evaluate


def history():
    return pd.DataFrame({'Close': np.linspace(100, 120, 150)},
                        index=pd.date_range('2020-01-01', periods=150))


UNIQUE_SORTED = 'history timestamps must be unique and sorted ascending'
DATETIME_INDEX = 'history must have non-missing datetime timestamps'
FINITE_POSITIVE = 'Close prices must be finite and positive; missing bars are not imputed'


@pytest.mark.parametrize('kind, message', [
    ('reversed', UNIQUE_SORTED),
    ('duplicate', UNIQUE_SORTED),
    ('nat', DATETIME_INDEX),
    ('range-index', DATETIME_INDEX),
    ('duplicate-column', 'history must be a DataFrame with unique columns'),
    ('no-close', 'history must contain a Close column'),
    ('negative', FINITE_POSITIVE),
    ('zero', FINITE_POSITIVE),
    ('nan', FINITE_POSITIVE),
    ('inf', FINITE_POSITIVE),
    ('complex', 'Close prices must be real numeric values'),
    ('overflow', 'Close prices produced non-finite returns'),
])
def test_invalid_history_rejected_before_policy(kind, message):
    frame = history()
    if kind == 'reversed':
        frame = frame.iloc[::-1]
    elif kind in ('duplicate', 'nat'):
        dates = list(frame.index)
        dates[1] = dates[0] if kind == 'duplicate' else pd.NaT
        frame.index = dates
    elif kind == 'range-index':
        frame = frame.reset_index(drop=True)
    elif kind == 'duplicate-column':
        frame = pd.concat([frame, frame], axis=1)
    elif kind == 'no-close':
        frame = frame.rename(columns={'Close': 'close'})
    elif kind == 'overflow':
        # both finite and positive, so they pass the price check; the return
        # between them (1e300 / 1e-300) overflows to inf
        frame.iloc[20, 0] = 1e-300
        frame.iloc[21, 0] = 1e300
    else:
        if kind == 'complex':
            frame = frame.astype(complex)
        frame.iloc[20, 0] = {'negative': -1, 'zero': 0, 'nan': np.nan,
                             'inf': np.inf, 'complex': 1+2j}[kind]
    def forbidden(ctx):
        pytest.fail('policy must not run on invalid data')
    with pytest.raises(ValueError, match=message):
        evaluate(frame, grid_windows=(10,), grid_k=(.5,), policies={'sentinel': forbidden})


@pytest.mark.parametrize('kwargs', [{'fee_pct': -.1}, {'slippage_pct': np.inf},
    {'grid_k': (np.nan,)}, {'grid_windows': (True,)}, {'train_frac': np.nan},
    {'min_train': -1}, {'grid_k': ()}])
def test_invalid_evaluation_controls(kwargs):
    with pytest.raises(ValueError):
        evaluate(history(), **kwargs)


def test_valid_prices_preserve_daily_split():
    frame = history()
    options = {'grid_windows': (10,), 'grid_k': (.5,)}
    result = evaluate(frame, **options)
    assert result.test_size == 44
    assert result == evaluate(frame.copy(), **options)


def test_fixed_width_costs_cannot_overflow_into_rebates():
    with pytest.raises(ValueError, match='cost'):
        evaluate(history(), fee_pct=np.int8(100), slippage_pct=np.int8(100))


def test_numpy_cost_controls_match_python_values():
    options = {'grid_windows': (10,), 'grid_k': (.5,)}
    assert evaluate(history(), fee_pct=np.float32(.5), slippage_pct=np.int8(1), **options) == \
        evaluate(history(), fee_pct=.5, slippage_pct=1., **options)
