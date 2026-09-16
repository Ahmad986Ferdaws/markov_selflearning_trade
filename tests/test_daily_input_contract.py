import numpy as np
import pandas as pd
import pytest

from app.services.evaluation import evaluate


def history():
    return pd.DataFrame({'Close': np.linspace(100, 120, 150)},
                        index=pd.date_range('2020-01-01', periods=150))


@pytest.mark.parametrize('kind', ['reversed', 'duplicate', 'nat', 'negative', 'zero',
                                 'nan', 'inf', 'complex', 'duplicate-column'])
def test_invalid_history_rejected_before_policy(kind):
    frame = history()
    if kind == 'reversed':
        frame = frame.iloc[::-1]
    elif kind in ('duplicate', 'nat'):
        dates = list(frame.index)
        dates[1] = dates[0] if kind == 'duplicate' else pd.NaT
        frame.index = dates
    elif kind == 'duplicate-column':
        frame = pd.concat([frame, frame], axis=1)
    else:
        if kind == 'complex':
            frame = frame.astype(complex)
        frame.iloc[20, 0] = {'negative': -1, 'zero': 0, 'nan': np.nan,
                             'inf': np.inf, 'complex': 1+2j}[kind]
    def forbidden(ctx):
        pytest.fail('policy must not run on invalid data')
    with pytest.raises(ValueError):
        evaluate(frame, grid_windows=(10,), grid_k=(.5,), policies={'sentinel': forbidden})


@pytest.mark.parametrize('kwargs', [{'fee_pct': -.1}, {'slippage_pct': np.inf},
    {'grid_k': (np.nan,)}, {'grid_windows': (True,)}, {'train_frac': np.nan},
    {'min_train': -1}, {'grid_k': ()}])
def test_invalid_evaluation_controls(kwargs):
    with pytest.raises(ValueError):
        evaluate(history(), **kwargs)


def test_valid_prices_preserve_daily_split():
    frame = history()
    options = dict(grid_windows=(10,), grid_k=(.5,))
    result = evaluate(frame, **options)
    assert result.test_size == 44
    assert result == evaluate(frame.copy(), **options)
