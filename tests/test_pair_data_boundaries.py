import numpy as np
import pandas as pd
import pytest

from kalman.data import align_pair


def frame():
    return pd.DataFrame({'Open': [1., 2., 3.], 'Close': [2., 3., 4.]},
                        index=pd.date_range('2020-01-01', periods=3))


@pytest.mark.parametrize('bad', [-1., 0., np.inf, -np.inf, 1+2j])
def test_invalid_cell_is_not_hidden_by_missing_neighbor(bad):
    data = frame().astype(complex if np.iscomplexobj(bad) else float)
    data.iloc[1] = [np.nan, bad]
    with pytest.raises(ValueError):
        align_pair(data, frame(), 'a', 'b', min_overlap=2)


def test_duplicate_columns_rejected():
    data = pd.concat([frame(), frame()['Close']], axis=1)
    with pytest.raises(ValueError, match='columns'):
        align_pair(data, frame(), 'a', 'b', min_overlap=2)


def test_missing_observations_survive_without_imputation():
    data = frame()
    data.iloc[1, 0] = np.nan
    result = align_pair(data, frame(), 'a', 'b', min_overlap=2)
    pd.testing.assert_frame_equal(result.p1, data)
    assert len(result) == 3


def test_decimal_object_prices_remain_supported():
    from decimal import Decimal
    data = frame().map(lambda v: Decimal(str(v)))
    result = align_pair(data, frame(), 'a', 'b', min_overlap=2)
    pd.testing.assert_frame_equal(result.p1, frame())


def test_object_wrapped_complex_is_not_silently_cast():
    data = frame().astype(object)
    data.iloc[1, 1] = np.complex128(1+2j)
    with pytest.raises(ValueError):
        align_pair(data, frame(), 'a', 'b', min_overlap=2)
