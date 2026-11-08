"""`_validate_frame` rejects bool/complex prices by dtype first and scans cells
only in object-dtype columns, so a plain float64 frame is validated without a
Python loop over every cell. The semantics are pinned here: a bool or complex
dtype column is rejected, Decimal objects are still accepted, and an
object-wrapped complex or bool cell is still rejected.
"""

from __future__ import annotations

import time
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from kalman.data import _validate_frame


def _frame(n: int = 3, **cols) -> pd.DataFrame:
    base = {"Open": np.linspace(10.0, 11.0, n), "Close": np.linspace(10.5, 11.5, n)}
    base.update(cols)
    return pd.DataFrame(base, index=pd.bdate_range("2020-01-01", periods=n))


def test_bool_dtype_close_column_is_rejected():
    df = _frame(Close=np.array([True, True, False]))
    assert df["Close"].dtype.kind == "b"
    with pytest.raises(ValueError, match="real numeric"):
        _validate_frame(df, "bools")


def test_complex128_column_is_rejected():
    df = _frame(Open=np.array([1 + 0j, 2 + 0j, 3 + 0j]))
    assert df["Open"].dtype == np.complex128
    with pytest.raises(ValueError, match="real numeric"):
        _validate_frame(df, "complex")


def test_object_column_of_decimals_is_accepted_and_cast_to_float():
    df = _frame(Close=np.array([Decimal("10.5"), Decimal("11.0"), Decimal("11.5")], dtype=object))
    assert df["Close"].dtype.kind == "O"
    out = _validate_frame(df, "decimals")
    assert out["Close"].dtype == np.float64
    np.testing.assert_allclose(out["Close"].to_numpy(), [10.5, 11.0, 11.5])


@pytest.mark.parametrize("bad", [np.complex128(1 + 2j), 1 + 2j, True, np.bool_(False)])
@pytest.mark.parametrize("col", [0, 1])
def test_object_wrapped_complex_or_bool_cell_is_still_rejected(bad, col):
    df = _frame().astype(object)
    df.iloc[-1, col] = bad                          # last row: the scan reaches the end
    with pytest.raises(ValueError, match="real numeric"):
        _validate_frame(df, "wrapped")


def test_integer_and_float_columns_skip_the_scan_and_still_validate():
    df = _frame(Open=np.array([10, 11, 12]))
    assert df["Open"].dtype.kind == "i"
    out = _validate_frame(df, "ints")
    assert out.dtypes.tolist() == [np.float64, np.float64]
    df.iloc[1, 1] = np.nan                           # interior NaN survives, explicitly
    assert np.isnan(_validate_frame(df, "nan")["Close"].iloc[1])


def test_float64_frame_of_100k_rows_validates_well_under_a_second():
    n = 100_000
    df = pd.DataFrame({"Open": np.linspace(10.0, 20.0, n), "Close": np.linspace(11.0, 21.0, n)},
                      index=pd.date_range("2000-01-01", periods=n, freq="h"))
    t0 = time.perf_counter()
    out = _validate_frame(df, "big")
    elapsed = time.perf_counter() - t0
    assert len(out) == n and out.dtypes.tolist() == [np.float64, np.float64]
    assert elapsed < 1.0, f"validation took {elapsed:.3f}s"
