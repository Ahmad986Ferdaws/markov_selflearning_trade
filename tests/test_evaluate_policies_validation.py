"""evaluate() (app/services/evaluation.py) validates its policies mapping like every other control.

_validate_controls refuses a bool, a non-finite value or an empty grid, but the
policies mapping went unchecked: ``{}`` produced a DailyReport with no policy
rows (a trading table with a header and nothing under it), a non-callable
value died deep inside the walk-forward with a bare TypeError, and a non-str
key was accepted. Each is now a ValueError raised before any work is done; the
default path (policies=None -> baseline + buy_hold) is unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.evaluation import evaluate

GRID = dict(grid_windows=(10,), grid_k=(0.5,))      # one config: fast, deterministic


def _trending_history(periods: int = 400, seed: int = 0) -> pd.DataFrame:
    """Alternating up/down regimes (drift >~ vol), as in tests/test_evaluation.py."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=periods, freq="D")
    drift = np.where((np.arange(periods) // 50) % 2 == 0, 0.015, -0.015)
    rets = drift + rng.normal(0, 0.012, periods)
    return pd.DataFrame({"Close": 100 * np.cumprod(1 + rets)}, index=idx)


@pytest.mark.parametrize("policies", [
    {},                                 # empty: a report with no policy rows
    {"x": 1},                           # non-callable value: used to be a TypeError mid-walk
    {1: lambda ctx: 1.0},               # non-str key
], ids=["empty", "non-callable", "non-str-key"])
def test_a_bad_policies_mapping_is_a_value_error_before_any_work(policies):
    with pytest.raises(ValueError, match="policies must be a non-empty mapping"):
        evaluate(_trending_history(), policies=policies, **GRID)


def test_a_bad_policies_mapping_is_never_a_type_error():
    try:
        evaluate(_trending_history(), policies={"x": 1}, **GRID)
    except TypeError as error:                      # pragma: no cover - the defect
        pytest.fail(f"non-callable policy escaped as TypeError: {error}")
    except ValueError:
        pass


def test_the_default_policies_still_yield_baseline_and_buy_hold_rows():
    report = evaluate(_trending_history(), policies=None, **GRID)
    assert {p.name for p in report.policies} == {"baseline", "buy_hold"}
