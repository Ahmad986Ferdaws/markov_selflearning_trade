"""regime_feature's point-in-time slice (app/services/regime.py) compares the
calendar DATE of ``as_of`` with the calendar dates of the history index, so a
timezone-aware as_of, a naive as_of, a datetime.datetime and a timezone-aware
history index must all cut the same bars, and the as_of day's own bar must be
excluded whatever the awareness. tests/test_metrics_correctness.py pins the
strictly-before rule with naive inputs only; nothing else passed a tz-aware
index or as_of.

Offline and deterministic: a seeded walk with alternating 50-day drift blocks
(the recipe tests/test_evaluation.py uses) so bull, bear and sideways days all
occur and the transition counts, hence p_next, move with every bar. A pure
driftless walk on this seed labels every day sideways and its feature is the
constant (0, 0, 1), which no slicing rule could ever change.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from app.services.regime import RegimeFeature, regime_feature

AS_OF = "2020-07-01 00:30"


def _history() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    drift = np.where((np.arange(200) // 50) % 2 == 0, 0.015, -0.015)
    closes = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.02, 200)))
    return pd.DataFrame({"Close": closes}, index=pd.bdate_range("2020-01-01", periods=200))


def _new_york(history: pd.DataFrame) -> pd.DataFrame:
    aware = history.copy()
    aware.index = aware.index.tz_localize("America/New_York")
    return aware


def test_aware_naive_and_datetime_as_of_cut_the_same_bars_on_naive_and_aware_indexes():
    h = _history()

    utc_as_of = regime_feature(h, as_of=pd.Timestamp(AS_OF, tz="UTC"))
    naive_as_of = regime_feature(h, as_of=pd.Timestamp(AS_OF))
    datetime_as_of = regime_feature(h, as_of=datetime(2020, 7, 1, 0, 30))
    aware_index = regime_feature(_new_york(h), as_of=pd.Timestamp(AS_OF, tz="UTC"))

    assert isinstance(utc_as_of, RegimeFeature)
    assert utc_as_of == naive_as_of == datetime_as_of == aware_index


def test_the_as_of_day_is_excluded_whatever_the_awareness():
    h = _history()
    # 2020-07-01 is a Wednesday bar in the index; the slice must stop at the
    # 06-30 close, exactly as if the frame had been truncated there.
    truncated = regime_feature(h.loc[:"2020-06-30"])
    assert regime_feature(h.loc[:"2020-07-01"]) != truncated   # the boundary is live on this data

    assert regime_feature(h, as_of=pd.Timestamp(AS_OF, tz="UTC")) == truncated
    assert regime_feature(h, as_of=pd.Timestamp(AS_OF)) == truncated
    assert regime_feature(h, as_of=datetime(2020, 7, 1, 0, 30)) == truncated
    assert regime_feature(_new_york(h), as_of=pd.Timestamp(AS_OF, tz="UTC")) == truncated
