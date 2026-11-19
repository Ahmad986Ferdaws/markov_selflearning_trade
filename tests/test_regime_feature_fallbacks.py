"""Two fallbacks in app/services/regime.py: regime_feature on a history too
short to label, and the frame shaping fetch_daily_history applies to whatever
yfinance returns.

No network: yf.download is monkeypatched to a recorder that hands back a
prepared frame and remembers how it was called.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services import regime
from app.services.regime import STATE_ORDER, fetch_daily_history, regime_feature


def _closes(rows: int) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="D")
    return pd.DataFrame({"Close": np.linspace(100.0, 150.0, rows)}, index=idx)


# --- regime_feature: fewer returns than the (default, 20-day) window ------------
@pytest.mark.parametrize("rows", [2, 10, 20])     # 1, 9 and 19 returns: no label possible
def test_a_history_shorter_than_the_window_is_sideways_with_a_uniform_forecast(rows):
    feat = regime_feature(_closes(rows))
    assert feat.state == "sideways"
    assert set(feat.p_next) == set(STATE_ORDER)
    assert all(p == pytest.approx(1 / 3) for p in feat.p_next.values())
    assert sum(feat.p_next.values()) == pytest.approx(1.0)


# --- fetch_daily_history: shaping the download -----------------------------------
def _patch_download(monkeypatch, frame: pd.DataFrame) -> list[tuple]:
    calls: list[tuple] = []

    def fake_download(symbol, **kwargs):
        calls.append((symbol, kwargs))
        return frame

    monkeypatch.setattr(regime.yf, "download", fake_download)
    return calls


def test_an_empty_download_is_refused_with_the_symbol_in_the_message(monkeypatch):
    _patch_download(monkeypatch, pd.DataFrame())
    with pytest.raises(ValueError, match="FAKE-USD"):
        fetch_daily_history("FAKE-USD")


def test_multiindex_columns_are_flattened_to_their_level_0_names(monkeypatch):
    columns = pd.MultiIndex.from_tuples([("Close", "BTC-USD"), ("Open", "BTC-USD")])
    frame = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=columns,
                         index=pd.date_range("2024-01-01", periods=2, freq="D"))
    _patch_download(monkeypatch, frame)
    out = fetch_daily_history("BTC-USD")
    assert not isinstance(out.columns, pd.MultiIndex)
    assert list(out.columns) == ["Close", "Open"]


@pytest.mark.parametrize("years", [1, 20])
def test_download_asks_for_daily_adjusted_bars_over_years_y(monkeypatch, years):
    calls = _patch_download(monkeypatch, _closes(3))
    fetch_daily_history("SPY", years=years)
    assert calls == [("SPY", dict(period=f"{years}y", interval="1d",
                                  progress=False, auto_adjust=True))]


def test_the_default_history_length_is_three_years(monkeypatch):
    calls = _patch_download(monkeypatch, _closes(3))
    fetch_daily_history("SPY")
    [(_, kwargs)] = calls
    assert kwargs["period"] == "3y"
