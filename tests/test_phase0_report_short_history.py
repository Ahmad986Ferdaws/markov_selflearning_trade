"""run_phase0_report (app/services/regime.py) says when the backtest never scored a day.

walk_forward_backtest returns an all-zero BacktestResult when fewer than two
steps were scored. The report used to print that as
``No cost  — Sharpe: 0.000, Return: 0.00%, MaxDD: 0.00%``, an empty regime
mix and, because 0.0 >= 0.0 with costs > 0, the NOTE about costs failing to
lower Sharpe: a backtest that never ran dressed up as a zero-return result
with a cost anomaly. It now prints one "no scored day" line instead. The
long-history path is unchanged. load_or_fetch is monkeypatched, so no
snapshot is read and nothing is downloaded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.services import data_cache
from app.services.regime import run_phase0_report, walk_forward_backtest


def _history(rows: int, seed: int = 0) -> pd.DataFrame:
    """A seeded random-walk Close frame with `rows` daily bars."""
    rng = np.random.default_rng(seed)
    closes = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, rows))
    return pd.DataFrame({"Close": closes},
                        index=pd.date_range("2020-01-01", periods=rows, freq="D"))


def _serve(monkeypatch, frame: pd.DataFrame) -> None:
    # run_phase0_report imports load_or_fetch lazily, so the patch on the
    # data_cache module is what it resolves at call time.
    monkeypatch.setattr(data_cache, "load_or_fetch",
                        lambda symbol, years=3, **kw: (frame, "synthetic"))


def test_a_history_too_short_to_score_a_day_is_flagged_not_reported_as_zero_sharpe(monkeypatch):
    frame = _history(40)
    assert len(walk_forward_backtest(frame).returns) == 0       # the premise: nothing scored
    _serve(monkeypatch, frame)

    out = run_phase0_report(symbol="SHORT-USD")

    assert "Backtest: no scored day — 39 daily returns; the walk-forward needs more than min_train=60" in out
    assert "Sharpe:" not in out
    assert "Regime mix:" not in out
    assert "NOTE: Costs" not in out
    assert "Phase 0 Regime Report: SHORT-USD" in out               # the header is still there


def test_a_long_history_still_reports_both_sharpe_lines_and_the_note_only_when_earned(monkeypatch):
    frame = _history(400)
    _serve(monkeypatch, frame)

    out = run_phase0_report(symbol="LONG-USD")

    assert "No cost  — Sharpe:" in out
    assert "With cost — Sharpe:" in out
    assert "Regime mix:" in out
    assert "no scored day" not in out
    bt_no_cost = walk_forward_backtest(frame, fee_pct=0, slippage_pct=0)
    bt_cost = walk_forward_backtest(frame, fee_pct=0.3, slippage_pct=0.5)
    assert len(bt_cost.returns) >= 2
    assert ("NOTE: Costs should typically lower Sharpe vs no-cost run." in out) == (
        bt_cost.annual_sharpe >= bt_no_cost.annual_sharpe)
