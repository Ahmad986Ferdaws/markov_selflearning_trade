"""perf_metrics' Sortino must use the downside deviation sqrt(mean(min(r,0)^2))
over all returns. It used np.std of the negative returns about their own mean,
which measures the dispersion of the losses, not their size: for similar-sized
losses the denominator collapsed to a floating-point residue (4.6e16 for a
strategy whose Sortino is 11.2) and on ordinary returns it was biased ~15%.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from kalman.research import BARS_PER_YEAR, perf_metrics


def _ledger(rets) -> pd.DataFrame:
    r = np.concatenate([[0.0], np.asarray(rets, float)])   # forced 0.0 at bar 0
    eq = 100.0 * np.cumprod(1.0 + r)
    return pd.DataFrame({"equity": eq, "ret": r, "gross": np.ones_like(eq),
                         "turnover": np.zeros_like(eq), "costs": np.zeros_like(eq)})


def _textbook_sortino(rets) -> float:
    rets = np.asarray(rets, float)
    return float(np.mean(rets) / math.sqrt(np.mean(np.minimum(rets, 0.0) ** 2)) * math.sqrt(BARS_PER_YEAR))


def test_alternating_loss_gain_matches_the_closed_form():
    rets = [-0.01, 0.02] * 100
    m = perf_metrics(_ledger(rets))
    assert m["sortino"] == pytest.approx(11.22, abs=0.01)     # was 4.6e16
    assert m["sortino"] == pytest.approx(_textbook_sortino(rets), rel=1e-12)
    assert m["sharpe"] == pytest.approx(5.28, abs=0.01)       # unchanged


def test_identical_losses_no_longer_explode():
    rets = [-0.01] * 50 + [0.03] * 50
    m = perf_metrics(_ledger(rets))
    assert np.isfinite(m["sortino"]) and m["sortino"] == pytest.approx(_textbook_sortino(rets), rel=1e-12)


def test_gaussian_returns_match_downside_deviation_definition():
    rets = np.random.default_rng(3).normal(-0.0002, 0.01, 800)
    m = perf_metrics(_ledger(rets))
    assert m["sortino"] == pytest.approx(_textbook_sortino(rets), rel=1e-12)
    assert (m["sortino"] < 0) == (np.mean(rets) < 0)           # sign follows the sample mean


def test_no_downside_is_undefined_not_zero():
    m = perf_metrics(_ledger([0.01] * 40))
    assert math.isnan(m["sortino"])                            # like calmar when undefined
    assert np.isfinite(m["sharpe"]) and m["sharpe"] == 0.0     # zero spread -> engine's 0.0 rule


def test_other_metrics_untouched():
    rets = np.random.default_rng(5).normal(0.0003, 0.012, 500)
    m = perf_metrics(_ledger(rets))
    assert m["vol"] == pytest.approx(float(np.std(rets, ddof=1)) * math.sqrt(BARS_PER_YEAR))
    assert m["total_return"] == pytest.approx(float(np.prod(1 + rets) - 1.0))
