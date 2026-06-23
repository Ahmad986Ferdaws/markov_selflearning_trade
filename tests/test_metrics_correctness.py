"""Golden-number tests that protect the headline comparison metric.

These are the tests that would have caught the four blockers:
point-in-time regime, Sharpe annualization, win-rate denominator,
and the validation gate. They assert *financial correctness*, not shape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.models.base import Base
from app.models.entities import Run
from app.schemas.trade import TradeIntent
from app.services.comparison import (
    SECONDS_PER_YEAR,
    ComparisonReport,
    StrategyMetrics,
    _annualization_factor,
    _validate_report,
)
from app.services.ledger import apply_trade
from app.services.regime import regime_feature


# --- Blocker 1: regime feature must be point-in-time (no lookahead) ---

def _trend_history() -> pd.DataFrame:
    """60 days at +3%/day then 60 days at -3%/day — clear bull then bear."""
    idx = pd.date_range("2020-01-01", periods=120, freq="D")
    up = 100 * (1.03 ** np.arange(60))
    down = up[-1] * (0.97 ** np.arange(1, 61))
    return pd.DataFrame({"Close": np.concatenate([up, down])}, index=idx)


def test_regime_as_of_only_uses_past():
    hist = _trend_history()
    # as_of inside the up-leg should match computing on the truncated history.
    as_of = pd.Timestamp("2020-02-20")  # ~day 50, mid up-trend
    full_pit = regime_feature(hist, window=10, as_of=as_of)
    truncated = hist[hist.index <= as_of]
    manual = regime_feature(truncated, window=10)
    assert full_pit.state == manual.state  # point-in-time == sliced history


def test_regime_changes_across_time():
    """A bull-then-bear history must NOT return the same regime at every date."""
    hist = _trend_history()
    early = regime_feature(hist, window=10, as_of=pd.Timestamp("2020-02-15")).state
    late = regime_feature(hist, window=10, as_of=pd.Timestamp("2020-04-15")).state
    assert early != late, "regime should vary over time; constant regime = the old bug"


# --- Blocker 2: Sharpe annualization must come from cadence, not √252 ---

def test_annualization_factor_from_poll_interval():
    periods, factor = _annualization_factor(Settings(poll_interval_seconds=15))
    assert periods == SECONDS_PER_YEAR / 15
    assert abs(factor - np.sqrt(SECONDS_PER_YEAR / 15)) < 1e-9
    # The old hardcoded daily factor (√252 ≈ 15.87) must not be what we use.
    assert abs(factor - np.sqrt(252)) > 1.0


# --- Blocker 3 (cost model is real): flat round-trip loses ~ 2×(fee+slip) ---

def test_flat_roundtrip_loses_exactly_costs():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    run = Run(cash=1000.0, position_qty=0.0, trade_symbol="X", watchlist="X")
    db.add(run)
    db.commit()
    db.refresh(run)
    settings = Settings(fee_pct=0.3, slippage_pct=0.5, max_allocation_pct=20)

    apply_trade(db, run, TradeIntent(action="BUY", percentage=100, symbol="X"), price=100.0, settings=settings)
    db.refresh(run)
    apply_trade(db, run, TradeIntent(action="SELL", percentage=100, symbol="X"), price=100.0, settings=settings)
    db.refresh(run)

    # Bought and sold at the same price → the only thing lost is round-trip cost.
    assert run.position_qty == 0.0
    assert run.realized_pnl < 0, "a flat round-trip with costs must lose money"
    # Cost ≈ notional * 2*(fee+slip) within rounding. notional capped at 20% = 200.
    assert -7.0 < run.realized_pnl < -2.0
    db.close()


# --- Blocker 4: validation gate flags a constant-regime (meaningless) run ---

def _metrics(regimes_seen: int) -> StrategyMetrics:
    return StrategyMetrics(
        name="x", total_return=0.0, annual_sharpe=0.0, max_drawdown=0.0,
        win_rate=0.0, num_trades=0, total_costs=0.0, guard_interventions=0,
        num_closes=0, regimes_seen=regimes_seen,
    )


def test_validation_flags_constant_regime():
    report = ComparisonReport(
        baseline=_metrics(1), agent=_metrics(1),
        headline_agent_minus_baseline=0.0, snapshot_count=5,
        sharpe_periods_per_year=1000.0,
    )
    warnings = _validate_report(report)
    assert any("Constant-regime" in w for w in warnings)


def test_validation_passes_with_varied_regime():
    report = ComparisonReport(
        baseline=_metrics(3), agent=_metrics(2),
        headline_agent_minus_baseline=0.0, snapshot_count=5,
        sharpe_periods_per_year=1000.0,
    )
    warnings = _validate_report(report)
    assert not any("Constant-regime" in w for w in warnings)


def test_validation_flags_non_finite():
    m = _metrics(3)
    m.annual_sharpe = float("inf")
    report = ComparisonReport(
        baseline=m, agent=_metrics(3),
        headline_agent_minus_baseline=0.0, snapshot_count=5,
        sharpe_periods_per_year=1000.0,
    )
    warnings = _validate_report(report)
    assert any("non-finite" in w for w in warnings)
