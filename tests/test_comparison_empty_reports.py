"""run_comparison's two empty-report exits (app/services/comparison.py).

A run with no snapshots at all, and a run whose snapshots all belong to
another symbol, each return an all-zero report with one warning and never
reach the benchmark download. fetch_daily_history is replaced by a function
that raises, so reaching it would fail the test rather than hit the network.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.models.base import Base
from app.models.entities import Run, Snapshot
from app.services import comparison as comparison_module
from app.services.comparison import run_comparison


@pytest.fixture
def db(monkeypatch):
    def _no_network(*_args, **_kwargs):
        raise AssertionError("fetch_daily_history reached for a run with nothing to compare")

    monkeypatch.setattr(comparison_module, "fetch_daily_history", _no_network)
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def _assert_all_zero(report) -> None:
    assert (report.baseline.name, report.agent.name) == ("baseline", "agent")
    assert report.snapshot_count == 0
    assert report.headline_agent_minus_baseline == 0.0
    assert report.sharpe_periods_per_year == 0.0
    for metrics in (report.baseline, report.agent):
        assert (metrics.total_return, metrics.annual_sharpe, metrics.max_drawdown,
                metrics.win_rate, metrics.total_costs) == (0.0, 0.0, 0.0, 0.0, 0.0)
        assert (metrics.num_trades, metrics.guard_interventions,
                metrics.num_closes, metrics.regimes_seen) == (0, 0, 0, 0)


def test_a_run_without_snapshots_returns_an_all_zero_report_with_one_warning(db):
    db.add(Run(id=1, watchlist="BTC-USD", trade_symbol="BTC-USD", cash=1000.0, status="stopped"))
    db.commit()

    report = run_comparison(db, 1, Settings(anthropic_api_key=""))

    _assert_all_zero(report)
    assert report.warnings == ["No snapshots for run; nothing to compare."]


def test_a_run_whose_snapshots_are_all_another_symbol_refuses_to_mix(db):
    db.add(Run(id=1, watchlist="ETH-USD,BTC-USD", trade_symbol="BTC-USD", cash=1000.0, status="stopped"))
    db.add(Snapshot(run_id=1, symbol="ETH-USD", price=2000.0, volume=1.0, source="test"))
    db.add(Snapshot(run_id=1, symbol="ETH-USD", price=2010.0, volume=1.0, source="test"))
    db.commit()

    report = run_comparison(db, 1, Settings(anthropic_api_key=""))

    _assert_all_zero(report)
    [warning] = report.warnings
    assert "refusing to compare across mixed symbols" in warning
    assert "'BTC-USD'" in warning


def test_the_two_empty_reports_do_not_share_a_warning_list():
    first = comparison_module._empty_report("one")
    second = comparison_module._empty_report("two")
    assert (first.warnings, second.warnings) == (["one"], ["two"])
    assert first.warnings is not second.warnings
