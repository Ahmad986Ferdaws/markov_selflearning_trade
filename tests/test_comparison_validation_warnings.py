"""_validate_report emits each report-level warning once (app/services/comparison.py).

The Sharpe-annualization check describes the report, not a strategy, yet it
used to sit inside the per-strategy loop and was appended twice. These pins
build a ComparisonReport by hand so no replay, DB or network is involved.
"""

from __future__ import annotations

from app.services.comparison import ComparisonReport, StrategyMetrics, _validate_report


def _metrics(name: str, regimes_seen: int = 3) -> StrategyMetrics:
    return StrategyMetrics(
        name=name, total_return=0.0, annual_sharpe=0.0, max_drawdown=0.0,
        win_rate=0.0, num_trades=0, total_costs=0.0, guard_interventions=0,
        num_closes=0, regimes_seen=regimes_seen,
    )


def _report(snapshot_count: int, periods_per_year: float,
            baseline: StrategyMetrics | None = None) -> ComparisonReport:
    return ComparisonReport(
        baseline=baseline or _metrics("baseline"),
        agent=_metrics("agent"),
        headline_agent_minus_baseline=0.0,
        snapshot_count=snapshot_count,
        sharpe_periods_per_year=periods_per_year,
    )


def test_unresolved_annualization_is_warned_about_exactly_once_per_report():
    warnings = _validate_report(_report(snapshot_count=2, periods_per_year=0.0))
    annualization = [w for w in warnings if "annualization" in w]
    assert len(annualization) == 1
    assert "poll_interval_seconds" in annualization[0]


def test_a_single_snapshot_report_gets_no_annualization_warning():
    warnings = _validate_report(_report(snapshot_count=1, periods_per_year=0.0))
    assert not [w for w in warnings if "annualization" in w]


def test_a_non_finite_metric_still_gets_its_own_warning_next_to_the_single_annualization_one():
    broken = _metrics("baseline")
    broken.annual_sharpe = float("nan")
    warnings = _validate_report(_report(snapshot_count=2, periods_per_year=0.0, baseline=broken))
    assert [w for w in warnings if "non-finite" in w] == ["baseline.annual_sharpe is non-finite (nan)."]
    assert len([w for w in warnings if "annualization" in w]) == 1
