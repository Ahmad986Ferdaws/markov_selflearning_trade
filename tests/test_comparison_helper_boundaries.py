"""The small helpers of app/services/comparison.py, pinned at their boundaries.

_snapshot_metrics falls back to a four-key dict whenever the stored payload is
not a JSON object; _trade_snapshots_for_run keeps only the run's own symbol in
order; _annualization_factor clamps a non-positive poll interval to one
second; format_report labels the Sharpe row by whether an annualization
factor exists. Snapshot and Run rows are built directly; no database is used.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from app.config import Settings
from app.models.entities import Run, Snapshot
from app.services.comparison import (
    SECONDS_PER_YEAR,
    ComparisonReport,
    StrategyMetrics,
    _annualization_factor,
    _snapshot_metrics,
    _trade_snapshots_for_run,
    format_report,
)

STAMP = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def _snapshot(symbol: str = "BTC-USD", price: float = 100.0, payload: str = "{}",
              created_at: datetime = STAMP) -> Snapshot:
    return Snapshot(run_id=1, symbol=symbol, price=price, volume=12.5, source="test",
                    payload_json=payload, created_at=created_at)


FALLBACK = {"symbol": "BTC-USD", "price": 100.0, "volume": 12.5, "timestamp": STAMP.isoformat()}


# --- _snapshot_metrics -----------------------------------------------------------------
def test_a_json_object_payload_is_returned_as_stored():
    payload = {"symbol": "BTC-USD", "price": 100.0, "rsi": 55.5}
    assert _snapshot_metrics(_snapshot(payload=json.dumps(payload))) == payload


@pytest.mark.parametrize("payload", [
    "not json",            # invalid JSON
    "[1, 2, 3]",           # a JSON value that is not an object
    "",                    # empty payload
])
def test_anything_but_a_json_object_falls_back_to_the_four_core_fields(payload):
    assert _snapshot_metrics(_snapshot(payload=payload)) == FALLBACK


def test_the_fallback_timestamp_is_the_snapshot_time_in_isoformat():
    metrics = _snapshot_metrics(_snapshot(payload=""))
    assert metrics["timestamp"] == "2024-01-02T03:04:05+00:00"


# --- _trade_snapshots_for_run ----------------------------------------------------------
def test_only_the_runs_trade_symbol_survives_and_order_is_kept():
    run = Run(id=1, watchlist="ETH-USD,BTC-USD", trade_symbol="BTC-USD")
    rows = [_snapshot("ETH-USD", 2000.0), _snapshot("BTC-USD", 100.0),
            _snapshot("ETH-USD", 2010.0), _snapshot("BTC-USD", 101.0), _snapshot("BTC-USD", 99.0)]
    kept = _trade_snapshots_for_run(run, rows)
    assert kept == [rows[1], rows[3], rows[4]]
    assert [s.price for s in kept] == [100.0, 101.0, 99.0]


def test_no_snapshot_of_the_trade_symbol_means_an_empty_list_not_a_fallback():
    run = Run(id=1, watchlist="ETH-USD", trade_symbol="BTC-USD")
    assert _trade_snapshots_for_run(run, [_snapshot("ETH-USD"), _snapshot("ETH-USD")]) == []


# --- _annualization_factor -------------------------------------------------------------
@pytest.mark.parametrize("interval", [0, -5])
def test_a_non_positive_poll_interval_is_clamped_to_one_second(interval):
    periods, factor = _annualization_factor(Settings(poll_interval_seconds=interval))
    assert periods == SECONDS_PER_YEAR
    assert factor == pytest.approx(SECONDS_PER_YEAR ** 0.5)


def test_a_one_second_interval_is_the_same_boundary_from_the_other_side():
    assert _annualization_factor(Settings(poll_interval_seconds=1))[0] == SECONDS_PER_YEAR
    assert _annualization_factor(Settings(poll_interval_seconds=2))[0] == SECONDS_PER_YEAR / 2


# --- format_report ---------------------------------------------------------------------
def _metrics(name: str) -> StrategyMetrics:
    return StrategyMetrics(name=name, total_return=0.0, annual_sharpe=0.0, max_drawdown=0.0,
                           win_rate=0.0, num_trades=0, total_costs=0.0, guard_interventions=0)


def _report(periods_per_year: float) -> ComparisonReport:
    return ComparisonReport(baseline=_metrics("baseline"), agent=_metrics("agent"),
                            headline_agent_minus_baseline=0.0, snapshot_count=0,
                            sharpe_periods_per_year=periods_per_year)


def test_the_sharpe_row_is_labelled_plainly_without_an_annualization_factor():
    rows = format_report(_report(0.0)).splitlines()
    [sharpe_row] = [r for r in rows if r.startswith("Sharpe")]
    assert sharpe_row.split()[0:1] == ["Sharpe"]
    assert "ann@" not in sharpe_row


def test_the_sharpe_row_names_the_annualization_factor_when_there_is_one():
    rows = format_report(_report(2_102_400.0)).splitlines()   # 15-second cadence
    [sharpe_row] = [r for r in rows if r.startswith("Sharpe")]
    assert sharpe_row.startswith("Sharpe (ann@2,102,400/yr)")
