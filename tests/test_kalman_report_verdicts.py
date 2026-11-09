"""format_report's honesty wording (kalman/research.py): the WARNING when a
Kalman variant's mean innovation z-std leaves [0.5, 2.0] (inclusive bounds,
each Kalman variant named), and the two VERDICT sentences — "no robust edge"
unless kalman_fixed's test Sharpe beats both 0.5 and the cash row.

Hand-built result dicts, no walk-forward runs. test_kalman_zero_zstd.py pins
only how the z-std number prints; neither the warning text nor the positive
verdict had a test.
"""

from __future__ import annotations

import pytest

from kalman.research import format_report

NO_EDGE = "VERDICT: no robust edge after costs"
POSITIVE = "VERDICT: test Sharpe is positive"


def _row(sharpe: float, z_std) -> dict:
    return {"folds": 1, "mean_test_sharpe": sharpe, "sharpe_ci90": (None, None),
            "total_test_return": 0.0, "total_costs": 0.0, "trades": 0,
            "gated_bars": 0, "mean_z_std": z_std}


def _report(kalman_sharpe: float, cash_sharpe: float = 0.0, z_std=1.0,
            variant: str = "kalman_fixed") -> str:
    res = {"folds": [{"diagnostics": "d", "calibrated": {}}],
           "aggregate": {variant: _row(kalman_sharpe, z_std), "cash": _row(cash_sharpe, None)}}
    return format_report(res, "SYN")


def _warnings(report: str) -> list[str]:
    return [ln for ln in report.splitlines() if ln.startswith("WARNING")]


def test_header_names_the_fold_count_and_prints_fold_0_calibration_once():
    report = _report(0.0)
    assert "folds: 1  (train/val/test chronological, test untouched)" in report.splitlines()
    assert report.count("fold 0 calibration:") == 1


@pytest.mark.parametrize("z_std, shown", [(3.0, "3.00"), (0.49, "0.49")])
def test_a_z_std_outside_the_band_warns_with_the_value(z_std, shown):
    warnings = _warnings(_report(0.0, z_std=z_std))
    assert len(warnings) == 1
    assert warnings[0].startswith(f"WARNING: kalman_fixed innovation z-std is {shown} (far from 1)")


@pytest.mark.parametrize("z_std", [0.5, 2.0])
def test_the_band_bounds_are_inclusive(z_std):
    assert _warnings(_report(0.0, z_std=z_std)) == []


def test_the_warning_names_the_adaptive_variant_too():
    warnings = _warnings(_report(0.0, z_std=3.0, variant="kalman_adaptive"))
    assert len(warnings) == 1
    assert warnings[0].startswith("WARNING: kalman_adaptive innovation z-std is 3.00 (far from 1)")


@pytest.mark.parametrize("kalman_sharpe, cash_sharpe, verdict", [
    (0.5, 0.0, NO_EDGE),         # exactly 0.5 is not an edge
    (0.51, 0.0, POSITIVE),
    (0.7, 0.8, NO_EDGE),         # cash dominates
    (0.9, 0.8, POSITIVE),
])
def test_the_verdict_is_the_last_line_and_needs_a_sharpe_above_both_floors(
        kalman_sharpe, cash_sharpe, verdict):
    last = _report(kalman_sharpe, cash_sharpe).splitlines()[-1]
    assert last.startswith(verdict)
