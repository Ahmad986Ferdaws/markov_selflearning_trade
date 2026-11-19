"""README and docs/19 make two claims about the logic-depth and model-logic
studies that no test pins yet (tests/test_instrument_validation.py already
pins pooled_predictions, divergences and the minimum self-probability):

  README:  the matrix carries calibration, "better log-loss than a sticky
           baseline, 28/28 runs";
  docs/19: "the obvious fixes" (duration hazards, second-order memory) fire
           almost never and win nothing.

Read the committed summaries the way a reader would and pin those numbers,
tying each aggregate to the per-run rows it is summed from.
"""

from __future__ import annotations

import json
from pathlib import Path

import app

ROOT = Path(app.__file__).resolve().parent.parent


def _summary(study: str) -> dict:
    return json.loads((ROOT / "results" / study / "summary.json").read_text(encoding="utf-8"))


# --- README: "better log-loss than a sticky baseline, 28/28 runs" -----------------
def test_the_markov_row_beats_the_sticky_baseline_on_log_loss_in_every_run():
    summary = _summary("logic_depth")
    agg, runs = summary["aggregate"], summary["runs"]
    assert agg["runs_markov_better_ll"] == 28 == len(runs)
    assert agg["mean_ll_improvement"] > 0
    assert all(r["ll_improvement"] > 0 for r in runs)


# --- docs/19: "the obvious fixes fire almost never" --------------------------------
def test_second_order_memory_fires_twenty_times_across_all_runs():
    summary = _summary("logic_depth")
    agg, runs = summary["aggregate"], summary["runs"]
    assert agg["total_m2_attempts"] == 20
    assert sum(r["m2"]["attempts"] for r in runs) == agg["total_m2_attempts"]


def test_the_duration_model_fires_on_five_runs_twenty_three_times_in_total():
    summary = _summary("model_logic")
    agg, runs = summary["aggregate"], summary["runs"]
    assert agg["total_duration_attempts"] == 23
    assert agg["runs_where_duration_model_fired"] == 5
    assert sum(r["duration_model"]["attempts"] for r in runs) == agg["total_duration_attempts"]
    assert sum(r["duration_model"]["attempts"] > 0 for r in runs) == agg["runs_where_duration_model_fired"]


def test_every_run_where_memory_two_fired_has_a_negligible_edge():
    runs = _summary("logic_depth")["runs"]
    firing = [r for r in runs if r["m2"]["attempts"] > 0]
    assert firing, "no run ever fired?"
    for r in firing:
        assert r["m2"]["edge"] is not None, r["symbol"]
        assert -0.02 <= r["m2"]["edge"] <= 0.01, (r["symbol"], r["m2"]["edge"])
