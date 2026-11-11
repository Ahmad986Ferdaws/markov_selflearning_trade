"""scripts/logic_depth_study.py's two data preconditions survive python -O.

The module docstring promises that for EVERY run the incremental estimator is
checked against the engine and the label sequence is checked gap-free. Both
checks used to be bare `assert` statements, which the interpreter strips
under -O, silently disabling exactly the guarantees the docstring makes.
They are explicit `raise ValueError(...)` checks now (the rule kalman/ledger.py
already applies), with the same messages, plus a clear error for an all-None
label list instead of an IndexError. analyze() is called in-process on
synthetic data with load_or_fetch patched out; main() is never called and
nothing is written.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.logic_depth_study as study

SCRIPT = Path(study.__file__)


def _synthetic_history(n: int = 300) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    steps = np.random.default_rng(21).normal(0.0, 0.01, n)
    return pd.DataFrame({"Close": 100.0 * np.exp(np.cumsum(steps)), "Volume": 1.0}, index=idx)


def _patch_history(monkeypatch, frame: pd.DataFrame) -> None:
    monkeypatch.setattr(study, "load_or_fetch", lambda symbol, years: (frame, None))


def test_the_script_contains_no_bare_assert_statement():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    asserts = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Assert)]
    assert asserts == []


def test_a_mid_series_label_gap_is_a_value_error_not_an_assertion_error(monkeypatch):
    frame = _synthetic_history()
    _patch_history(monkeypatch, frame)
    n_labels = len(frame) - 1                       # one label per return
    bull, bear, _ = study.STATE_ORDER

    def gappy(returns, window, k):
        labels = [None] * 10 + [bull] * (n_labels - 10)
        labels[150] = None                          # the gap
        labels[200:] = [bear] * (n_labels - 200)
        return labels

    monkeypatch.setattr(study, "causal_states", gappy)
    with pytest.raises(ValueError, match="SYN: mid-series label gap"):
        study.analyze("SYN", 3, 10, 0.5)


def test_an_all_none_label_list_is_a_clear_value_error_not_an_index_error(monkeypatch):
    frame = _synthetic_history()
    _patch_history(monkeypatch, frame)
    monkeypatch.setattr(study, "causal_states", lambda returns, window, k: [None] * (len(frame) - 1))
    with pytest.raises(ValueError, match="SYN: no labeled bars after warm-up"):
        study.analyze("SYN", 3, 10, 0.5)


def test_estimator_drift_is_a_value_error_not_an_assertion_error(monkeypatch):
    _patch_history(monkeypatch, _synthetic_history())
    wrong = np.full((study.N_STATES, study.N_STATES), 1.0 / study.N_STATES)
    monkeypatch.setattr(study, "estimate_transition_matrix", lambda observed: wrong)
    with pytest.raises(ValueError, match=r"SYN: estimator drift at t=\d+"):
        study.analyze("SYN", 3, 10, 0.5)


def test_the_happy_path_still_runs_both_checks_and_returns_the_summary_row(monkeypatch):
    frame = _synthetic_history()
    _patch_history(monkeypatch, frame)
    calls = []
    real = study.estimate_transition_matrix
    monkeypatch.setattr(study, "estimate_transition_matrix",
                        lambda observed: calls.append(len(observed)) or real(observed))
    out = study.analyze("SYN", 3, 10, 0.5)
    assert out["symbol"] == "SYN" and out["window"] == 10 and out["k"] == 0.5
    assert out["n_scored"] > 0 and 0 <= out["divergences"] <= out["n_scored"]
    assert len(calls) >= 1                          # the estimator self-check executed
