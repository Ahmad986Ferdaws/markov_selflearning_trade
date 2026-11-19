"""Gate 6 of evaluate() (app/services/evaluation.py): when a policy named
'agent' is present and at least 25% of test-day forecasts put their most-likely
next regime below 1/3 + 0.15, a caveat is appended saying the agent acted on
low-confidence forecasts. Like every honesty gate it only appends — the agent
row is never dropped — and it keys on the literal policy name.

Offline and deterministic: a seeded random walk, and the transition-matrix
estimator monkeypatched to a uniform 3x3 so every forecast is exactly 1/3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import app.services.evaluation as evaluation
from app.services.evaluation import evaluate, policy_regime_baseline

GATE_PREFIX = ("Agent acted on low-confidence forecasts on 100% of test days "
               "(most-likely next regime below 48%)")


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    closes = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, 400)))
    return pd.DataFrame({"Close": closes}, index=pd.bdate_range("2019-01-01", periods=400))


def _half_long_agent(ctx) -> float:
    return 0.5


def _uniform_matrix(states) -> np.ndarray:
    return np.full((3, 3), 1 / 3)


def _gate_hits(report) -> list[str]:
    return [w for w in report.warnings if w.startswith("Agent acted")]


def test_uniform_forecasts_trip_gate_6_once_and_keep_the_agent_row(monkeypatch):
    monkeypatch.setattr(evaluation, "estimate_transition_matrix", _uniform_matrix)

    rep = evaluate(_frame(), policies={"agent": _half_long_agent, "baseline": policy_regime_baseline})

    hits = _gate_hits(rep)
    assert len(hits) == 1
    assert hits[0].startswith(GATE_PREFIX)
    assert hits[0].endswith("(a valid, honest finding).")
    assert [p.name for p in rep.policies] == ["agent", "baseline"]     # appended, never dropped


def test_gate_6_keys_on_the_literal_policy_name_agent(monkeypatch):
    monkeypatch.setattr(evaluation, "estimate_transition_matrix", _uniform_matrix)

    rep = evaluate(_frame(), policies={"llm": _half_long_agent, "baseline": policy_regime_baseline})

    assert _gate_hits(rep) == []


def test_the_real_estimator_does_not_trip_gate_6_on_this_data():
    rep = evaluate(_frame(), policies={"agent": _half_long_agent, "baseline": policy_regime_baseline})

    assert _gate_hits(rep) == []
    assert rep.accuracy.n == 119
