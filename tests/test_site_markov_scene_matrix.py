"""The site's Markov scene draws the engine's BTC transition matrix, not invented odds.

design/index.html hard-codes a 3x3 matrix ``M`` that the 3D scene turns into node
labels ("88% stay") and edge weights. design/README.md promises the page shows
engine data, never an invented series, so ``M`` must be the full-sample MLE the
engine computes for the headline configuration on the pinned BTC-USD 3y snapshot
(window 30, k 0.2), and its diagonal must be the ``matrix_diag`` that
results/model_logic/summary.json publishes for that run. Regexes only on the
page; the engine functions do the recomputation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

import app
from app.services.regime import STATE_ORDER, define_states, estimate_transition_matrix

ROOT = Path(app.__file__).resolve().parent.parent
INDEX = ROOT / "design" / "index.html"
PINNED = ROOT / "data" / "snapshots" / "BTC-USD_3y.pkl"
MODEL_LOGIC = ROOT / "results" / "model_logic" / "summary.json"

WINDOW, K = 30, 0.2
PAGE_KEY = {"bull": "bull", "bear": "bear", "sideways": "side"}   # engine label -> page key
TOLERANCE = 0.005                                                  # the page rounds to 2 dp

_MATRIX = re.compile(r"const M=(\{.*?\});")
_ROW = re.compile(r"(bull|side|bear):\{([^}]*)\}")
_CELL = re.compile(r"(bull|side|bear):(-?\d*\.?\d+)")


def _page_matrix() -> dict[str, dict[str, float]]:
    found = _MATRIX.findall(INDEX.read_text(encoding="utf-8"))
    assert len(found) == 1, f"expected exactly one 'const M=' in the page, found {len(found)}"
    rows = {row: {cell: float(v) for cell, v in _CELL.findall(body)} for row, body in _ROW.findall(found[0])}
    assert set(rows) == {"bull", "side", "bear"}
    assert all(set(r) == {"bull", "side", "bear"} for r in rows.values()), rows
    return rows


def _engine_matrix() -> dict[str, dict[str, float]]:
    closes = pd.read_pickle(PINNED)["Close"]
    states = define_states(closes.pct_change(fill_method=None).dropna(), window=WINDOW, k=K).dropna()
    matrix = estimate_transition_matrix(states)
    return {
        PAGE_KEY[src]: {PAGE_KEY[dst]: float(matrix[i, j]) for j, dst in enumerate(STATE_ORDER)}
        for i, src in enumerate(STATE_ORDER)
    }


def _published_diag() -> list[float]:
    runs = json.loads(MODEL_LOGIC.read_text(encoding="utf-8"))["runs"]
    hits = [r for r in runs if r["symbol"] == "BTC-USD" and r["years"] == 3]
    assert len(hits) == 1, f"expected one BTC-USD 3y run in model_logic, found {len(hits)}"
    assert (hits[0]["window"], hits[0]["k"]) == (WINDOW, K)
    return hits[0]["matrix_diag"]


def test_the_page_matrix_rows_sum_to_one_up_to_two_dp_rounding():
    # three cells each rounded to 2 dp can miss 1.0 by at most 3 * 0.005
    for row, cells in _page_matrix().items():
        assert sum(cells.values()) == pytest.approx(1.0, abs=3 * TOLERANCE), f"row {row!r} sums to {sum(cells.values())}"


def test_every_cell_of_the_page_matrix_is_the_engine_mle_on_the_pinned_snapshot():
    page, engine = _page_matrix(), _engine_matrix()
    for src in ("bull", "side", "bear"):
        for dst in ("bull", "side", "bear"):
            assert page[src][dst] == pytest.approx(engine[src][dst], abs=TOLERANCE), (
                f"M[{src}][{dst}] is {page[src][dst]} on the page but {engine[src][dst]:.4f} in the engine"
            )


def test_the_page_diagonal_is_the_matrix_diag_model_logic_publishes_for_btc_3y():
    page = _page_matrix()
    page_diag = [round(page[PAGE_KEY[s]][PAGE_KEY[s]], 2) for s in STATE_ORDER]
    assert page_diag == [round(v, 2) for v in _published_diag()]


def test_the_scene_label_still_reads_the_diagonal_as_a_stay_percentage():
    page = INDEX.read_text(encoding="utf-8")
    assert "${(M[k][k]*100|0)}% stay" in page


def test_zero_probability_edges_carry_no_particles():
    # the engine matrix has exact zeros (bull->bear, bear->bull). The faint arcs are the graph's
    # skeleton and never encoded odds; the moving particles do, so a zero cell must draw none.
    assert any(v == 0 for row in _page_matrix().values() for v in row.values()), "no zero cell left to guard"
    page = INDEX.read_text(encoding="utf-8")
    assert "edges.forEach(e=>{ if(e.p<=0) return; const cnt=Math.round(e.p*7)+1;" in page
