"""Pins for the walk-forward fold layout: the exact (train, validation, test)
index tuples for expanding and rolling training windows, that every variant is
evaluated on every fold with a grid-chosen entry threshold, and that the result
dictionary has the JSON-able shape the CLI writes.

tests/test_kalman_research.py pins the report's determinism on this same
layout; this file pins the indices themselves. The two walk_forward calls run
once per module.
"""

from __future__ import annotations

import json

import pytest

from kalman.data import synthetic_pair
from kalman.ledger import LedgerConfig
from kalman.research import VARIANTS, WalkForwardConfig, walk_forward

# statsmodels' adfuller deprecation is raised from kalman/pairs.py on every
# qualify_pair call and is already visible, unfiltered, in
# tests/test_kalman_research.py; it is not repeated here.
pytestmark = pytest.mark.filterwarnings("ignore:adfuller currently returns:FutureWarning")

EXPANDING = [((0, 400), (400, 520), (520, 640)),
             ((0, 520), (520, 640), (640, 760)),
             ((0, 640), (640, 760), (760, 880))]
ROLLING = [((0, 400), (400, 520), (520, 640)),
           ((120, 520), (520, 640), (640, 760)),
           ((240, 640), (640, 760), (760, 880))]


@pytest.fixture(scope="module")
def results() -> dict:
    pair, _ = synthetic_pair(n=900, seed=4)
    sizes = dict(train=400, validation=120, test=120)
    return {
        "expanding": walk_forward(pair, WalkForwardConfig(**sizes), LedgerConfig()),
        "rolling": walk_forward(pair, WalkForwardConfig(expanding=False, **sizes),
                                LedgerConfig()),
    }


def _layout(res: dict) -> list:
    return [(f["train"], f["validation"], f["test"]) for f in res["folds"]]


def test_expanding_folds_anchor_the_training_start_at_zero(results):
    assert _layout(results["expanding"]) == EXPANDING
    assert [f["fold"] for f in results["expanding"]["folds"]] == [0, 1, 2]


def test_rolling_folds_advance_the_training_start_by_the_test_length(results):
    assert _layout(results["rolling"]) == ROLLING
    # only the train start moves: train ends, validation and test are shared
    assert [(tr[1], va, te) for tr, va, te in ROLLING] == \
        [(tr[1], va, te) for tr, va, te in EXPANDING]
    # a fourth fold would need bar 1000 > 900, so there are exactly three
    assert len(results["rolling"]["folds"]) == 3


@pytest.mark.parametrize("kind", ["expanding", "rolling"])
def test_every_variant_has_one_metric_dict_per_fold_with_a_grid_entry(results, kind):
    res = results[kind]
    grid = WalkForwardConfig().entry_exit_grid
    assert set(res["by_variant"]) == set(VARIANTS) == set(res["aggregate"])
    for v in VARIANTS:
        ms = res["by_variant"][v]
        assert len(ms) == 3
        for m in ms:
            assert (m["chosen_entry_z"], m["chosen_exit_z"]) in grid
        assert res["aggregate"][v]["folds"] == 3


@pytest.mark.parametrize("kind", ["expanding", "rolling"])
def test_result_drops_test_returns_and_is_json_serialisable(results, kind):
    res = results[kind]
    assert "test_returns" not in res
    assert set(res) == {"folds", "by_variant", "aggregate"}
    back = json.loads(json.dumps(res, default=float))
    assert [list(f["train"]) for f in back["folds"]] == [list(t) for t, _, _ in _layout(res)]
    assert set(back["aggregate"]) == set(VARIANTS)
