"""scripts/_verify_divergence.py used to run its 20-asset study at import time
(a module-level loop over SYMS, no __main__ guard), so it could not be
imported, linted meaningfully, or tested. The study now lives in main():
importing the module is cheap, loads no snapshot, and exposes the pieces a
test can exercise on synthetic data.
"""

from __future__ import annotations

import importlib
import sys
import time

import numpy as np
import pandas as pd

from app.services import data_cache
from app.services.regime import STATE_ORDER

MODULE = "scripts._verify_divergence"


def _explode(*args, **kwargs):
    raise AssertionError("importing the verification script must not load any snapshot")


def _fresh_import(monkeypatch):
    monkeypatch.setattr(data_cache, "load_or_fetch", _explode)
    monkeypatch.delitem(sys.modules, MODULE, raising=False)
    started = time.perf_counter()
    module = importlib.import_module(MODULE)
    return module, time.perf_counter() - started


def test_import_is_cheap_loads_no_data_and_exposes_the_entry_points(monkeypatch):
    module, elapsed = _fresh_import(monkeypatch)
    assert elapsed < 1.0
    assert callable(module.main)
    assert callable(module.walk)
    assert callable(module.select_and_test)
    assert len(module.SYMS) == 20


def test_select_and_test_counts_exactly_the_divergences_it_lists(monkeypatch):
    module, _ = _fresh_import(monkeypatch)
    returns = pd.Series(np.random.default_rng(0).normal(0.0, 0.01, 240))
    out = module.select_and_test(returns)
    assert isinstance(out, tuple) and len(out) == 5
    window, k, n_pred, divergences, indices = out
    assert window in module.GRID_W and k in module.GRID_K
    assert 0 <= divergences <= n_pred
    assert divergences == len(indices)
    assert all(0 <= i < n_pred for i in indices)


def test_walk_flags_every_step_on_a_planted_alternation(monkeypatch):
    module, _ = _fresh_import(monkeypatch)
    bull, bear, _ = STATE_ORDER
    states = [bull, bear] * 100
    preds, actual, cur = module.walk(pd.Series(np.zeros(len(states))), states, 100, len(states))
    assert len(preds) == len(actual) == len(cur) == 99
    assert all(p != c for p, c in zip(preds, cur))     # the argmax leaves the diagonal every day
    assert preds == actual                              # and is right every time
