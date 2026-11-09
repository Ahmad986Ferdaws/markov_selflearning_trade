"""scripts.model_logic_study._configs() is the bridge from the committed run
records to the two follow-up studies: it reads (symbol, years, window, k,
data_hash) out of results/robustness and results/long_history so each study
re-analyses exactly the configuration the headline numbers were produced
with, on exactly the snapshot they were produced from. Pin what it reads
today, so a renamed, dropped or re-run record fails here before it silently
changes a study.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import app
from scripts.model_logic_study import _configs

ROOT = Path(app.__file__).resolve().parent.parent
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
GRID_WINDOWS = (10, 20, 30)
GRID_K = (0.2, 0.35, 0.5, 0.75)


@pytest.fixture(autouse=True)
def _at_repo_root(monkeypatch):
    monkeypatch.chdir(ROOT)      # _configs() globs results/ relative to the cwd


def test_it_reads_twenty_robustness_and_eight_long_history_records():
    cfgs = _configs()
    assert len(cfgs) == 28
    robustness = [c for c in cfgs if c[1] == 3]
    long_history = [c for c in cfgs if c[1] != 3]
    assert (len(robustness), len(long_history)) == (20, 8)
    assert all(years in (10, 12, 20) for _, years, *_ in long_history)
    assert len({(sym, years) for sym, years, *_ in cfgs}) == 28     # one config per (symbol, span)


def test_it_skips_the_sub_period_slices():
    assert all(":" not in sym for sym, *_ in _configs())


def test_every_config_lies_on_the_published_grid_with_a_full_hash():
    for sym, years, window, k, digest in _configs():
        assert isinstance(window, int) and window in GRID_WINDOWS, (sym, years, window)
        assert isinstance(k, float) and k in GRID_K, (sym, years, k)
        # the full sha256, not a prefix: analyze() compares it to history_hash()
        assert SHA256_HEX.match(digest), (sym, years, digest)


def test_the_headline_btc_config_is_the_one_the_run_record_names():
    (btc,) = [c for c in _configs() if c[:2] == ("BTC-USD", 3)]
    assert btc[2:4] == (30, 0.2)
    assert btc[4].startswith("4a150b23")
