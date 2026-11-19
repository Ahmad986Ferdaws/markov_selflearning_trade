"""Importing scripts/long_history_study.py must leave the filesystem alone.

The study used to create results/long_history/ at module import (a bare
`OUT.mkdir` at the top of the file), so any tool or test that merely imported
it dropped a stray results/ tree into whatever the current directory was. The
mkdir now lives in the two functions that actually write files.
"""

from __future__ import annotations

import importlib

import scripts.long_history_study as study


def test_importing_the_study_creates_no_results_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    importlib.reload(study)
    assert not (tmp_path / "results").exists()


def test_the_module_still_exposes_its_eight_symbol_year_pairs():
    assert len(study.SYMBOLS) == 8
    assert len(set(study.SYMBOLS)) == 8
    assert all(isinstance(sym, str) and isinstance(years, int) for sym, years in study.SYMBOLS)
    assert study.OUT.as_posix() == "results/long_history"
