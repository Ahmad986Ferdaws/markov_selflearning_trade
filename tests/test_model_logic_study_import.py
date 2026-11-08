"""Importing scripts/model_logic_study.py must leave the filesystem alone.

The study used to create results/model_logic/ at module import (a bare
`OUT.mkdir` at the top of the file). scripts/logic_depth_study.py imports
`_configs` from this module, so merely importing THAT script dropped a stray
results/model_logic/ tree into the current directory too. The mkdir now runs
in main(), right before the one file the study writes.
"""

from __future__ import annotations

import importlib

import scripts.model_logic_study as study


def test_importing_the_study_creates_no_results_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    importlib.reload(study)
    assert not (tmp_path / "results").exists()


def test_the_bridge_function_is_still_importable():
    assert callable(study._configs)
    assert study.OUT.as_posix() == "results/model_logic"
