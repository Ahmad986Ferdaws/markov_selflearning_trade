"""Importing scripts/logic_depth_study.py must leave the filesystem alone.

The study used to create results/logic_depth/ at module import (a bare
`OUT.mkdir` at the top of the file), so any tool or test that merely imported
it dropped a stray results/ tree into the current directory. The mkdir now
runs in main(), right before the one file the study writes.

Only results/logic_depth is asserted on: importing this module also imports
scripts.model_logic_study, whose own import-time mkdir is a sibling fix.
"""

from __future__ import annotations

import importlib

import scripts.logic_depth_study as study


def test_importing_the_study_creates_no_logic_depth_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    importlib.reload(study)
    assert not (tmp_path / "results" / "logic_depth").exists()


def test_the_module_still_exposes_its_entry_points():
    assert callable(study.main) and callable(study.analyze)
    assert study.OUT.as_posix() == "results/logic_depth"
