"""Run scripts/independent_rederivation.py the way CI does, as a subprocess,
and pin its verdicts: the pinned snapshot reproduces the headline (exit 0),
a tampered copy is refused before scoring (exit 2)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

import app

ROOT = Path(app.__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "independent_rederivation.py"
PINNED = ROOT / "data" / "snapshots" / "BTC-USD_3y.pkl"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          cwd=ROOT, capture_output=True, text=True, timeout=120)


def test_the_pinned_snapshot_reproduces_the_headline():
    result = _run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "REPRODUCED" in result.stdout
    assert "predictions=328  switch_days=30  switch_attempts=0" in result.stdout
    assert result.stdout.startswith(f"snapshot={PINNED.relative_to(ROOT)}")


def test_a_tampered_snapshot_is_refused_before_any_scoring(tmp_path):
    hist = pd.read_pickle(PINNED)
    hist.iloc[-1, hist.columns.get_loc("Close")] += 1.0
    tampered = tmp_path / "tampered.pkl"
    hist.to_pickle(tampered)
    result = _run("--snapshot", str(tampered))
    assert result.returncode == 2, result.stdout + result.stderr
    assert "SNAPSHOT CHANGED" in result.stdout
    assert "REPRODUCED" not in result.stdout and "MISMATCH" not in result.stdout


def test_help_names_the_flag_and_the_exit_codes():
    result = _run("--help")
    assert result.returncode == 0
    assert "--snapshot" in result.stdout
    assert "exit codes: 0 reproduced, 1 mismatch, 2 snapshot changed" in result.stdout
