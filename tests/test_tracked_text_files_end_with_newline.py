"""`pre-commit run --all-files` must leave the committed tree untouched.

Two opt-in hooks rewrite files: end-of-file-fixer appends a missing final
newline to every tracked .py/.toml/.yml/.yaml/.md/.cff file, and
mixed-line-ending --fix=lf converts CRLF. docs/13-reframed-plan.md used to end
without a newline, and results/robustness/summary.csv is stored CRLF in the
index because .gitattributes only normalises when a file is added or
committed, never on checkout. The CSV is frozen evidence (the study no longer
writes it), so it carries `-text` in .gitattributes and is excluded from the
hook rather than rewritten; this file pins both halves so a future file cannot
re-introduce the drift.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import app

ROOT = Path(app.__file__).resolve().parent.parent
TEXT_SUFFIXES = {".py", ".toml", ".yml", ".yaml", ".md", ".cff"}
SKIPPED_DIRS = {".git", ".venv", "node_modules"}
FROZEN_CSV = ROOT / "results" / "robustness" / "summary.csv"


def _tracked_files() -> list[Path]:
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True,
                             capture_output=True).stdout
        names = [n for n in out.decode("utf-8").split("\0") if n]
        if names:
            return [ROOT / n for n in names]
    except (OSError, subprocess.CalledProcessError):
        pass
    return [p for p in ROOT.rglob("*")                 # no git: walk, skipping the usual
            if p.is_file() and not (SKIPPED_DIRS & set(p.relative_to(ROOT).parts))]


def test_every_tracked_text_file_ends_with_a_newline():
    offenders = [
        p.relative_to(ROOT).as_posix() for p in _tracked_files()
        if p.suffix in TEXT_SUFFIXES and p.is_file() and p.stat().st_size > 0
        and not p.read_bytes().endswith(b"\n")
    ]
    assert offenders == [], f"end-of-file-fixer would rewrite: {offenders}"


def test_the_scan_saw_the_repository():
    files = _tracked_files()
    assert ROOT / "docs" / "13-reframed-plan.md" in files
    assert FROZEN_CSV in files
    assert sum(p.suffix in TEXT_SUFFIXES for p in files) > 100


def test_the_robustness_summary_csv_is_still_stored_crlf():
    raw = FROZEN_CSV.read_bytes()
    assert b"\r\n" in raw                           # the published bytes, not a normalised copy
    assert raw.endswith(b"\r\n")


def test_the_mixed_line_ending_hook_excludes_the_frozen_csv():
    cfg = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    block = re.search(r"- id: mixed-line-ending\n(?:\s+\S.*\n)+", cfg)
    assert block is not None, "mixed-line-ending hook missing"
    assert "args: [--fix=lf]" in block.group(0)
    assert r"exclude: ^results/robustness/summary\.csv$" in block.group(0)


def test_gitattributes_marks_the_frozen_csv_as_not_text():
    lines = (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
    assert "* text=auto eol=lf" in lines
    assert "results/robustness/summary.csv -text" in lines
    # the override must follow the catch-all rule, or the catch-all wins
    assert lines.index("results/robustness/summary.csv -text") > lines.index("* text=auto eol=lf")
