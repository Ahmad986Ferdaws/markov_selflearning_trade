"""CITATION.cff is what a citing paper copies. It must carry the same version
as the code, point at the repository pyproject names, declare the licence the
repository ships under, and be dated no later than today. The five scalar
fields are read with regexes on the text: the suite has no yaml dependency
and does not need one for this.
"""

from __future__ import annotations

import datetime as dt
import re
import tomllib
from pathlib import Path

import app

ROOT = Path(app.__file__).resolve().parent.parent


def _field(name: str) -> str:
    """The value of a top-level `name:` line (unindented, so nested keys of the
    same name never match), with surrounding quotes stripped."""
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(rf"^{re.escape(name)}:[ \t]*(.+?)[ \t]*$", text, flags=re.MULTILINE)
    assert match, f"CITATION.cff has no top-level '{name}:' field"
    return match.group(1).strip("\"'")


def test_citation_version_matches_the_code():
    assert _field("version") == app.__version__


def test_citation_repository_matches_pyproject():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert _field("repository-code") == pyproject["project"]["urls"]["Repository"]


def test_citation_license_is_mit():
    assert _field("license") == "MIT"


def test_citation_release_date_is_an_iso_date_not_in_the_future():
    released = _field("date-released")
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", released), released
    assert dt.date.fromisoformat(released) <= dt.date.today()


def test_citation_format_version_is_pinned():
    assert _field("cff-version") == "1.2.0"
