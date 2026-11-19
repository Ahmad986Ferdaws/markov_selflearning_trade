"""One version, spelled in three places, must agree.

app.__version__ is the code's copy; pyproject.toml is what pip installs; the
newest CHANGELOG heading is what the reader sees. The web app's OpenAPI
document must advertise the same string (it used to say 0.1.0 at 0.4.1).
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import app

ROOT = Path(app.__file__).resolve().parent.parent


def test_pyproject_version_matches_the_code():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == app.__version__


def test_newest_changelog_entry_matches_the_code():
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    headings = re.findall(r"^## \[(\d+\.\d+\.\d+)\]", text, flags=re.MULTILINE)
    assert headings, "CHANGELOG.md has no '## [x.y.z]' heading"
    assert headings[0] == app.__version__


def test_openapi_document_advertises_the_same_version():
    from fastapi.testclient import TestClient

    from app.main import app as web_app

    info = TestClient(web_app).get("/openapi.json").json()["info"]
    assert info["version"] == app.__version__
