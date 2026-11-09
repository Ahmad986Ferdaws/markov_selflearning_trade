"""pyproject.toml declares the 'Typing :: Typed' classifier, which promises
the PEP 561 marker that lets a type checker in a consuming project read this
package's annotations. Neither package shipped one, so the annotations were
ignored downstream. Pin both halves together: the markers exist and are
empty, and the classifier that makes them a promise is still declared.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import app
import kalman

ROOT = Path(app.__file__).resolve().parent.parent


def test_both_packages_ship_an_empty_py_typed_marker():
    for package in (app, kalman):
        marker = Path(package.__file__).resolve().parent / "py.typed"
        assert marker.is_file(), f"{package.__name__} has no py.typed marker"
        assert marker.read_bytes() == b"", f"{package.__name__}/py.typed should be empty (PEP 561)"


def test_pyproject_still_makes_the_typed_promise_for_both_packages():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "Typing :: Typed" in pyproject["project"]["classifiers"]
    # hatch ships every file inside these package directories, markers included
    assert set(pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]) == {"app", "kalman"}
