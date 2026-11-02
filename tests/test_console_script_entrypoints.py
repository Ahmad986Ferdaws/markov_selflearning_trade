"""Every console script declared in pyproject.toml must resolve to an
importable callable, and the set must be exactly the five CLIs the README
documents. A typo in [project.scripts] otherwise only surfaces after an
install, as a broken launcher."""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest

import app

ROOT = Path(app.__file__).resolve().parent.parent
SCRIPTS: dict[str, str] = tomllib.loads(
    (ROOT / "pyproject.toml").read_text(encoding="utf-8")
)["project"]["scripts"]


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_console_script_target_is_an_importable_callable(name):
    module_name, sep, attr = SCRIPTS[name].partition(":")
    assert sep == ":", f"{name}: entry point must be 'module:callable'"
    module = importlib.import_module(module_name)
    assert callable(getattr(module, attr)), f"{name} -> {SCRIPTS[name]} is not callable"
