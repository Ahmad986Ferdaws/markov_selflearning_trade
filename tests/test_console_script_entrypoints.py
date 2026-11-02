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


def test_the_registered_clis_are_exactly_the_five_the_readme_documents():
    assert set(SCRIPTS) == {"regime-cli", "compare-cli", "evaluate-cli", "diagnose-cli", "kalman-cli"}


def test_every_cli_module_has_a_main_with_no_required_arguments():
    """Each launcher calls main() with nothing; a main that grew a required
    positional parameter would crash at the shell, not in a test."""
    import inspect

    for name, target in SCRIPTS.items():
        module_name, _, attr = target.partition(":")
        fn = getattr(importlib.import_module(module_name), attr)
        required = [
            p.name for p in inspect.signature(fn).parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        assert not required, f"{name}: main() requires {required}; the launcher passes nothing"
