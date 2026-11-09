"""Every relative link in the repository's Markdown must resolve to a file.

tests/test_scope_tripwires.py::test_every_docs_link_in_the_readme_resolves
covers README -> docs/. This generalises it: README.md, CONTRIBUTING.md,
SECURITY.md, CHANGELOG.md, CODE_OF_CONDUCT.md, design/philosophy.md, every
docs/*.md, the pull-request and issue templates under .github/ and every
folder README (data/, design/, media/, results/, scripts/, tests/) are
scanned for `](target)` links; http(s), mailto and pure-anchor targets are
skipped, a trailing #fragment is stripped, and the rest must exist relative
to the file that links to it. One parametrised case per link, so a failure
names the file and the link.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import app

ROOT = Path(app.__file__).resolve().parent.parent
MARKDOWN = [
    ROOT / "README.md", ROOT / "CONTRIBUTING.md", ROOT / "SECURITY.md", ROOT / "CHANGELOG.md",
    ROOT / "CODE_OF_CONDUCT.md",
    ROOT / "design" / "philosophy.md",
    *sorted((ROOT / "docs").glob("*.md")),
    ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md",
    *sorted((ROOT / ".github" / "ISSUE_TEMPLATE").glob("*.md")),
    *sorted(p for p in ROOT.glob("*/README.md") if p.parent.name != "docs"),   # docs/ is above
]
LINK = re.compile(r"\]\(([^)\s]+)[^)]*\)")          # ](target) or ](target "title")
EXTERNAL = ("http://", "https://", "mailto:")


def _relative_links(path: Path) -> list[str]:
    targets = LINK.findall(path.read_text(encoding="utf-8"))
    return [t for t in targets if not t.startswith(EXTERNAL) and not t.startswith("#")]


CASES = [(path, link) for path in MARKDOWN for link in _relative_links(path)]


def test_the_scan_covers_every_named_file_and_finds_links():
    assert all(path.is_file() for path in MARKDOWN)
    assert len(MARKDOWN) >= 30                      # top-level files, docs/, templates, folder READMEs
    assert len(CASES) > 50


def test_the_link_filter_keeps_relative_targets_only(tmp_path):
    sample = tmp_path / "sample.md"
    sample.write_text(
        "[a](docs/01-scope.md) [b](https://x.y/z) [c](mailto:me@x.y) [d](#anchor) "
        '[e](../README.md#top) ![f](media/demo.gif) [g](16-robustness-study.md "title")',
        encoding="utf-8",
    )
    assert _relative_links(sample) == [
        "docs/01-scope.md", "../README.md#top", "media/demo.gif", "16-robustness-study.md",
    ]


@pytest.mark.parametrize(
    ("path", "link"), CASES,
    ids=[f"{path.relative_to(ROOT)} -> {link}" for path, link in CASES],
)
def test_relative_link_resolves(path: Path, link: str):
    target = link.split("#", 1)[0]
    assert (path.parent / target).exists(), f"{path.relative_to(ROOT)} links to missing {link}"
