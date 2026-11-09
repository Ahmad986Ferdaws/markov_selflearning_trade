"""The site stays readable with JavaScript disabled.

design/index.html hides the verdict label, lede, anatomy, chart card, results table,
pull-quote and disclosure cards behind ``.rv{opacity:0}`` until a script reveals
them, and the ``nogsap`` class that undoes it is itself added by JavaScript. A
``<noscript><style>`` block in the <head> mirrors the ``.nogsap`` rules so that,
without scripts, every section, the filled bars and all three method steps are
visible. These pin that block's presence, placement and content with regexes only.
"""

from __future__ import annotations

import re
from pathlib import Path

import app

ROOT = Path(app.__file__).resolve().parent.parent
INDEX = ROOT / "design" / "index.html"

_NOSCRIPT = re.compile(r"<noscript>(.*?)</noscript>", re.DOTALL)


def _page() -> str:
    return INDEX.read_text(encoding="utf-8")


def _noscript() -> str:
    blocks = _NOSCRIPT.findall(_page())
    assert len(blocks) == 1, f"expected exactly one <noscript> block, found {len(blocks)}"
    return blocks[0]


def test_the_page_has_exactly_one_noscript_block_and_it_sits_in_the_head():
    page = _page()
    assert page.count("<noscript") == 1
    assert page.find("<noscript") < page.find("</head>")


def test_the_noscript_block_is_a_stylesheet_and_carries_no_script():
    block = _noscript()
    assert "<style>" in block and "</style>" in block
    assert "<script" not in block.lower()


def test_the_noscript_stylesheet_reveals_the_hidden_sections_steps_and_lanes():
    css = _noscript()
    for selector in (".rv", ".steps .step", ".lane.m .fill"):
        assert selector in css, f"the no-JS stylesheet does not mention {selector!r}"


def test_the_noscript_stylesheet_targets_the_root_not_the_script_added_class():
    """The rules must apply with no script at all, so they hang off ``html``, not ``.nogsap``."""
    rules = re.sub(r"/\*.*?\*/", "", _noscript(), flags=re.DOTALL)  # selectors only, not the comment
    assert ".nogsap" not in rules
    assert "html .rv" in rules
