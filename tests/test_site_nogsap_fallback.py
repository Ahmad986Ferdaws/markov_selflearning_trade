"""The site's no-GSAP fallback shows the real lane values and every method step.

When the CDN scripts do not load, design/index.html adds the ``nogsap`` class and
returns before any animation runs. The ``.nogsap`` CSS fills the race bars, so the
markup itself must carry the final lane values (the animated branch resets them to
0.0% in script before the count-up) and the stylesheet must lift the 30% opacity the
method steps otherwise keep until a ScrollTrigger activates them. Regexes only.
"""

from __future__ import annotations

import re
from pathlib import Path

import app

ROOT = Path(app.__file__).resolve().parent.parent
INDEX = ROOT / "design" / "index.html"

_LANE_VALUE = re.compile(r'<div class="val" data-to="([^"]+)" data-suf="%">([^<]*)</div>')
_RESET = "$$('.val',raceEl).forEach(v=>v.textContent=(0).toFixed(1)+v.dataset.suf);"
_TRIGGER = "ScrollTrigger.create({trigger:raceEl"


def _page() -> str:
    return INDEX.read_text(encoding="utf-8")


def _stylesheet() -> str:
    """The main stylesheet, ignoring any <noscript> fallback sheet the head may carry."""
    page = re.sub(r"<noscript>.*?</noscript>", "", _page(), flags=re.DOTALL)
    blocks = re.findall(r"<style>(.*?)</style>", page, re.DOTALL)
    assert len(blocks) == 1, f"expected exactly one <style> block, found {len(blocks)}"
    return blocks[0]


def test_the_page_has_the_three_race_lane_values():
    assert len(_LANE_VALUE.findall(_page())) == 3


def test_every_lane_value_in_the_markup_equals_its_data_to_target():
    for target, text in _LANE_VALUE.findall(_page()):
        assert text == f"{float(target):.1f}%", f"lane data-to={target!r} renders {text!r} without scripts"


def test_the_lane_values_are_not_the_old_zero_placeholders():
    assert "0.0%" not in [text for _, text in _LANE_VALUE.findall(_page())]


def test_the_nogsap_stylesheet_lifts_the_method_step_opacity():
    assert ".nogsap .steps .step{opacity:1" in _stylesheet()


def test_the_animated_branch_still_resets_the_lane_values_before_the_count_up():
    """The count-up must keep starting from 0.0%: the reset moved from markup to script."""
    page = _page()
    reset, trigger = page.find(_RESET), page.find(_TRIGGER)
    assert reset != -1, "the animated branch no longer resets the lane values to 0.0%"
    assert trigger != -1
    assert reset < trigger, "the reset must run before the race ScrollTrigger is created"
