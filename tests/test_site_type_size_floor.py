"""The site's stylesheet keeps every font-size at or above the 12 px floor.

design/philosophy.md records that the sub-12 px micro-labels (axis, scrub, tags,
chips, badge) were raised to 12 px and docs/STATUS.md claims ">=12 px micro-text".
This pins that claim against the ``<style>`` block of design/index.html with a
regex; the 11 px canvas fonts drawn inside the aria-hidden scene canvases are
decorative bitmaps in the script, not stylesheet rules, and are not covered.
"""

from __future__ import annotations

import re
from pathlib import Path

import app

ROOT = Path(app.__file__).resolve().parent.parent
INDEX = ROOT / "design" / "index.html"

FLOOR_PX = 12.0
# A sanity floor on the match count, so a stylesheet moved out of the page cannot
# pass by vacuity: today the block declares ~45 pixel font sizes.
MIN_EXPECTED_RULES = 20


def _stylesheet() -> str:
    page = INDEX.read_text(encoding="utf-8")
    blocks = re.findall(r"<style>(.*?)</style>", page, re.DOTALL)
    assert len(blocks) == 1, f"expected exactly one <style> block, found {len(blocks)}"
    return blocks[0]


def _pixel_font_sizes() -> list[float]:
    return [float(v) for v in re.findall(r"font-size:(\d+(?:\.\d+)?)px", _stylesheet())]


def test_the_stylesheet_declares_enough_pixel_font_sizes_to_be_the_real_one():
    assert len(_pixel_font_sizes()) >= MIN_EXPECTED_RULES


def test_no_stylesheet_font_size_is_below_the_twelve_px_floor():
    sizes = _pixel_font_sizes()
    assert min(sizes) >= FLOOR_PX, f"font sizes below {FLOOR_PX:g}px: {sorted(s for s in sizes if s < FLOOR_PX)}"


def test_the_three_former_eleven_px_labels_now_say_twelve():
    css = _stylesheet()
    for selector in (".arow .ak small,.arow .av small", ".rtable .pend .badge", ".dchips span"):
        rule = re.search(re.escape(selector) + r"\{([^}]*)\}", css)
        assert rule is not None, f"rule {selector!r} missing from the stylesheet"
        assert "font-size:12px" in rule.group(1), f"{selector!r} is not at the 12px floor: {rule.group(1)!r}"
