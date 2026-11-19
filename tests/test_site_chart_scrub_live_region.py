"""The equity chart's keyboard scrub values are announced to screen readers.

design/index.html makes the equity chart focusable and its aria-label invites
users to read values with the arrow keys, but those values are written into the
``#scrub`` readout with innerHTML. Unless that element is a live region a
screen-reader user who presses the arrow keys hears nothing. These pin the
``aria-live="polite" aria-atomic="true"`` attributes on it with regexes only.
"""

from __future__ import annotations

import re
from pathlib import Path

import app

ROOT = Path(app.__file__).resolve().parent.parent
INDEX = ROOT / "design" / "index.html"

_SCRUB = re.compile(r'<div\b([^>]*\bid="scrub"[^>]*)>')
_ATTR = re.compile(r'([a-zA-Z-]+)\s*=\s*"([^"]*)"')


def _scrub_attrs() -> dict[str, str]:
    found = _SCRUB.findall(INDEX.read_text(encoding="utf-8"))
    assert len(found) == 1, f'expected exactly one element with id="scrub", found {len(found)}'
    return {name.lower(): value for name, value in _ATTR.findall(found[0])}


def test_the_scrub_readout_is_a_polite_live_region():
    assert _scrub_attrs().get("aria-live") == "polite"


def test_the_scrub_readout_is_announced_atomically():
    assert _scrub_attrs().get("aria-atomic") == "true"


def test_the_chart_still_invites_arrow_key_reading_and_writes_into_the_scrub():
    page = INDEX.read_text(encoding="utf-8")
    chart = re.search(r'<svg\b[^>]*\bid="equity"[^>]*>', page)
    assert chart is not None
    assert "arrow keys" in chart.group(0)
    assert "scrub.innerHTML=" in page
