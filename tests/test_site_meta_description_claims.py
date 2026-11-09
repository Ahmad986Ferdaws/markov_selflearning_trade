"""The site's meta description keeps the two headline claims apart.

The exactly-zero edge is the prediction identity (hit_rate == persistence_hit_rate
on 20/20 assets), computed before any cost enters; what the engine reports net of
costs is a loss (-41.1% against -47.7% on the headline window), not zero, and
README.md keeps the two claims apart. design/index.html's description must not
fuse them into "net of costs, the edge is exactly zero". Regexes only.
"""

from __future__ import annotations

import re
from pathlib import Path

import app

ROOT = Path(app.__file__).resolve().parent.parent
INDEX = ROOT / "design" / "index.html"

MAX_LENGTH = 170   # search snippets truncate around 155-160 characters


def _description() -> str:
    found = re.findall(r'<meta\s+name="description"\s+content="([^"]*)"', INDEX.read_text(encoding="utf-8"))
    assert len(found) == 1, f"expected exactly one description meta tag, found {len(found)}"
    return found[0]


def test_no_sentence_calls_the_zero_edge_a_net_of_costs_result():
    for sentence in _description().split(". "):
        assert not ("net of costs" in sentence and "exactly zero" in sentence), (
            f"this sentence fuses the two claims: {sentence!r}"
        )


def test_the_description_names_the_baseline_the_zero_edge_is_measured_against():
    assert "persistence" in _description()


def test_the_description_fits_a_search_snippet():
    assert len(_description()) <= MAX_LENGTH


def test_the_description_still_states_the_zero_edge_verdict():
    assert "exactly zero" in _description()
