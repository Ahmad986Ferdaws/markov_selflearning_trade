"""The site's pull-quote quotes warnings[0] of the headline record verbatim.

design/index.html cites its pull-quote as ``warnings[0] · results/BTC-USD_4a150b23.json``.
A quotation attributed to a record must be the record's own words, so the text
between the curly quotes (tags stripped) has to be a substring of that warning.
Regexes only; no HTML parser.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import app

ROOT = Path(app.__file__).resolve().parent.parent
INDEX = ROOT / "design" / "index.html"
RECORD = ROOT / "results" / "BTC-USD_4a150b23.json"

_QUOTE = re.compile(r'<blockquote class="quote rv">\s*<p>(.*?)</p>', re.DOTALL)
_CITE = re.compile(r'<blockquote class="quote rv">.*?<cite>(.*?)</cite>', re.DOTALL)


def _warning() -> str:
    return json.loads(RECORD.read_text(encoding="utf-8"))["warnings"][0]


def _quoted_text() -> str:
    found = _QUOTE.findall(INDEX.read_text(encoding="utf-8"))
    assert len(found) == 1, f"expected exactly one pull-quote paragraph, found {len(found)}"
    text = re.sub(r"<[^>]+>", "", found[0]).strip()
    assert text.startswith("“") and text.endswith("”"), f"not wrapped in curly quotes: {text!r}"
    return text[1:-1]


def test_the_pull_quote_is_a_verbatim_substring_of_warnings_zero():
    quoted, warning = _quoted_text(), _warning()
    assert quoted in warning, f"the page quotes {quoted!r}, which is not in warnings[0]: {warning!r}"


def test_the_pull_quote_keeps_the_persistence_clause_the_record_ends_on():
    assert _quoted_text().endswith("high accuracy here is mostly persistence, not skill.")


def test_the_citation_still_names_warnings_zero_of_the_headline_record():
    cite = _CITE.findall(INDEX.read_text(encoding="utf-8"))
    assert len(cite) == 1
    assert "warnings[0]" in cite[0] and "results/BTC-USD_4a150b23.json" in cite[0]
