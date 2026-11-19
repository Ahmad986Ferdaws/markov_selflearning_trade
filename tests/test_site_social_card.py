"""The site's social card must name an image a link preview can fetch.

design/index.html is deployed with design/ as the root, so a relative
``../media/...`` image path escapes the site and Twitter, Slack and iMessage
render the card without a picture. These pin the fix: og:image and
twitter:image are absolute https URLs, they name a file that exists in this
repository, and og:image:alt is present. Regexes only; no HTML parser.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import app

ROOT = Path(app.__file__).resolve().parent.parent
INDEX = ROOT / "design" / "index.html"
REPO_RAW = "https://raw.githubusercontent.com/Ahmad986Ferdaws/markov_selflearning_trade/main/"

IMAGE_TAGS = [("property", "og:image"), ("name", "twitter:image")]


def _meta(attr: str, name: str) -> str:
    """The content of the single ``<meta attr="name" content="...">`` tag."""
    text = INDEX.read_text(encoding="utf-8")
    found = re.findall(rf'<meta\s+{attr}="{re.escape(name)}"\s+content="([^"]*)"', text)
    assert len(found) == 1, f"expected exactly one {name} meta tag, found {len(found)}"
    return found[0]


@pytest.mark.parametrize("attr, name", IMAGE_TAGS)
def test_social_image_is_an_absolute_https_url(attr, name):
    url = _meta(attr, name)
    assert url.startswith("https://"), f"{name} must be absolute for link previews: {url!r}"
    assert "../" not in url


@pytest.mark.parametrize("attr, name", IMAGE_TAGS)
def test_social_image_names_a_file_committed_in_this_repository(attr, name):
    url = _meta(attr, name)
    assert url.startswith(REPO_RAW), f"{name} should be served from this repository: {url!r}"
    relative = url[len(REPO_RAW):]
    assert (ROOT / relative).is_file(), f"{name} points at {relative}, which is not in the repository"


def test_og_image_and_twitter_image_name_the_same_file():
    assert _meta("property", "og:image") == _meta("name", "twitter:image")


def test_og_image_alt_is_present_and_describes_the_verdict():
    alt = _meta("property", "og:image:alt")
    assert alt.strip(), "og:image:alt must not be empty"
    assert "90.9%" in alt and "zero" in alt
