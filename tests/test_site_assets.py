"""Tripwires for the static site under design/: what the page loads, what the headers allow.

design/index.html is one file that pulls four scripts from CDNs; the headers in
design/vercel.json say which hosts a browser may load from. These pin, with
regexes and no HTML parser:

  1. every remote <script src="https://..."> carries a sha384 integrity hash and
     crossorigin="anonymous" (SECURITY.md's out-of-scope note relies on this);
  2. every CDN host the page uses for a script, a stylesheet or a preconnect is
     allowed by the CSP's script-src / style-src / font-src;
  3. every local href/src the page references exists under design/;
  4. design/robots.txt allows '/';
  5. the page declares lang="en", a viewport meta and a <title>.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlsplit

import app

ROOT = Path(app.__file__).resolve().parent.parent
DESIGN = ROOT / "design"
INDEX = DESIGN / "index.html"

_TAG = re.compile(r"<(script|link|meta)\b([^>]*)>", re.IGNORECASE)
_ATTR = re.compile(r'([a-zA-Z-]+)\s*=\s*"([^"]*)"')
_REMOTE = re.compile(r"^(https?:)?//|^#|^data:|^mailto:|^javascript:", re.IGNORECASE)


def _page() -> str:
    return INDEX.read_text(encoding="utf-8")


def _tags(kind: str) -> list[dict[str, str]]:
    """Attribute dicts (lower-cased names) of every <kind ...> tag in the page."""
    return [
        {name.lower(): value for name, value in _ATTR.findall(attrs)}
        for tag, attrs in _TAG.findall(_page())
        if tag.lower() == kind
    ]


def _remote_scripts() -> list[dict[str, str]]:
    return [t for t in _tags("script") if t.get("src", "").startswith("https://")]


def _csp() -> dict[str, list[str]]:
    """The site-wide Content-Security-Policy from vercel.json as {directive: [sources]}."""
    config = json.loads((DESIGN / "vercel.json").read_text(encoding="utf-8"))
    headers = [h for rule in config["headers"] if rule["source"] == "/(.*)" for h in rule["headers"]]
    value = next(h["value"] for h in headers if h["key"] == "Content-Security-Policy")
    policy: dict[str, list[str]] = {}
    for directive in value.split(";"):
        parts = directive.split()
        if parts:
            policy[parts[0]] = parts[1:]
    return policy


def _host(url: str) -> str:
    return urlsplit(url).netloc


# --- 1. Subresource integrity --------------------------------------------------------
def test_the_page_loads_exactly_four_remote_scripts():
    assert len(_remote_scripts()) == 4


def test_every_remote_script_is_pinned_by_sha384_and_loaded_anonymously():
    for tag in _remote_scripts():
        src = tag["src"]
        assert re.fullmatch(r"sha384-[A-Za-z0-9+/]{64}", tag.get("integrity", "")), f"{src}: no sha384 pin"
        assert tag.get("crossorigin") == "anonymous", f"{src}: SRI needs crossorigin=anonymous"


def test_no_script_is_loaded_over_plain_http():
    assert not [t["src"] for t in _tags("script") if t.get("src", "").startswith("http://")]


# --- 2. CSP hosts ----------------------------------------------------------------------
def test_every_script_host_is_allowed_by_script_src():
    allowed = {_host(s) for s in _csp()["script-src"]}
    for tag in _remote_scripts():
        assert _host(tag["src"]) in allowed, f"{tag['src']} is not allowed by script-src"


def test_every_stylesheet_host_is_allowed_by_style_src():
    allowed = {_host(s) for s in _csp()["style-src"]}
    sheets = [t for t in _tags("link")
              if t.get("rel") == "stylesheet" and t.get("href", "").startswith("https://")]
    assert sheets, "the page no longer loads a remote stylesheet?"
    for tag in sheets:
        assert _host(tag["href"]) in allowed, f"{tag['href']} is not allowed by style-src"


def test_every_preconnect_host_is_allowed_for_styles_or_fonts():
    policy = _csp()
    allowed = {_host(s) for s in policy["style-src"] + policy["font-src"]}
    preconnects = [t for t in _tags("link") if t.get("rel") == "preconnect"]
    assert preconnects, "the page no longer preconnects to its font hosts?"
    for tag in preconnects:
        assert _host(tag["href"]) in allowed, f"preconnect to {tag['href']}, which no directive allows"


# --- 3. Local assets ---------------------------------------------------------------------
def test_every_local_href_or_src_exists_under_design():
    refs = {value for _, value in re.findall(r'\b(href|src)="([^"]*)"', _page())}
    local = sorted(v for v in refs if not _REMOTE.match(v))
    assert local, "the page references no local asset at all?"
    missing = [v for v in local if not (DESIGN / v.split("?")[0].split("#")[0]).is_file()]
    assert not missing, f"referenced but not under design/: {missing}"


def test_the_favicon_is_the_local_svg():
    icons = [t for t in _tags("link") if t.get("rel") == "icon"]
    assert len(icons) == 1
    assert icons[0]["href"] == "favicon.svg"
    assert (DESIGN / "favicon.svg").is_file()


# --- 4. robots.txt -----------------------------------------------------------------------
def test_robots_txt_allows_the_whole_site():
    text = (DESIGN / "robots.txt").read_text(encoding="utf-8")
    assert re.search(r"^User-agent:\s*\*\s*$", text, re.MULTILINE)
    assert re.search(r"^Allow:\s*/\s*$", text, re.MULTILINE)
    assert not re.search(r"^Disallow:\s*\S", text, re.MULTILINE)


# --- 5. Document basics ------------------------------------------------------------------
def test_the_page_declares_english_a_viewport_and_a_title():
    page = _page()
    assert re.search(r'<html\b[^>]*\blang="en"', page)
    viewports = [t for t in _tags("meta") if t.get("name") == "viewport"]
    assert len(viewports) == 1 and "width=device-width" in viewports[0]["content"]
    title = re.search(r"<title>([^<]+)</title>", page)
    assert title is not None and title.group(1).strip()
