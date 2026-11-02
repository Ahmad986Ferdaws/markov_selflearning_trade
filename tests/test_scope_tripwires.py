"""Executable versions of the standing rules in README 'Safety & scope'.

  1. No exchange, wallet, or chain SDK is imported anywhere in the code.
  2. No credential-shaped literal is committed, and .env itself is ignored.
  3. Every docs/ link in README.md resolves to a file in the repository.

tests/test_agent_invariants.py keeps the agent off the web surface; these
cover the rest of the scope statement.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import app

ROOT = Path(app.__file__).resolve().parent.parent
SOURCE_DIRS = ("app", "kalman", "scripts", "tests")

# Exchange clients, wallets, chain SDKs, brokers. Any of these appearing as an
# import is a scope violation, whatever the surrounding code says it is for.
BANNED_MODULES = (
    "binance", "ccxt", "web3", "solana", "solders", "eth_account", "bitcoinlib",
    "alpaca", "alpaca_trade_api", "ib_insync", "ibapi", "kucoin", "okx", "bybit",
)


def _py_files():
    for name in SOURCE_DIRS:
        yield from (ROOT / name).rglob("*.py")


def test_no_exchange_wallet_or_chain_sdk_is_imported_anywhere():
    pattern = re.compile(
        r"^\s*(?:import|from)\s+(?:" + "|".join(BANNED_MODULES) + r")\b", re.MULTILINE
    )
    offenders = [
        str(f.relative_to(ROOT)) for f in _py_files()
        if pattern.search(f.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"out-of-scope SDK imported in: {offenders}"


def _tracked_text_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
    suffixes = (".py", ".md", ".toml", ".yml", ".yaml", ".json", ".csv", ".html", ".js",
                ".mjs", ".cff", ".txt", ".example", ".lock")
    return [ROOT / p for p in out.stdout.split() if p.endswith(suffixes)]


CREDENTIAL_SHAPES = (
    r"sk-ant-[A-Za-z0-9_-]{20,}",                 # Anthropic API key
    r"AKIA[0-9A-Z]{16}",                          # AWS access key id
    r"gh[pousr]_[A-Za-z0-9]{36}",                 # GitHub token
    r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----",
)


def test_no_credential_shaped_literal_is_committed():
    pattern = re.compile("|".join(CREDENTIAL_SHAPES))
    hits = [
        str(f.relative_to(ROOT)) for f in _tracked_text_files()
        if pattern.search(f.read_text(encoding="utf-8", errors="replace"))
    ]
    assert not hits, f"credential-shaped literal committed in: {hits}"


def test_dot_env_is_ignored_and_untracked():
    ignored = subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=ROOT)
    assert ignored.returncode == 0, ".env must be git-ignored"
    tracked = subprocess.run(["git", "ls-files", ".env"], cwd=ROOT, capture_output=True, text=True)
    assert tracked.stdout.strip() == "", ".env must never be tracked"


def test_every_docs_link_in_the_readme_resolves():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    links = re.findall(r"\]\((docs/[^)#]*)", readme)
    assert links, "README no longer links into docs/?"
    missing = [link for link in links if not (ROOT / link).exists()]
    assert not missing, f"README links to missing docs: {missing}"
