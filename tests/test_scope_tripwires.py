"""Executable versions of the standing rules in README 'Safety & scope'.

  1. No exchange, wallet, or chain SDK is imported anywhere in the code.
  2. No credential-shaped literal is committed, and no .env file is tracked.
  3. Every docs/ link in README.md resolves to a file in the repository.

tests/test_agent_invariants.py keeps the agent off the web surface; these
cover the rest of the scope statement.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest

import app

ROOT = Path(app.__file__).resolve().parent.parent
SOURCE_DIRS = ("app", "kalman", "scripts", "tests")

# Exchange clients, wallets, chain SDKs, brokers. Any of these appearing as an
# import is a scope violation, whatever the surrounding code says it is for.
BANNED_MODULES = frozenset({
    "binance", "ccxt", "web3", "solana", "solders", "eth_account", "bitcoinlib",
    "alpaca", "alpaca_trade_api", "ib_insync", "ibapi", "kucoin", "okx", "bybit",
})


def _py_files():
    for name in SOURCE_DIRS:
        yield from (ROOT / name).rglob("*.py")


def _imported_modules(source: str) -> set[str]:
    """Top-level module name of every import in the source, every alias of a
    multi-module `import a, b` included, nested function-level imports included.
    Relative imports (`from . import x`) name nothing outside the package."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def test_the_import_scan_sees_every_alias_and_every_nesting():
    assert "ccxt" in _imported_modules("import os, ccxt\n")
    assert "web3" in _imported_modules("def f():\n    from web3.auto import w3\n")
    assert "binance" in _imported_modules("import binance.client as bc\n")
    assert _imported_modules("from . import sibling\nfrom .x import y\n") == set()


def test_no_exchange_wallet_or_chain_sdk_is_imported_anywhere():
    offenders = {}
    for path in _py_files():
        banned = _imported_modules(path.read_text(encoding="utf-8")) & BANNED_MODULES
        if banned:
            offenders[str(path.relative_to(ROOT))] = sorted(banned)
    assert not offenders, f"out-of-scope SDK imported: {offenders}"


def _tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
    return [ROOT / p for p in out.stdout.decode("utf-8").split("\0") if p]


def _text_of(path: Path) -> str | None:
    """The file's text, or None when it is binary (a NUL byte in the first
    8 KiB) or not UTF-8. Every tracked file is a candidate; nothing is
    excluded by extension, so a .pem, a shell script, or a .env.production
    would be scanned like anything else."""
    with path.open("rb") as handle:
        head = handle.read(8192)
        if b"\0" in head:
            return None
        rest = handle.read()
    try:
        return (head + rest).decode("utf-8")
    except UnicodeDecodeError:
        return None


CREDENTIAL_SHAPES = (
    r"sk-ant-[A-Za-z0-9_-]{20,}",                                   # Anthropic API key
    r"AKIA[0-9A-Z]{16}",                                            # AWS access key id
    r"gh[pousr]_[A-Za-z0-9]{36}",                                   # GitHub classic tokens
    r"github_pat_[A-Za-z0-9_]{22,}",                                # GitHub fine-grained PAT
    r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY(?: BLOCK)?-----",  # PEM / OpenPGP armor
)
_CREDENTIAL = re.compile("|".join(CREDENTIAL_SHAPES))


@pytest.mark.parametrize("sample", [
    "sk-ant-" + "a" * 24,
    "AKIA" + "A" * 16,
    "ghp_" + "x" * 36,
    "github_pat_" + "y" * 30,
    "-----BEGIN " + "PGP PRIVATE KEY BLOCK" + "-----",
    "-----BEGIN " + "OPENSSH PRIVATE KEY" + "-----",
    "-----BEGIN " + "PRIVATE KEY" + "-----",
])
def test_each_credential_shape_is_recognised(sample):
    assert _CREDENTIAL.search(sample)


def test_no_credential_shaped_literal_is_committed_in_any_tracked_file():
    hits = []
    for path in _tracked_files():
        text = _text_of(path)
        if text is not None and _CREDENTIAL.search(text):
            hits.append(str(path.relative_to(ROOT)))
    assert not hits, f"credential-shaped literal committed in: {hits}"


def test_no_env_file_is_tracked_except_the_example_and_dot_env_is_ignored():
    env_like = [
        str(p.relative_to(ROOT)) for p in _tracked_files()
        if p.name == ".env" or (p.name.startswith(".env.") and p.name != ".env.example")
    ]
    assert env_like == [], f".env files must never be tracked: {env_like}"
    ignored = subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=ROOT)
    assert ignored.returncode == 0, ".env must be git-ignored"


def test_every_docs_link_in_the_readme_resolves():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    links = re.findall(r"\]\((docs/[^)#]*)", readme)
    assert links, "README no longer links into docs/?"
    missing = [link for link in links if not (ROOT / link).exists()]
    assert not missing, f"README links to missing docs: {missing}"
