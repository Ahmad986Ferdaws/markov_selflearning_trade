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
