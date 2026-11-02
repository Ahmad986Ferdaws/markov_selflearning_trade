"""The committed run records are the evidence. These tests read them the way
a reader would and pin what README and the docs claim about them. No engine
code runs; only JSON is parsed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import app

ROOT = Path(app.__file__).resolve().parent.parent
RESULTS = ROOT / "results"
HEADLINE = RESULTS / "BTC-USD_4a150b23.json"

REPORT_KEYS = {
    "accuracy", "chosen_k", "chosen_window", "data_hash", "policies", "regime_mix",
    "symbol", "test_size", "train_best_score", "train_size", "warnings",
}
ACCURACY_KEYS = {
    "balanced_accuracy", "hit_rate", "log_loss", "n", "n_switch", "naive_balanced",
    "naive_hit_rate", "naive_log_loss", "persistence_balanced", "persistence_hit_rate",
    "switch_attempts", "switch_recall",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_headline_record_has_the_full_report_shape():
    rec = _load(HEADLINE)
    assert set(rec) == REPORT_KEYS
    assert set(rec["accuracy"]) == ACCURACY_KEYS
    assert rec["data_hash"].startswith("4a150b23")          # the filename names the data
    assert rec["symbol"] == "BTC-USD"
    assert {p["name"] for p in rec["policies"]} == {"baseline", "buy_hold"}
    assert (rec["chosen_window"], rec["chosen_k"]) == (30, 0.2)


def test_headline_record_is_the_receipt_the_readme_quotes():
    rec = _load(HEADLINE)
    acc = rec["accuracy"]
    assert acc["n"] == 328
    assert acc["hit_rate"] == acc["persistence_hit_rate"]       # the identity, exactly
    assert acc["hit_rate"] == pytest.approx(298 / 328)
    assert acc["naive_hit_rate"] == pytest.approx(0.720, abs=1e-3)
    assert (acc["n_switch"], acc["switch_attempts"], acc["switch_recall"]) == (30, 0, 0.0)
    assert any("persistence" in w for w in rec["warnings"])    # the gate fired and was kept
