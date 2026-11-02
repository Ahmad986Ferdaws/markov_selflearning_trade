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


# --- robustness study (docs/16) ----------------------------------------------------
ROBUSTNESS = RESULTS / "robustness"


def _asset_records(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*.json") if p.name != "summary.json")


def test_every_robustness_record_is_a_full_report_tied_to_persistence():
    records = _asset_records(ROBUSTNESS)
    assert len(records) == 20
    for path in records:
        rec = _load(path)
        assert set(rec) == REPORT_KEYS, path.name
        acc = rec["accuracy"]
        assert acc["hit_rate"] == acc["persistence_hit_rate"], path.name
        assert acc["balanced_accuracy"] == acc["persistence_balanced"], path.name


def test_robustness_summary_agrees_with_its_records_and_with_docs_16():
    summary = _load(ROBUSTNESS / "summary.json")
    agg = summary["aggregate"]
    assert agg["n_assets"] == 20 == len(summary["assets"])
    assert agg["n_failed"] == 0
    assert agg["assets_model_beats_persistence"] == 0
    assert agg["mean_edge_vs_persist_bal"] == 0.0 == agg["max_edge_vs_persist_bal"]
    for asset in summary["assets"]:
        assert asset["edge_vs_persist_hit"] == 0.0, asset["symbol"]
        assert asset["edge_vs_persist_bal"] == 0.0, asset["symbol"]
        assert not asset["beats_persistence"], asset["symbol"]
    # docs/16: the only non-zero edge anywhere is early-BTC, and it is negative
    early = next(t for t in summary["temporal"] if t["symbol"] == "BTC-USD:early")
    assert early["edge_vs_persist_bal"] < 0
    assert all(t["edge_vs_persist_bal"] <= 0 for t in summary["temporal"])


# --- long-history study (docs/18) ----------------------------------------------------
LONG_HISTORY = RESULTS / "long_history"


def test_long_history_records_hold_the_identity_over_twenty_years():
    records = _asset_records(LONG_HISTORY)
    assert len(records) == 8
    for path in records:
        rec = _load(path)
        assert REPORT_KEYS <= set(rec), path.name
        acc = rec["accuracy"]
        assert acc["hit_rate"] == acc["persistence_hit_rate"], path.name
        assert acc["switch_attempts"] == 0, path.name


def test_long_history_summary_reports_zero_edge_on_every_asset():
    summary = _load(LONG_HISTORY / "summary.json")
    assets = summary["assets"]
    assert len(assets) == 8
    edge_keys = [k for k in assets[0] if k.startswith("edge_vs_persist")]
    assert edge_keys, "summary assets carry no edge_vs_persist_* fields"
    for asset in assets:
        for key in edge_keys:
            assert asset[key] == 0.0, (asset.get("symbol"), key)
