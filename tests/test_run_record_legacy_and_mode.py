"""Two things `make eval` on a fresh clone exposed after content addressing:

1. The committed receipt keeps its legacy `{symbol}_{data_hash[:8]}.json`
   name, so a byte-for-byte equal report was written AGAIN under the new
   `{symbol}_{sha256}.json` name and the tree was dirty after step one.
   A value-identical legacy record is now reused (touched, bytes untouched).
2. Records and the agent cache were written 0o600 (NamedTemporaryFile's
   default survives the rename) while everything else in the repo is 0o644.
"""

from __future__ import annotations

import json
import os
import shutil
import stat

import pytest
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from app.services.agent_policy import ResponseCache
from app.services.data_cache import atomic_write_text, load_or_fetch, save_run_record
from app.services.evaluation import evaluate

CANONICAL = Path("results/BTC-USD_4a150b23.json")


@dataclass
class Report:
    symbol: str = "SYN"
    data_hash: str = "abcdef12" * 8
    total_return: float = 0.1


def _legacy_bytes(report) -> str:
    # exactly what the pre-content-addressing writer produced (unsorted keys)
    return json.dumps(asdict(report), indent=2, default=str)


def _umask() -> int:
    u = os.umask(0)
    os.umask(u)
    return u


def test_value_identical_legacy_record_is_reused_not_duplicated(tmp_path):
    legacy = tmp_path / "SYN_abcdef12.json"
    legacy.write_text(_legacy_bytes(Report()))
    os.utime(legacy, (100, 100))
    path = save_run_record(Report(), tmp_path)
    assert path == legacy
    assert [p.name for p in tmp_path.iterdir()] == ["SYN_abcdef12.json"]
    assert legacy.read_text() == _legacy_bytes(Report())   # bytes untouched
    assert legacy.stat().st_mtime > 100                     # but now the latest record


def test_differing_legacy_json_is_preserved_and_a_new_record_written(tmp_path):
    legacy = tmp_path / "SYN_abcdef12.json"
    legacy.write_text(_legacy_bytes(replace(Report(), total_return=0.2)))
    path = save_run_record(Report(), tmp_path)
    assert path != legacy and path.parent == tmp_path
    assert json.loads(legacy.read_text())["total_return"] == 0.2
    assert json.loads(path.read_text())["total_return"] == 0.1


def test_unreadable_legacy_file_is_left_alone(tmp_path):
    legacy = tmp_path / "SYN_abcdef12.json"
    legacy.write_text("historical evidence, not JSON")
    path = save_run_record(Report(), tmp_path)
    assert path != legacy and legacy.read_text() == "historical evidence, not JSON"


def test_legacy_lookup_never_follows_a_symlink_out_of_the_results_dir(tmp_path):
    # the old naming replaced "/" with "_", so no symbol string can name a path
    # outside results/; the guard matters for a legacy entry that is a symlink
    results = tmp_path / "results"
    results.mkdir()
    outside = tmp_path / "elsewhere.json"
    outside.write_text(_legacy_bytes(Report()))           # value-identical target
    (results / "SYN_abcdef12.json").symlink_to(outside)
    before = outside.stat().st_mtime
    path = save_run_record(Report(), results)
    assert path.parent == results and not path.is_symlink()   # a new record inside
    assert outside.read_text() == _legacy_bytes(Report()) and outside.stat().st_mtime == before


@pytest.mark.parametrize("symbol", ["A" * 300, "SY\x00N", "../escape", "", "."])
def test_degenerate_symbols_still_yield_a_record(tmp_path, symbol):
    # main returned a record for these; the legacy lookup must not raise
    path = save_run_record(replace(Report(), symbol=symbol), tmp_path)
    assert path.parent == tmp_path and path.exists()


def test_legacy_reuse_is_type_strict(tmp_path):
    legacy = tmp_path / "SYN_abcdef12.json"
    legacy.write_text(json.dumps({"symbol": "SYN", "data_hash": "abcdef12" * 8, "total_return": 1}))
    path = save_run_record(replace(Report(), total_return=1.0), tmp_path)   # 1 vs 1.0
    assert path != legacy


def test_fresh_evaluate_reuses_the_committed_receipt(tmp_path):
    # the `make eval` case: regenerate the canonical report, save next to a
    # copy of the committed receipt -> that receipt is returned, nothing new
    shutil.copy(CANONICAL, tmp_path / CANONICAL.name)
    history, _ = load_or_fetch("BTC-USD", years=3)
    report = evaluate(history, symbol="BTC-USD", train_frac=0.7,
                      grid_windows=(10, 20, 30), grid_k=(0.2, 0.35, 0.5, 0.75),
                      fee_pct=0.3, slippage_pct=0.5)
    assert save_run_record(report, tmp_path).name == CANONICAL.name
    assert [p.name for p in tmp_path.iterdir()] == [CANONICAL.name]


def test_new_record_mode_follows_the_umask(tmp_path):
    path = save_run_record(Report(), tmp_path)
    assert stat.S_IMODE(path.stat().st_mode) == (0o666 & ~_umask())


def test_agent_cache_mode_follows_the_umask(tmp_path):
    cache = ResponseCache(tmp_path / "anthropic_model.json")
    cache.get_or_call("k", lambda: "v")
    cache.save()
    assert stat.S_IMODE(cache.path.stat().st_mode) == (0o666 & ~_umask())
    assert json.loads(cache.path.read_text()) == {"k": "v"}


def test_writers_never_touch_the_process_umask(tmp_path, monkeypatch):
    # an os.umask(0)/restore pair would race every other thread in the process
    def forbidden(_mask):
        raise AssertionError("os.umask must not be called by the writers")
    monkeypatch.setattr(os, "umask", forbidden)
    save_run_record(Report(), tmp_path)
    cache = ResponseCache(tmp_path / "c.json")
    cache.get_or_call("k", lambda: "v")
    cache.save()


def test_atomic_write_leaves_no_temporary_and_replaces_in_place(tmp_path):
    target = tmp_path / "out.json"
    target.write_text("old")
    atomic_write_text(target, "new")
    assert target.read_text() == "new"
    assert [p.name for p in tmp_path.iterdir()] == ["out.json"]
