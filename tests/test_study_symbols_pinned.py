"""Every symbol the two studies name must have its pinned snapshot committed,
and the committed run records must name exactly those symbols. A renamed or
dropped snapshot, or a record without a study entry, fails here loudly
instead of surfacing as a live download or a silently shorter study.
"""

from __future__ import annotations

from pathlib import Path

import app
from scripts.long_history_study import SYMBOLS as LONG_HISTORY_SYMBOLS
from scripts.robustness_study import SYMBOLS as ROBUSTNESS_SYMBOLS

ROOT = Path(app.__file__).resolve().parent.parent
SNAPSHOTS = ROOT / "data" / "snapshots"
RESULTS = ROOT / "results"


def _snapshot(symbol: str, years: int) -> Path:
    return SNAPSHOTS / f"{symbol.replace('/', '_')}_{years}y.pkl"     # load_or_fetch's naming rule


def _record_name(symbol: str) -> str:
    return symbol.replace("/", "_").replace("=", "_")                  # the studies' naming rule


def _record_stems(folder: Path) -> set[str]:
    return {p.stem for p in folder.glob("*.json") if p.name != "summary.json"}


def test_robustness_basket_is_twenty_unique_symbols_each_with_a_pinned_three_year_snapshot():
    assert len(ROBUSTNESS_SYMBOLS) == 20
    assert len(set(ROBUSTNESS_SYMBOLS)) == 20
    missing = [s for s in ROBUSTNESS_SYMBOLS if not _snapshot(s, 3).exists()]
    assert missing == [], f"no pinned 3y snapshot for {missing}"


def test_long_history_basket_is_eight_unique_pairs_each_with_a_pinned_snapshot():
    assert len(LONG_HISTORY_SYMBOLS) == 8
    assert len(set(LONG_HISTORY_SYMBOLS)) == 8
    missing = [(s, y) for s, y in LONG_HISTORY_SYMBOLS if not _snapshot(s, y).exists()]
    assert missing == [], f"no pinned snapshot for {missing}"


def test_the_robustness_records_name_exactly_the_robustness_basket():
    full_window = {stem for stem in _record_stems(RESULTS / "robustness") if ":" not in stem}
    assert full_window == {_record_name(s) for s in ROBUSTNESS_SYMBOLS}


def test_the_long_history_records_name_exactly_the_long_history_basket():
    assert _record_stems(RESULTS / "long_history") == {_record_name(s) for s, _ in LONG_HISTORY_SYMBOLS}
