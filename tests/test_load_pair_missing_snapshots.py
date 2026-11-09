"""`load_pair_from_snapshots` names every missing pinned pickle at once, so a
user with neither file fixes both after one run instead of hitting the second
error on the re-run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalman.data import PairData, load_pair_from_snapshots


def _leg(n: int = 260, base: float = 100.0) -> pd.DataFrame:
    closes = base + np.arange(n, dtype=float) * 0.1
    return pd.DataFrame({"Open": closes * 1.001, "Close": closes},
                        index=pd.bdate_range("2020-01-01", periods=n))


def test_both_missing_names_both_files(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"pinned snapshot\(s\) missing") as info:
        load_pair_from_snapshots("AAA", "BBB", years=1, snap_dir=tmp_path)
    msg = str(info.value)
    assert str(tmp_path / "AAA_1y.pkl") in msg
    assert str(tmp_path / "BBB_1y.pkl") in msg
    assert msg.index("AAA_1y.pkl") < msg.index("BBB_1y.pkl")   # listed in argument order


def test_only_the_second_missing_names_only_it(tmp_path):
    _leg().to_pickle(tmp_path / "AAA_1y.pkl")
    with pytest.raises(FileNotFoundError, match="missing") as info:
        load_pair_from_snapshots("AAA", "BBB", years=1, snap_dir=tmp_path)
    msg = str(info.value)
    assert str(tmp_path / "BBB_1y.pkl") in msg
    assert "AAA_1y.pkl" not in msg


def test_only_the_first_missing_names_only_it(tmp_path):
    _leg().to_pickle(tmp_path / "BBB_1y.pkl")
    with pytest.raises(FileNotFoundError, match="missing") as info:
        load_pair_from_snapshots("AAA", "BBB", years=1, snap_dir=tmp_path)
    msg = str(info.value)
    assert str(tmp_path / "AAA_1y.pkl") in msg
    assert "BBB_1y.pkl" not in msg


def test_both_present_returns_a_pair(tmp_path):
    _leg(base=100.0).to_pickle(tmp_path / "AAA_1y.pkl")
    _leg(base=50.0).to_pickle(tmp_path / "BBB_1y.pkl")
    pair = load_pair_from_snapshots("AAA", "BBB", years=1, snap_dir=str(tmp_path))
    assert isinstance(pair, PairData)
    assert (pair.name1, pair.name2) == ("AAA", "BBB")
    assert len(pair) == 260                            # >= the default 250-bar overlap
    assert pair.p2["Close"].iloc[0] == pytest.approx(50.0)


def test_overlap_boundary_still_enforced_once_both_files_exist(tmp_path):
    _leg(n=249).to_pickle(tmp_path / "AAA_1y.pkl")
    _leg(n=249).to_pickle(tmp_path / "BBB_1y.pkl")
    with pytest.raises(ValueError, match="overlap too short"):
        load_pair_from_snapshots("AAA", "BBB", years=1, snap_dir=tmp_path)
