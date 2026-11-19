"""scripts/independent_rederivation.py reports a missing snapshot as exit 2.

The module docstring defines exit 1 as MISMATCH (engine or rule bug). An
uncaught FileNotFoundError from pd.read_pickle also exits 1, so CI would have
reported a deleted pickle as a rule bug. The missing-file case now prints
SNAPSHOT MISSING and returns 2 alongside SNAPSHOT CHANGED, the other
'wrong data on disk' verdict. tests/test_independent_rederivation_script.py
pins the exit 0 / exit 2 (tampered) / --help paths as a subprocess; this
file calls main() in-process on a path that does not exist.
"""

from __future__ import annotations

import pickle

import pytest

import scripts.independent_rederivation as rederivation


def test_missing_snapshot_returns_two_and_names_the_path(tmp_path, capsys):
    absent = tmp_path / "absent.pkl"
    assert not absent.exists()
    rc = rederivation.main(["--snapshot", str(absent)])
    out = capsys.readouterr().out
    assert rc == 2
    assert "SNAPSHOT MISSING" in out
    assert str(absent) in out
    assert "re-pin" in out
    assert "REPRODUCED" not in out and "MISMATCH" not in out and "SNAPSHOT CHANGED" not in out


def test_missing_snapshot_is_reported_before_any_hash_or_scoring_line(tmp_path, capsys):
    rc = rederivation.main(["--snapshot", str(tmp_path / "gone.pkl")])
    out = capsys.readouterr().out
    assert rc == 2
    assert "hash=" not in out and "predictions=" not in out
    assert out.strip().startswith("SNAPSHOT MISSING")


def test_only_a_missing_file_is_caught_other_read_errors_still_propagate(tmp_path):
    not_a_pickle = tmp_path / "garbage.pkl"
    not_a_pickle.write_bytes(b"this is not a pickle")
    with pytest.raises(pickle.UnpicklingError):
        rederivation.main(["--snapshot", str(not_a_pickle)])


def test_docstring_and_help_name_the_missing_case_consistently():
    assert "2  SNAPSHOT CHANGED or missing" in rederivation.__doc__
    # the argparse epilog is pinned verbatim by the subprocess test and stays as is
    parser_help = rederivation._parse_args.__code__.co_consts
    assert "exit codes: 0 reproduced, 1 mismatch, 2 snapshot changed" in parser_help
