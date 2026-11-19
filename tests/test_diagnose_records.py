"""diagnose-cli must refuse anything that is not a run record with held-out
predictions. It used to default every missing field to 0.0 and print
"IDENTICAL to persistence" (exit 0) over an empty dict, a study summary, or
the {"summary","report"} envelope every committed long-history file uses —
a fabricated verdict from the tool whose only job is honest narration.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.cli import diagnose
from app.cli.diagnose import NotARunRecord, as_run_record, format_diagnosis, main

BTC = Path("results/BTC-USD_4a150b23.json")


def _record(hit, per, bal, pbal, n=100, **extra):
    rec = {"symbol": "X", "data_hash": "abc", "policies": [], "warnings": [],
           "accuracy": {"hit_rate": hit, "persistence_hit_rate": per,
                        "balanced_accuracy": bal, "persistence_balanced": pbal, "n": n}}
    rec.update(extra)
    return rec


# --- refusals -----------------------------------------------------------------
@pytest.mark.parametrize("bad", [
    {}, [], "text", None, 3,
    {"aggregate": {"edge": 0.0}},                        # a study summary
    {"symbol": "X", "accuracy": "not a dict"},
    {"symbol": "X", "accuracy": {"hit_rate": 0.9}},      # edge fields missing
])
def test_non_records_are_refused(bad):
    with pytest.raises(NotARunRecord):
        format_diagnosis(bad)
    assert issubclass(NotARunRecord, ValueError)


def test_zero_predictions_is_refused_not_narrated():
    with pytest.raises(NotARunRecord, match="no held-out predictions"):
        format_diagnosis(_record(0.0, 0.0, 0.0, 0.0, n=0))


def test_non_integer_n_is_refused():
    with pytest.raises(NotARunRecord, match="integer count"):
        format_diagnosis(_record(0.9, 0.9, 0.8, 0.8, n="many"))


# --- accepted shapes ------------------------------------------------------------
def test_long_history_envelope_is_unwrapped():
    env = json.loads(Path("results/long_history/SPY.json").read_text())
    assert "accuracy" not in env and "report" in env       # the committed shape
    out = format_diagnosis(env)
    assert "Diagnosis: SPY" in out and "n=1508 held-out days" in out
    assert "n=0" not in out


def test_robustness_record_and_canonical_record_still_narrate():
    rob = json.loads(Path("results/robustness/BTC-USD.json").read_text())
    assert "held-out days" in format_diagnosis(rob)
    out = format_diagnosis(json.loads(BTC.read_text()))
    assert "IDENTICAL to persistence" in out and "90.9%" in out


def test_as_run_record_returns_the_inner_report():
    inner = _record(0.9, 0.9, 0.8, 0.8)
    assert as_run_record({"summary": {}, "report": inner}) is inner


# --- the negative-edge branch ---------------------------------------------------
def test_large_negative_edge_is_called_worse_not_noise():
    out = format_diagnosis(_record(0.40, 0.90, 0.40, 0.90))
    assert "WORSE than persistence" in out
    assert "within noise" not in out


def test_small_edges_remain_within_noise():
    out = format_diagnosis(_record(0.90, 0.91, 0.80, 0.81))
    assert "within noise" in out


# --- CLI behaviour ----------------------------------------------------------------
def test_main_corrupt_json_exits_1_with_one_line(tmp_path, capsys):
    bad = tmp_path / "corrupt.json"
    bad.write_text('{"symbol": "X", ')
    with pytest.raises(SystemExit) as exc:
        main([str(bad)])
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("diagnose-cli: ") and "not valid JSON" in err
    assert "Traceback" not in err


def test_main_json_array_exits_1(tmp_path, capsys):
    arr = tmp_path / "array.json"
    arr.write_text("[1, 2, 3]")
    with pytest.raises(SystemExit) as exc:
        main([str(arr)])
    assert exc.value.code == 1
    assert "not an evaluate-cli run record" in capsys.readouterr().err


def test_main_missing_path_exits_1(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main([str(tmp_path / "nope.json")])
    assert exc.value.code == 1
    assert "no run record found" in capsys.readouterr().err


def test_main_help_exits_0_with_usage(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "usage: diagnose-cli" in capsys.readouterr().out


def test_main_narrates_a_valid_record(capsys):
    main([str(BTC)])
    out = capsys.readouterr().out
    assert out.startswith(f"[record] {BTC}") and "IDENTICAL to persistence" in out


def test_main_default_picks_newest_record(tmp_path, monkeypatch, capsys):
    import os
    import time

    old, new = tmp_path / "a.json", tmp_path / "b.json"
    old.write_text(json.dumps(_record(0.9, 0.9, 0.8, 0.8, symbol="OLD")))
    new.write_text(json.dumps(_record(0.9, 0.9, 0.8, 0.8, symbol="NEW")))
    now = time.time()
    os.utime(old, (now - 100, now - 100))
    os.utime(new, (now, now))
    monkeypatch.setattr(diagnose, "RESULTS_DIR", tmp_path)
    main([])
    assert "Diagnosis: NEW" in capsys.readouterr().out


# --- values, not just keys (Codex review on #26) -----------------------------------
@pytest.mark.parametrize("value", [None, "0.9", True, float("nan"), float("inf"), [0.9]])
def test_non_numeric_accuracy_values_are_refused(value):
    rec = _record(0.9, 0.9, 0.8, 0.8)
    rec["accuracy"]["hit_rate"] = value
    with pytest.raises(NotARunRecord, match="finite number"):
        format_diagnosis(rec)


@pytest.mark.parametrize("n", [1.9, 328.0, "328", True, None])
def test_non_integral_prediction_counts_are_refused(n):
    with pytest.raises(NotARunRecord, match="integer count"):
        format_diagnosis(_record(0.9, 0.9, 0.8, 0.8, n=n))


@pytest.mark.parametrize("bad", [
    {"policies": [None]}, {"policies": "baseline"},
    {"policies": [{"name": "b", "total_return": None}]},
    {"warnings": "gate fired"}, {"warnings": [None]},
])
def test_malformed_policies_or_warnings_are_refused(bad):
    with pytest.raises(NotARunRecord):
        format_diagnosis(_record(0.9, 0.9, 0.8, 0.8, **bad))


def test_main_null_metric_exits_1_without_traceback(tmp_path, capsys):
    rec = _record(0.9, 0.9, 0.8, 0.8)
    rec["accuracy"]["persistence_hit_rate"] = None
    path = tmp_path / "null.json"
    path.write_text(json.dumps(rec))
    with pytest.raises(SystemExit) as exc:
        main([str(path)])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "finite number" in captured.err and "Traceback" not in captured.err
    assert "[record]" not in captured.out       # refused before any narration


# --- inputs that still escaped as tracebacks (independent review of #26) -------
@pytest.mark.parametrize("bad", [
    {"policies": [{"name": None, "total_return": 0.1}]},
    {"policies": [{"name": ["b"], "total_return": 0.1}]},
    {"symbol": None}, {"symbol": {"x": 1}},
])
def test_non_string_names_are_refused(bad):
    with pytest.raises(NotARunRecord, match="must be a string"):
        format_diagnosis(_record(0.9, 0.9, 0.8, 0.8, **bad))


def test_integer_too_large_for_a_float_is_refused():
    rec = _record(0.9, 0.9, 0.8, 0.8)
    rec["accuracy"]["hit_rate"] = 10 ** 400          # math.isfinite would overflow
    with pytest.raises(NotARunRecord, match="finite number"):
        format_diagnosis(rec)
    rec = _record(0.9, 0.9, 0.8, 0.8, policies=[{"name": "b", "total_return": 10 ** 400}])
    with pytest.raises(NotARunRecord, match="finite number"):
        format_diagnosis(rec)


def test_main_non_utf8_file_exits_1_with_one_line(tmp_path, capsys):
    bad = tmp_path / "binary.json"
    bad.write_bytes(b"\xff\xfe\x00garbage")
    with pytest.raises(SystemExit) as exc:
        main([str(bad)])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert captured.err == f"diagnose-cli: {bad}: not a UTF-8 text file\n"
    assert captured.out == ""


def test_main_refusal_names_the_file(tmp_path, capsys):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"aggregate": {"edge": 0.0}}))
    with pytest.raises(SystemExit):
        main([str(summary)])
    err = capsys.readouterr().err
    assert str(summary) in err and "no 'accuracy' block" in err
