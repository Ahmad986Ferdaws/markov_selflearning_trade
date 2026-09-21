"""Argument handling for evaluate-cli and kalman-cli.

The hand-rolled argv scan silently ignored a bare `--provider`, took the NEXT
FLAG as the provider name (`--provider --cache-only` ran a full evaluation
with "unknown llm_provider='--cache-only'" and wrote a record, exit 0), ran a
~40 s evaluation on `--help` or any typo, and let a `.env` grid typo surface
as a traceback after the data load. kalman-cli's `--help` exited 1.
"""

from __future__ import annotations

import pytest

from app.cli import evaluate as evaluate_cli
from app.config import Settings
from kalman import cli as kalman_cli


@pytest.fixture
def no_data(monkeypatch):
    """Fail loudly if the CLI reaches the data layer."""
    def boom(*a, **k):
        raise AssertionError("load_or_fetch must not be reached for a bad invocation")
    monkeypatch.setattr(evaluate_cli, "load_or_fetch", boom)


@pytest.mark.parametrize("argv", [
    ["--provider"],                 # missing value used to be silently ignored
    ["--provider", "--cache-only"], # next flag used to become the provider name
    ["--provider", "bogus"],
    ["--years", "0"],
    ["--years", "three"],
    ["--not-a-flag"],
    ["--ref"],                      # abbreviation would have meant --refresh: a live fetch
    ["--prov", "none"],
    ["--sym", "BTC-USD"],
])
def test_evaluate_bad_invocations_exit_2_before_touching_data(argv, no_data, capsys):
    with pytest.raises(SystemExit) as exc:
        evaluate_cli.main(argv)
    assert exc.value.code == 2
    assert "Traceback" not in capsys.readouterr().err


def test_evaluate_help_exits_0_with_usage(no_data, capsys):
    with pytest.raises(SystemExit) as exc:
        evaluate_cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "usage: evaluate-cli" in out and "--cache-only" in out and "--symbol" in out


def test_evaluate_env_grid_typo_is_one_line_not_a_traceback(no_data, monkeypatch, capsys):
    monkeypatch.setattr(evaluate_cli, "get_settings",
                        lambda: Settings(_env_file=None, eval_grid_windows="10,abc"))
    with pytest.raises(SystemExit) as exc:
        evaluate_cli.main([])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("evaluate-cli: invalid EVAL_GRID_WINDOWS") and "abc" in err


def test_evaluate_engine_value_error_is_one_line(monkeypatch, capsys):
    import pandas as pd
    tiny = pd.DataFrame({"Close": [1.0, 1.1, 1.2]},
                        index=pd.date_range("2020-01-01", periods=3, freq="D"))
    monkeypatch.setattr(evaluate_cli, "load_or_fetch", lambda *a, **k: (tiny, "cache:tiny"))
    monkeypatch.setattr(evaluate_cli, "get_settings", lambda: Settings(_env_file=None))
    with pytest.raises(SystemExit) as exc:
        evaluate_cli.main(["--provider", "none"])
    assert exc.value.code == 2
    assert "Not enough history" in capsys.readouterr().err


def test_evaluate_symbol_flag_runs_the_pinned_snapshot(monkeypatch, tmp_path, capsys):
    # end-to-end on the committed BTC snapshot; the record goes to tmp, not results/
    from app.services.data_cache import save_run_record

    monkeypatch.setattr(evaluate_cli, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(evaluate_cli, "save_run_record", lambda report: save_run_record(report, tmp_path))
    evaluate_cli.main(["--symbol", "BTC-USD", "--years", "3", "--provider", "none"])
    out = capsys.readouterr().out
    assert "cache:data/snapshots/BTC-USD_3y.pkl" in out
    assert "Model        hit-rate: 90.9%" in out          # the CI determinism grep
    assert "Persistence  hit-rate: 90.9%" in out
    assert list(tmp_path.glob("BTC-USD_*.json"))


# --- kalman-cli -----------------------------------------------------------------
def test_kalman_help_exits_0(capsys):
    with pytest.raises(SystemExit) as exc:
        kalman_cli.main(["--help"])
    assert exc.value.code == 0
    assert "usage: kalman-cli" in capsys.readouterr().out


def test_kalman_unknown_command_exits_2(capsys):
    with pytest.raises(SystemExit) as exc:
        kalman_cli.main(["bogus"])
    assert exc.value.code == 2


def test_kalman_default_is_demo(monkeypatch):
    calls = []
    monkeypatch.setitem(kalman_cli.COMMANDS, "demo", lambda: calls.append("demo"))
    kalman_cli.main([])
    assert calls == ["demo"]


def test_kalman_pair_symbols_reach_the_command(monkeypatch):
    calls = []
    monkeypatch.setitem(kalman_cli.COMMANDS, "pairs", lambda *s: calls.append(s))
    kalman_cli.main(["pairs", "AAPL", "MSFT"])
    kalman_cli.main(["pairs"])
    assert calls == [("AAPL", "MSFT"), ()]


@pytest.mark.parametrize("argv", [["demo", "AAPL"], ["pairs", "A", "B", "C"]])
def test_kalman_symbol_arity_is_checked(argv, monkeypatch):
    monkeypatch.setitem(kalman_cli.COMMANDS, "demo", lambda: None)
    monkeypatch.setitem(kalman_cli.COMMANDS, "pairs", lambda *s: None)
    with pytest.raises(SystemExit) as exc:
        kalman_cli.main(argv)
    assert exc.value.code == 2
