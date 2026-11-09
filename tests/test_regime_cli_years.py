"""regime-cli --years: the history length that selects the pinned snapshot
({symbol}_{years}y.pkl). run_phase0_report is replaced by a recorder so no
report is computed and nothing is read from disk."""

from __future__ import annotations

import pytest

from app.cli import regime as cli
from app.config import Settings

_SETTINGS = Settings(regime_symbol="BTC-USD", regime_window=20, regime_k=0.5,
                     regime_bull_thresh=0.02, regime_bear_thresh=-0.02,
                     fee_pct=0.3, slippage_pct=0.5)


@pytest.fixture
def recorded(monkeypatch):
    calls: list[dict] = []

    def fake_report(**kwargs):
        calls.append(kwargs)
        return "ok"

    monkeypatch.setattr(cli, "run_phase0_report", fake_report)
    monkeypatch.setattr(cli, "get_settings", lambda: _SETTINGS)
    return calls


@pytest.mark.parametrize("years", [1, 20])
def test_years_is_forwarded_as_given(recorded, capsys, years):
    cli.main(["--years", str(years)])
    [kwargs] = recorded
    assert kwargs["years"] == years
    assert capsys.readouterr().out.strip() == "ok"


def test_without_the_flag_the_report_keeps_its_own_default_of_three_years(recorded):
    cli.main([])
    [kwargs] = recorded
    assert kwargs.get("years", 3) == 3


def test_symbol_and_years_together_name_a_long_pin(recorded):
    cli.main(["--symbol", "SPY", "--years", "20"])
    [kwargs] = recorded
    assert (kwargs["symbol"], kwargs["years"]) == ("SPY", 20)


@pytest.mark.parametrize("value", ["0", "-1"])
def test_years_below_one_is_a_usage_error_not_a_report(recorded, value):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--years", value])
    assert exc.value.code == 2          # argparse usage error
    assert recorded == []               # the report was never attempted
