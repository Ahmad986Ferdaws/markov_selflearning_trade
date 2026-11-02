"""regime-cli argument handling. run_phase0_report is replaced by a capture so
no report is computed and nothing is read from disk."""

from __future__ import annotations

import pytest

from app.cli import regime as cli
from app.config import Settings

_SETTINGS = Settings(regime_symbol="BTC-USD", regime_window=20, regime_k=0.5,
                     regime_bull_thresh=0.02, regime_bear_thresh=-0.02,
                     fee_pct=0.3, slippage_pct=0.5)


@pytest.fixture
def captured(monkeypatch):
    calls: list[dict] = []

    def fake_report(**kwargs):
        calls.append(kwargs)
        return "REPORT"

    monkeypatch.setattr(cli, "run_phase0_report", fake_report)
    monkeypatch.setattr(cli, "get_settings", lambda: _SETTINGS)
    return calls


def test_no_flags_means_every_value_comes_from_settings(captured, capsys):
    cli.main([])
    assert captured == [dict(symbol="BTC-USD", window=20, k=0.5, bull_thresh=0.02,
                             bear_thresh=-0.02, fee_pct=0.3, slippage_pct=0.5)]
    assert capsys.readouterr().out.strip() == "REPORT"


def test_each_flag_overrides_only_the_setting_it_names(captured):
    cli.main(["--symbol", "ETH-USD", "--window", "30", "--k", "0.2",
              "--fee-pct", "0", "--slippage-pct", "1.25"])
    [kwargs] = captured
    assert kwargs["symbol"] == "ETH-USD"
    assert (kwargs["window"], kwargs["k"]) == (30, 0.2)
    assert (kwargs["fee_pct"], kwargs["slippage_pct"]) == (0.0, 1.25)
    assert (kwargs["bull_thresh"], kwargs["bear_thresh"]) == (0.02, -0.02)   # not a flag; untouched


def test_a_single_override_leaves_the_others_on_their_settings(captured):
    cli.main(["--window", "10"])
    [kwargs] = captured
    assert kwargs["window"] == 10
    assert (kwargs["symbol"], kwargs["k"]) == ("BTC-USD", 0.5)
