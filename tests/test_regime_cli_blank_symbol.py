"""regime-cli --symbol refuses a blank value instead of reporting on REGIME_SYMBOL.

`args.symbol or s.regime_symbol` let ``--symbol ""`` fall through to the
settings symbol, printing a report nobody asked for, and forwarded
``--symbol "   "`` verbatim (which would name a snapshot path "   _3y.pkl").
A blank value is now an argparse usage error; a padded one is stripped; omitting
the flag still falls back to the setting. run_phase0_report is replaced by a
capture so no report is computed and nothing is read from disk.
"""

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


@pytest.mark.parametrize("symbol", ["", "   "], ids=["empty", "spaces"])
def test_a_blank_symbol_is_a_usage_error_not_a_report_on_the_settings_symbol(captured, capsys, symbol):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--symbol", symbol])
    assert exc.value.code == 2          # argparse usage error
    assert captured == []               # the report was never attempted
    assert "--symbol must not be blank" in capsys.readouterr().err


def test_a_padded_symbol_is_stripped_before_it_names_the_report(captured):
    cli.main(["--symbol", " ETH-USD "])
    [kwargs] = captured
    assert kwargs["symbol"] == "ETH-USD"


def test_omitting_the_flag_still_reports_on_the_settings_symbol(captured):
    cli.main([])
    [kwargs] = captured
    assert kwargs["symbol"] == "BTC-USD"
