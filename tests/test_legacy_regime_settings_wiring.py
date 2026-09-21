"""REGIME_K and REGIME_MODE must reach the legacy regime replay and runner.

Both legacy call sites of regime_feature passed only window and the
bull/bear thresholds — the pair define_states documents as INERT in zscore
mode — and omitted k, so tuning REGIME_K in .env changed regime-cli's report
but left the intraday runner and GET /runs/{id}/comparison byte-identical.
REGIME_MODE was read by nothing at all.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pandas as pd

from app.config import Settings
from app.services import runner
from app.services.comparison import _build_regime_lookup
from app.services.data_cache import load_or_fetch
from app.services.regime import regime_feature


def _snapshots(history: pd.DataFrame, n: int = 300):
    # one synthetic snapshot per trading day, stamped mid-session
    days = history.index[-n:]
    return [SimpleNamespace(created_at=(d + pd.Timedelta(hours=15)).to_pydatetime()) for d in days]


def _states(lookup):
    return [feat.state for _, feat in sorted(lookup.items())]


def test_regime_k_changes_the_comparison_regime_lookup():
    history, _ = load_or_fetch("BTC-USD", years=3)
    snaps = _snapshots(history)
    tight = _build_regime_lookup(snaps, history, Settings(_env_file=None, regime_k=0.05))
    loose = _build_regime_lookup(snaps, history, Settings(_env_file=None, regime_k=3.0))
    assert _states(tight) != _states(loose)          # used to be 0/300 dates different
    assert set(_states(loose)) == {"sideways"}       # k=3: nothing clears the band


def test_regime_mode_changes_the_comparison_regime_lookup():
    history, _ = load_or_fetch("BTC-USD", years=3)
    snaps = _snapshots(history)
    z = _build_regime_lookup(snaps, history, Settings(_env_file=None, regime_mode="zscore"))
    a = _build_regime_lookup(snaps, history, Settings(_env_file=None, regime_mode="absolute",
                                                      regime_bull_thresh=0.001,
                                                      regime_bear_thresh=-0.001))
    assert _states(z) != _states(a)


def test_regime_feature_accepts_mode_and_matches_define_states_semantics():
    history, _ = load_or_fetch("BTC-USD", years=3)
    zs = regime_feature(history, window=20, k=0.5, mode="zscore")
    ab = regime_feature(history, window=20, mode="absolute", bull_thresh=0.02, bear_thresh=-0.02)
    assert zs.state in {"bull", "bear", "sideways"} and ab.state in {"bull", "bear", "sideways"}
    # default mode is unchanged (zscore): omitting mode == passing it
    assert regime_feature(history, window=20, k=0.5).p_next == zs.p_next


def test_runner_passes_k_and_mode_from_settings():
    # source-level tripwire, as tests/test_kalman_ledger.py does for the ledger
    src = inspect.getsource(runner.run_loop)
    assert "k=settings.regime_k" in src
    assert "mode=settings.regime_mode" in src


# --- a typo must not silently select absolute mode (Codex review on #30) ------
def test_unknown_regime_mode_is_rejected_at_settings_and_at_the_feature():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(_env_file=None, regime_mode="zscor")
    history, _ = load_or_fetch("BTC-USD", years=3)
    with pytest.raises(ValueError, match="regime mode"):
        regime_feature(history, window=20, mode="absolut")


def test_regime_mode_is_case_and_whitespace_insensitive_but_typos_fail():
    import pytest
    from pydantic import ValidationError

    assert Settings(_env_file=None, regime_mode="Absolute").regime_mode == "absolute"
    assert Settings(_env_file=None, regime_mode=" ZSCORE ").regime_mode == "zscore"
    with pytest.raises(ValidationError):
        Settings(_env_file=None, regime_mode="zsc ore")
    assert Settings(_env_file=".env.example").regime_mode == "zscore"     # template documents it
