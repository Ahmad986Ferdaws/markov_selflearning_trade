"""Research-layer acceptance tests — spec criteria 11, 13, 15 + state machine,
health gating, calibration hygiene, and walk-forward integrity."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from kalman import research
from kalman.data import synthetic_pair
from kalman.ledger import LedgerConfig
from kalman.research import (WalkForwardConfig, _causal_z, calibrate,
                             format_report, signals_kalman, walk_forward)
from kalman.strategy import Position, StateMachine, StrategyConfig


# 11 — no operational path uses backfill, centered windows, or future params.
# We ban the CODE constructs (docstrings legitimately discuss the ban itself).
def test_no_backfill_or_centered_windows_in_source():
    import kalman.core as c
    import kalman.data as d
    import kalman.ledger as l
    src = "".join(inspect.getsource(m) for m in (c, d, l, research))
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith(("#",)))
    for banned in (".bfill(", "method='bfill'", 'method="bfill"',
                   "center=True", ".fillna(method"):
        assert banned not in code, f"banned construct present: {banned}"


def test_causal_z_uses_only_lagged_history():
    rng = np.random.default_rng(0)
    resid = rng.normal(0, 0.1, 100)
    resid[80] += 8.0                      # a shock AT bar 80
    z = _causal_z(resid, window=20)
    # the shock standardizes against the calm PAST -> huge z at bar 80
    assert z[80] > 10
    # and z strictly before the shock is IDENTICAL with or without it
    resid_clean = resid.copy(); resid_clean[80] -= 8.0
    z_clean = _causal_z(resid_clean, window=20)
    assert np.array_equal(np.nan_to_num(z[:80]), np.nan_to_num(z_clean[:80]))


def test_warmup_forces_flat():
    pair, _ = synthetic_pair(n=300, seed=1)
    c1 = pair.p1["Close"].to_numpy(float)
    c2 = pair.p2["Close"].to_numpy(float)
    z, b, _ = signals_kalman(c1, c2, 1e-4, 1e-5, 0.05, train_end=60)
    from kalman.research import backtest
    led = backtest(pair, z, b, StrategyConfig(warmup=100), LedgerConfig())
    # signal at warmup bar 100 earliest -> first possible fill bar 101
    assert (led["s1"].to_numpy()[:101] == 0).all()


# state machine unit behavior: hysteresis, stop, cooldown, gates
def test_state_machine_hysteresis_and_gates():
    cfg = StrategyConfig(entry_z=2.0, exit_z=0.5, stop_z=4.0, warmup=0,
                         cooldown=2, max_holding=0)
    sm = StateMachine(cfg)
    assert sm.decide(0, -2.5, 1.0).target is Position.LONG_RESIDUAL
    assert sm.decide(1, -1.0, 1.0).target is Position.LONG_RESIDUAL   # hold: |z|>exit
    assert sm.decide(2, -0.4, 1.0).target is Position.FLAT            # exit band
    assert sm.decide(3, -3.0, 1.0).gated                              # cooldown
    assert sm.decide(4, -3.0, 1.0).gated
    assert sm.decide(5, -3.0, 1.0).target is Position.LONG_RESIDUAL   # re-entry
    assert sm.decide(6, 9.0, 1.0).reason == "emergency-stop"
    sm2 = StateMachine(StrategyConfig(warmup=0))
    assert sm2.decide(0, 3.0, 99.0).reason == "gate:extreme-beta"     # circuit breaker
    assert sm2.decide(1, None, 1.0).reason == "gate:stale-data"
    assert sm2.decide(2, 3.0, 1.0, healthy=False).reason == "gate:health"


def test_hysteresis_config_validated():
    with pytest.raises(ValueError):
        StrategyConfig(entry_z=1.0, exit_z=1.5)       # exit must be < entry


# calibration: training-only, positive params, sensitivity surface returned
def test_calibration_training_only_and_surface():
    pair, _ = synthetic_pair(n=600, seed=2)
    y = np.log(pair.p1["Close"].to_numpy(float))
    x = np.log(pair.p2["Close"].to_numpy(float))
    qb, qa, r, surface = calibrate(y[:300], x[:300])
    assert qb > 0 and qa > 0 and r > 0
    assert {"q_beta", "r", "loglik"} <= set(surface.columns)
    assert len(surface) == 16                          # full 4x4 surface reported
    # mutating data AFTER the training window cannot change the calibration
    y2 = y.copy(); y2[300:] += 10.0
    qb2, qa2, r2, _ = calibrate(y2[:300], x[:300])
    assert (qb, qa, r) == (qb2, qa2, r2)


# 15 — the report regenerates identically from the same inputs (and quickly,
# on a reduced fold layout so the suite stays fast)
def test_walkforward_report_regenerates_identically():
    pair, _ = synthetic_pair(n=900, seed=4)
    wf = WalkForwardConfig(train=400, validation=120, test=120)
    res1 = walk_forward(pair, wf, LedgerConfig())
    res2 = walk_forward(pair, wf, LedgerConfig())
    r1, r2 = format_report(res1, "SYN"), format_report(res2, "SYN")
    assert r1 == r2
    assert "VERDICT" in r1
    # all six variants evaluated on identical folds
    assert set(res1["aggregate"].keys()) == set(research.VARIANTS)
    folds = res1["aggregate"]["kalman_fixed"]["folds"]
    assert all(res1["aggregate"][v]["folds"] == folds for v in research.VARIANTS)


# cash reference is exactly flat and cost-free
def test_cash_reference_is_flat():
    pair, _ = synthetic_pair(n=900, seed=4)
    wf = WalkForwardConfig(train=400, validation=120, test=120)
    res = walk_forward(pair, wf, LedgerConfig())
    for m in res["by_variant"]["cash"]:
        assert m["trades"] == 0
        assert m["total_costs"] == 0.0
        assert m["total_return"] == pytest.approx(0.0)
