"""Research engine: calibration, health, baselines, walk-forward, metrics.

Honesty rules encoded here:
  * Calibration (Q/R) maximizes TRAINING-window predictive likelihood only.
  * Validation chooses among a SMALL PREDECLARED config set; the test period
    is evaluated once, untouched, per fold.
  * Every strategy variant — static OLS, rolling OLS, EW regression, fixed-Q
    Kalman, adaptive-Q Kalman, and cash — flows through the SAME state
    machine and the SAME two-leg ledger with the SAME costs. A variant can
    only differ in how it produces (z_t, beta_t).
  * All rolling statistics are trailing (lagged). No centered windows, no
    backfill, flat until causal history exists.
  * Kalman optimality is conditional on correct model specification; nothing
    here treats filter output as evidence of edge. If the walk-forward shows
    no robust edge after costs, the report says so in plain words.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .core import AdaptiveQ, PairFilter, StepRecord, run_pair_filter
from .data import PairData
from .ledger import LedgerConfig, run_ledger
from .pairs import PairDiagnostics, qualify_pair
from .strategy import Decision, Position, StateMachine, StrategyConfig

BARS_PER_YEAR = 252


# --------------------------------------------------------------------------- #
# Signal producers — each yields per-bar (z, beta); identical downstream path
# --------------------------------------------------------------------------- #
def _causal_z(resid: np.ndarray, window: int = 60) -> np.ndarray:
    """Standardize residuals by TRAILING mean/std (ending at t-1). First
    `window` values are NaN -> the state machine's stale-data gate holds flat."""
    z = np.full(len(resid), np.nan)
    for t in range(window, len(resid)):
        past = resid[t - window:t]                  # ends at t-1: lagged
        m, s = float(np.mean(past)), float(np.std(past, ddof=1))
        if np.isfinite(resid[t]) and s > 0:
            z[t] = (resid[t] - m) / s
    return z


def signals_static_ols(y: np.ndarray, x: np.ndarray, train_end: int,
                       z_window: int = 60) -> tuple[np.ndarray, np.ndarray]:
    A = np.column_stack([x[:train_end], np.ones(train_end)])
    coef, *_ = np.linalg.lstsq(A, y[:train_end], rcond=None)
    resid = y - (coef[0] * x + coef[1])
    return _causal_z(resid, z_window), np.full(len(y), float(coef[0]))


def signals_rolling_ols(y: np.ndarray, x: np.ndarray, window: int = 120,
                        z_window: int = 60) -> tuple[np.ndarray, np.ndarray]:
    n = len(y)
    beta = np.full(n, np.nan)
    resid = np.full(n, np.nan)
    for t in range(window, n):
        ys, xs = y[t - window:t], x[t - window:t]   # ends at t-1: lagged fit
        A = np.column_stack([xs, np.ones(window)])
        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
        beta[t] = coef[0]
        resid[t] = y[t] - (coef[0] * x[t] + coef[1])
    return _causal_z(resid, z_window), beta


def signals_ew_ols(y: np.ndarray, x: np.ndarray, lam: float = 0.99,
                   z_window: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Exponentially weighted recursive least squares (forgetting factor lam).
    The bar-t residual uses the state fitted through t-1 (prediction error)."""
    n = len(y)
    theta = np.zeros(2)
    Pm = np.eye(2) * 1e3
    beta = np.full(n, np.nan)
    resid = np.full(n, np.nan)
    for t in range(n):
        h = np.array([x[t], 1.0])
        resid[t] = y[t] - h @ theta                  # PRE-update residual
        beta[t] = theta[0]
        denom = lam + h @ Pm @ h
        k = (Pm @ h) / denom
        theta = theta + k * resid[t]
        Pm = (Pm - np.outer(k, h @ Pm)) / lam
    return _causal_z(resid, z_window), beta


def signals_kalman(y: np.ndarray, x: np.ndarray, q_beta: float, q_alpha: float,
                   r: float, train_end: int, adaptive: AdaptiveQ | None = None,
                   ) -> tuple[np.ndarray, np.ndarray, list[StepRecord]]:
    f = PairFilter.from_ols(y[:train_end], x[:train_end], q_beta, q_alpha, r,
                            adaptive=adaptive)
    recs = run_pair_filter(y, x, f)
    z = np.array([r_.z if r_.z is not None else np.nan for r_ in recs])
    beta = np.array([float(r_.x_post[0]) for r_ in recs])
    return z, beta, recs


# --------------------------------------------------------------------------- #
# Calibration — training-only predictive likelihood
# --------------------------------------------------------------------------- #
def train_loglik(y: np.ndarray, x: np.ndarray, q_beta: float, q_alpha: float,
                 r: float, burn: int = 20) -> float:
    f = PairFilter.from_ols(y, x, q_beta, q_alpha, r)
    recs = run_pair_filter(y, x, f)
    lls = [rec.loglik for rec in recs[burn:] if rec.loglik is not None]
    return float(np.sum(lls)) if lls else -np.inf


def calibrate(y_train: np.ndarray, x_train: np.ndarray,
              q_betas=(1e-7, 1e-6, 1e-5, 1e-4),
              rs=(1e-5, 1e-4, 1e-3, 1e-2),
              q_alpha_frac: float = 0.1) -> tuple[float, float, float, pd.DataFrame]:
    """Positive log-space grid over (q_beta, r); q_alpha tied as a fraction of
    q_beta (different units — separately configurable, jointly gridding all
    three is left to the sensitivity surface). Returns best + full surface."""
    rows = []
    best = (-np.inf, None)
    for qb, r in itertools.product(q_betas, rs):
        ll = train_loglik(y_train, x_train, qb, qb * q_alpha_frac, r)
        rows.append({"q_beta": qb, "r": r, "loglik": ll})
        if ll > best[0]:
            best = (ll, (qb, qb * q_alpha_frac, r))
    if best[1] is None:
        raise RuntimeError("calibration failed: all likelihoods -inf")
    qb, qa, r = best[1]
    return qb, qa, r, pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Health monitoring — causal, unit-aware (never raw covariance trace)
# --------------------------------------------------------------------------- #
def health_series(recs: list[StepRecord], nis_window: int = 40,
                  nis_limit: float = 2.5, beta_unc_limit: float = 1.0,
                  cusum_k: float = 0.5, cusum_h: float = 15.0) -> np.ndarray:
    """healthy[t] uses information through t-1 ONLY (lagged windows), so it can
    gate the decision made after bar t without peeking at bar t."""
    n = len(recs)
    nis = np.array([r.nis if r.nis is not None else np.nan for r in recs])
    z = np.array([r.z if r.z is not None else np.nan for r in recs])
    beta = np.array([float(r.x_post[0]) for r in recs])
    beta_sd = np.array([math.sqrt(max(r.P_post[0, 0], 0.0)) for r in recs])

    healthy = np.ones(n, dtype=bool)
    cus = 0.0
    for t in range(n):
        past_nis = nis[max(0, t - nis_window):t]     # ends at t-1
        past_nis = past_nis[np.isfinite(past_nis)]
        if len(past_nis) >= nis_window // 2 and float(np.mean(past_nis)) > nis_limit:
            healthy[t] = False
        if t > 0 and np.isfinite(z[t - 1]):          # lagged CUSUM on |z|
            cus = max(0.0, cus + abs(z[t - 1]) - 1.0 - cusum_k)
            if cus > cusum_h:
                healthy[t] = False
        if t > 0 and abs(beta[t - 1]) > 1e-9 and \
                beta_sd[t - 1] / abs(beta[t - 1]) > beta_unc_limit:
            healthy[t] = False
    return healthy


def innovation_diagnostics(recs: list[StepRecord], start: int = 0) -> dict:
    z = np.array([r.z for r in recs[start:] if r.z is not None])
    nis = np.array([r.nis for r in recs[start:] if r.nis is not None])
    if len(z) < 10:
        return {"n": int(len(z))}
    ac1 = float(np.corrcoef(z[:-1], z[1:])[0, 1])
    return {
        "n": int(len(z)),
        "z_mean": float(np.mean(z)),
        "z_std": float(np.std(z, ddof=1)),
        "z_autocorr_1": ac1,
        "coverage_95": float(np.mean(np.abs(z) <= 1.96)),
        "nis_mean": float(np.mean(nis)),
    }


# --------------------------------------------------------------------------- #
# Backtest of one signal stream (shared by every variant)
# --------------------------------------------------------------------------- #
def backtest(pair: PairData, z: np.ndarray, beta: np.ndarray,
             scfg: StrategyConfig, lcfg: LedgerConfig,
             healthy: np.ndarray | None = None,
             start: int = 0, end: int | None = None) -> pd.DataFrame:
    end = len(pair) if end is None else end
    sl = slice(start, end)
    idx = pair.index[sl]
    o1 = pair.p1["Open"].to_numpy(float)[sl]
    o2 = pair.p2["Open"].to_numpy(float)[sl]
    c1 = pair.p1["Close"].to_numpy(float)[sl]
    c2 = pair.p2["Close"].to_numpy(float)[sl]
    zz, bb = z[sl], beta[sl]
    hh = (np.ones(len(zz), bool) if healthy is None else healthy[sl])

    sm = StateMachine(scfg)
    decisions: list[Decision] = []
    for t in range(len(zz)):
        zt = None if not np.isfinite(zz[t]) else float(zz[t])
        bt = float(bb[t]) if np.isfinite(bb[t]) else 0.0
        decisions.append(sm.decide(t, zt, bt, healthy=bool(hh[t])))
    return run_ledger(idx, o1, o2, c1, c2, decisions, np.nan_to_num(bb), lcfg)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def perf_metrics(led: pd.DataFrame, bars_per_year: int = BARS_PER_YEAR) -> dict:
    eq = led["equity"].to_numpy(float)
    rets = led["ret"].to_numpy(float)
    n = len(eq)
    if n < 2 or eq[0] <= 0:
        return {"n": n}
    years = n / bars_per_year
    cagr = (eq[-1] / eq[0]) ** (1 / years) - 1 if years > 0 and eq[-1] > 0 else np.nan
    vol = float(np.std(rets, ddof=1)) * math.sqrt(bars_per_year)
    sharpe = float(np.mean(rets) / np.std(rets, ddof=1) * math.sqrt(bars_per_year)) \
        if np.std(rets, ddof=1) > 0 else 0.0
    downside = rets[rets < 0]
    sortino = float(np.mean(rets) / np.std(downside, ddof=1) * math.sqrt(bars_per_year)) \
        if len(downside) > 1 and np.std(downside, ddof=1) > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    mdd = float(np.min(eq / peak - 1.0))
    calmar = float(cagr / abs(mdd)) if mdd < 0 and np.isfinite(cagr) else np.nan
    fills = led[led["fill_reason"].str.startswith("fill:", na=False)]
    entries = fills[fills["fill_reason"].str.contains("enter")]
    return {
        "n": n, "total_return": float(eq[-1] / eq[0] - 1.0), "cagr": float(cagr),
        "vol": vol, "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
        "max_drawdown": mdd, "turnover": float(led["turnover"].sum()),
        "total_costs": float(led["costs"].sum()), "trades": int(len(entries)),
        "avg_gross": float(led["gross"].mean()),
    }


def sharpe_ci_block_bootstrap(rets: np.ndarray, n_boot: int = 500,
                              block: int = 20, seed: int = 0,
                              bars_per_year: int = BARS_PER_YEAR) -> tuple[float, float]:
    """Moving-block bootstrap 90% CI for the annualized Sharpe."""
    rng = np.random.default_rng(seed)
    n = len(rets)
    if n < block * 2:
        return (float("nan"), float("nan"))
    sharpes = []
    for _ in range(n_boot):
        k = int(np.ceil(n / block))
        starts = rng.integers(0, n - block, size=k)
        sample = np.concatenate([rets[s:s + block] for s in starts])[:n]
        sd = np.std(sample, ddof=1)
        sharpes.append(np.mean(sample) / sd * math.sqrt(bars_per_year) if sd > 0 else 0.0)
    lo, hi = np.percentile(sharpes, [5, 95])
    return float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Walk-forward
# --------------------------------------------------------------------------- #
@dataclass
class WalkForwardConfig:
    train: int = 1000
    validation: int = 250
    test: int = 250
    expanding: bool = True            # expanding vs rolling training window
    # small, PREDECLARED validation set (entry_z, exit_z):
    entry_exit_grid: tuple = ((1.5, 0.5), (2.0, 0.5), (2.5, 1.0))


VARIANTS = ("static_ols", "rolling_ols", "ew_ols", "kalman_fixed", "kalman_adaptive", "cash")


def _variant_signals(name: str, y, x, train_end, qb, qa, r, sigma_ref):
    if name == "static_ols":
        return signals_static_ols(y, x, train_end) + (None,)
    if name == "rolling_ols":
        return signals_rolling_ols(y, x) + (None,)
    if name == "ew_ols":
        return signals_ew_ols(y, x) + (None,)
    if name == "kalman_fixed":
        z, b, recs = signals_kalman(y, x, qb, qa, r, train_end)
        return z, b, recs
    if name == "kalman_adaptive":
        ad = AdaptiveQ(sigma_ref=sigma_ref)
        z, b, recs = signals_kalman(y, x, qb, qa, r, train_end, adaptive=ad)
        return z, b, recs
    if name == "cash":
        n = len(y)
        return np.full(n, np.nan), np.zeros(n), None    # stale-gate -> always flat
    raise ValueError(name)


def walk_forward(pair: PairData, wf: WalkForwardConfig,
                 lcfg: LedgerConfig, use_log: bool = True,
                 seed: int = 0) -> dict:
    """Chronological folds: train (qualify + calibrate) -> validation (pick
    entry/exit from the predeclared grid by net Sharpe) -> test (once)."""
    c1 = pair.p1["Close"].to_numpy(float)
    c2 = pair.p2["Close"].to_numpy(float)
    y = np.log(c1) if use_log else c1
    x = np.log(c2) if use_log else c2
    n = len(y)

    folds = []
    fold_start = 0
    while fold_start + wf.train + wf.validation + wf.test <= n:
        tr0 = 0 if wf.expanding else fold_start
        tr1 = fold_start + wf.train
        va1 = tr1 + wf.validation
        te1 = va1 + wf.test
        folds.append((tr0, tr1, va1, te1))
        fold_start += wf.test

    results: dict = {"folds": [], "by_variant": {v: [] for v in VARIANTS}}
    for fi, (tr0, tr1, va1, te1) in enumerate(folds):
        diag = qualify_pair(pair, tr1, use_log=use_log)
        qb, qa, r, _surface = calibrate(y[tr0:tr1], x[tr0:tr1])
        sigma_ref = float(np.std(np.diff(y[tr0:tr1]), ddof=1))

        fold_rec = {"fold": fi, "train": (tr0, tr1), "validation": (tr1, va1),
                    "test": (va1, te1), "diagnostics": diag.summary(),
                    "calibrated": {"q_beta": qb, "q_alpha": qa, "r": r}}

        for v in VARIANTS:
            z, beta, recs = _variant_signals(v, y, x, tr1, qb, qa, r, sigma_ref)
            healthy = health_series(recs) if recs is not None else None

            # --- validation: pick (entry, exit) from the predeclared grid ---
            best = (-np.inf, wf.entry_exit_grid[0])
            if v != "cash":
                for ez, xz in wf.entry_exit_grid:
                    scfg = StrategyConfig(entry_z=ez, exit_z=xz)
                    led = backtest(pair, z, beta, scfg, lcfg, healthy, tr1, va1)
                    s = perf_metrics(led).get("sharpe", -np.inf)
                    if s > best[0]:
                        best = (s, (ez, xz))
            ez, xz = best[1]

            # --- test: once, untouched ---
            scfg = StrategyConfig(entry_z=ez, exit_z=xz)
            led = backtest(pair, z, beta, scfg, lcfg, healthy, va1, te1)
            m = perf_metrics(led)
            m["chosen_entry_z"], m["chosen_exit_z"] = ez, xz
            if recs is not None:
                m["innovations"] = innovation_diagnostics(recs[va1:te1])
            results["by_variant"][v].append(m)
        results["folds"].append(fold_rec)

    # aggregate per variant across test folds
    agg = {}
    for v in VARIANTS:
        ms = results["by_variant"][v]
        if ms:
            agg[v] = {
                "folds": len(ms),
                "mean_test_sharpe": float(np.mean([m.get("sharpe", 0.0) for m in ms])),
                "total_test_return": float(np.prod(
                    [1 + m.get("total_return", 0.0) for m in ms]) - 1),
                "total_costs": float(np.sum([m.get("total_costs", 0.0) for m in ms])),
                "trades": int(np.sum([m.get("trades", 0) for m in ms])),
            }
    results["aggregate"] = agg
    return results


def format_report(res: dict, pair_name: str) -> str:
    lines = [f"=== Kalman pairs walk-forward: {pair_name} ===",
             f"folds: {len(res['folds'])}  (train/val/test chronological, test untouched)",
             ""]
    for fr in res["folds"][:1]:
        lines.append(f"fold 0 diagnostics (train only): {fr['diagnostics']}")
        lines.append(f"fold 0 calibration: {fr['calibrated']}")
        lines.append("")
    lines.append(f"{'variant':<16} {'folds':>5} {'mean test Sharpe':>17} "
                 f"{'total test ret':>15} {'trades':>7} {'costs':>10}")
    for v, a in res["aggregate"].items():
        lines.append(f"{v:<16} {a['folds']:>5} {a['mean_test_sharpe']:>17.2f} "
                     f"{a['total_test_return']:>14.1%} {a['trades']:>7} "
                     f"{a['total_costs']:>10.0f}")
    ks = res["aggregate"].get("kalman_fixed", {})
    cash = res["aggregate"].get("cash", {})
    lines.append("")
    if ks and ks.get("mean_test_sharpe", 0) <= max(0.5, cash.get("mean_test_sharpe", 0)):
        lines.append("VERDICT: no robust edge after costs in this walk-forward. "
                     "The Kalman machinery is measurably correct; the EDGE is absent "
                     "— which is the honest, expected outcome for a well-known "
                     "public strategy on liquid instruments.")
    else:
        lines.append("VERDICT: test Sharpe is positive — treat with suspicion: "
                     "few folds, correlated bars, selection over the predeclared "
                     "grid. Verify with bootstrap CIs before believing it.")
    return "\n".join(lines)


def audit_table(recs: list[StepRecord], decisions: list[Decision],
                led: pd.DataFrame) -> pd.DataFrame:
    """Trade-level audit: prior state, innovation, signal, posterior, targets,
    fills, turnover, costs, realized P&L — one row per bar, join-safe."""
    rows = []
    for i, rec in enumerate(recs[:len(led)]):
        d = decisions[i] if i < len(decisions) else None
        rows.append({
            "t": rec.t,
            "beta_prior": float(rec.x_prior[0]), "alpha_prior": float(rec.x_prior[1]),
            "innovation": rec.innovation, "S": rec.innovation_var, "z": rec.z,
            "nis": rec.nis, "loglik": rec.loglik,
            "beta_post": float(rec.x_post[0]), "alpha_post": float(rec.x_post[1]),
            "beta_sd": math.sqrt(max(rec.P_post[0, 0], 0.0)),
            "decision": d.reason if d else None,
            "target": d.target.name if d else None,
        })
    at = pd.DataFrame(rows)
    led2 = led.reset_index()[["t", "s1", "s2", "cash", "equity", "turnover",
                              "costs", "fill_reason"]]
    return at.merge(led2, on="t", how="left")
