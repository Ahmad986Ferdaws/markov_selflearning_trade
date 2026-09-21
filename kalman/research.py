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
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .core import AdaptiveQ, PairFilter, StepRecord, run_pair_filter
from .data import PairData
from .ledger import LedgerConfig, run_ledger
from .pairs import qualify_pair
from .strategy import Decision, StateMachine, StrategyConfig

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
                       z_window: int = 60, train_start: int = 0,
                       ) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = y[train_start:train_end], x[train_start:train_end]
    A = np.column_stack([xs, np.ones(len(xs))])
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
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
                   train_start: int = 0,
                   ) -> tuple[np.ndarray, np.ndarray, list[StepRecord]]:
    f = PairFilter.from_ols(y[train_start:train_end], x[train_start:train_end],
                            q_beta, q_alpha, r, adaptive=adaptive)
    recs = run_pair_filter(y, x, f)
    z = np.array([r_.z if r_.z is not None else np.nan for r_ in recs])
    beta = np.array([float(r_.x_post[0]) for r_ in recs])
    return z, beta, recs


# --------------------------------------------------------------------------- #
# Calibration — training-only predictive likelihood
# --------------------------------------------------------------------------- #
def train_loglik(y: np.ndarray, x: np.ndarray, q_beta: float, q_alpha: float,
                 r: float, init: int | None = None) -> float:
    """PREDICTIVE likelihood: the OLS prior is fitted on the first `init` bars
    only, and only likelihoods from `init` onward are scored — so the prior has
    never seen any bar it is scored on. (Review found the old version fitted
    x0/P0 on the whole window and then scored that same window, biasing the
    grid toward stiff small-q filters.)"""
    n = len(y)
    i0 = init if init is not None else max(60, n // 5)
    if i0 >= n - 10:
        return -np.inf
    f = PairFilter.from_ols(y[:i0], x[:i0], q_beta, q_alpha, r)
    recs = run_pair_filter(y, x, f)
    lls = [rec.loglik for rec in recs[i0:] if rec.loglik is not None]
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
                cus = 0.0        # reset on alarm: gate for THIS bar, then re-arm
                                 # (no reset made gate duration scale with overshoot)
        if t > 0 and abs(beta[t - 1]) > 1e-9 and \
                beta_sd[t - 1] / abs(beta[t - 1]) > beta_unc_limit:
            healthy[t] = False
    return healthy


def health_from_signal(z: np.ndarray, window: int = 40, z2_limit: float = 2.5,
                       cusum_k: float = 0.5, cusum_h: float = 15.0) -> np.ndarray:
    """Variant-agnostic relationship-health gate built ONLY from the signal
    stream, so EVERY variant in the walk-forward faces the identical gate.
    (Review found the old comparison gated only the Kalman variants — the
    baselines faced four gates, Kalman five, violating the identical-
    assumptions contract.) Causal: healthy[t] uses z through t-1 only."""
    n = len(z)
    z2 = np.square(z)
    healthy = np.ones(n, dtype=bool)
    cus = 0.0
    for t in range(n):
        past = z2[max(0, t - window):t]
        past = past[np.isfinite(past)]
        if len(past) >= window // 2 and float(np.mean(past)) > z2_limit:
            healthy[t] = False
        if t > 0 and np.isfinite(z[t - 1]):
            cus = max(0.0, cus + abs(z[t - 1]) - 1.0 - cusum_k)
            if cus > cusum_h:
                healthy[t] = False
                cus = 0.0
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
             start: int = 0, end: int | None = None,
             with_decisions: bool = False):
    """Run one signal stream through the shared state machine + ledger.
    Returns the ledger DataFrame; with `with_decisions=True` returns
    (ledger, decisions) so `audit_table` can be built correctly."""
    end = len(pair) if end is None else end
    sl = slice(start, end)
    idx = pair.index[sl]
    o1 = pair.p1["Open"].to_numpy(float)[sl]
    o2 = pair.p2["Open"].to_numpy(float)[sl]
    c1 = pair.p1["Close"].to_numpy(float)[sl]
    c2 = pair.p2["Close"].to_numpy(float)[sl]
    zz, bb = z[sl], beta[sl]
    hh = (np.ones(len(zz), bool) if healthy is None else healthy[sl])

    # _seen=start: the warm-up counts CAUSAL HISTORY, which exists before this
    # slice. (Review: a fresh count per fold re-blanked the first 60 bars of
    # every validation/test window — 24% of each fold structurally dead.)
    sm = StateMachine(scfg, _seen=start)
    decisions: list[Decision] = []
    for t in range(len(zz)):
        zt = None if not np.isfinite(zz[t]) else float(zz[t])
        bt = float(bb[t]) if np.isfinite(bb[t]) else 0.0
        decisions.append(sm.decide(t, zt, bt, healthy=bool(hh[t])))
    led = run_ledger(idx, o1, o2, c1, c2, decisions, np.nan_to_num(bb), lcfg)
    return (led, decisions) if with_decisions else led


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def perf_metrics(led: pd.DataFrame, bars_per_year: int = BARS_PER_YEAR) -> dict:
    eq = led["equity"].to_numpy(float)
    rets = led["ret"].to_numpy(float)[1:]   # drop the forced 0.0 at bar 0
    n = len(eq)
    if n < 3 or eq[0] <= 0:
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
    # entries = flat -> invested transitions in the EXECUTED holdings; robust
    # to retried fills, whose reason string is 'fill:hold' (review finding)
    gross_arr = led["gross"].to_numpy(float)
    prev_gross = np.concatenate([[0.0], gross_arr[:-1]])
    entries = int(np.sum((prev_gross == 0.0) & (gross_arr > 0.0)))
    return {
        "n": n, "total_return": float(eq[-1] / eq[0] - 1.0), "cagr": float(cagr),
        "vol": vol, "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
        "max_drawdown": mdd, "turnover": float(led["turnover"].sum()),
        "total_costs": float(led["costs"].sum()), "trades": entries,
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
        starts = rng.integers(0, n - block + 1, size=k)   # final block included
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


def _variant_signals(name: str, y, x, train_end, qb, qa, r, sigma_ref,
                     train_start: int = 0):
    if name == "static_ols":
        return signals_static_ols(y, x, train_end, train_start=train_start) + (None,)
    if name == "rolling_ols":
        return signals_rolling_ols(y, x) + (None,)
    if name == "ew_ols":
        return signals_ew_ols(y, x) + (None,)
    if name == "kalman_fixed":
        z, b, recs = signals_kalman(y, x, qb, qa, r, train_end,
                                    train_start=train_start)
        return z, b, recs
    if name == "kalman_adaptive":
        ad = AdaptiveQ(sigma_ref=sigma_ref)
        z, b, recs = signals_kalman(y, x, qb, qa, r, train_end, adaptive=ad,
                                    train_start=train_start)
        return z, b, recs
    if name == "cash":
        n = len(y)
        return np.full(n, np.nan), np.zeros(n), None    # stale-gate -> always flat
    raise ValueError(name)


def walk_forward(pair: PairData, wf: WalkForwardConfig,
                 lcfg: LedgerConfig, use_log: bool = True) -> dict:
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

    results: dict = {"folds": [], "by_variant": {v: [] for v in VARIANTS},
                     "test_returns": {v: [] for v in VARIANTS}}
    for fi, (tr0, tr1, va1, te1) in enumerate(folds):
        diag = qualify_pair(pair, tr1, use_log=use_log, train_start=tr0)
        qb, qa, r, _surface = calibrate(y[tr0:tr1], x[tr0:tr1])
        sigma_ref = float(np.std(np.diff(y[tr0:tr1]), ddof=1))

        fold_rec = {"fold": fi, "train": (tr0, tr1), "validation": (tr1, va1),
                    "test": (va1, te1), "diagnostics": diag.summary(),
                    "calibrated": {"q_beta": qb, "q_alpha": qa, "r": r}}

        for v in VARIANTS:
            z, beta, recs = _variant_signals(v, y, x, tr1, qb, qa, r, sigma_ref,
                                             train_start=tr0)
            # IDENTICAL gate for every variant, built from its own signal
            # stream — never a Kalman-only extra hurdle
            healthy = None if v == "cash" else health_from_signal(z)

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
            m["gated_bars"] = int(0 if healthy is None else (~healthy[va1:te1]).sum())
            ztest = z[va1:te1]
            zt = ztest[np.isfinite(ztest)]
            m["z_std_test"] = float(np.std(zt, ddof=1)) if len(zt) > 2 else None
            if recs is not None:
                m["innovations"] = innovation_diagnostics(recs[va1:te1])
            results["by_variant"][v].append(m)
            results["test_returns"][v].append(led["ret"].to_numpy(float)[1:])
        results["folds"].append(fold_rec)

    # aggregate per variant across test folds
    agg = {}
    for v in VARIANTS:
        ms = results["by_variant"][v]
        if ms:
            pooled = (np.concatenate(results["test_returns"][v])
                      if results["test_returns"][v] else np.array([]))
            lo, hi = sharpe_ci_block_bootstrap(pooled)
            zs = [m["z_std_test"] for m in ms if m.get("z_std_test")]
            agg[v] = {
                "folds": len(ms),
                "mean_test_sharpe": float(np.mean([m.get("sharpe", 0.0) for m in ms])),
                "sharpe_ci90": (round(lo, 2) if np.isfinite(lo) else None,
                                round(hi, 2) if np.isfinite(hi) else None),
                "total_test_return": float(np.prod(
                    [1 + m.get("total_return", 0.0) for m in ms]) - 1),
                "total_costs": float(np.sum([m.get("total_costs", 0.0) for m in ms])),
                "trades": int(np.sum([m.get("trades", 0) for m in ms])),
                "gated_bars": int(np.sum([m.get("gated_bars", 0) for m in ms])),
                "mean_z_std": float(np.mean(zs)) if zs else None,
            }
    results["aggregate"] = agg
    results.pop("test_returns")     # arrays served their purpose; keep result JSON-able
    return results


def format_report(res: dict, pair_name: str) -> str:
    lines = [f"=== Kalman pairs walk-forward: {pair_name} ===",
             f"folds: {len(res['folds'])}  (train/val/test chronological, test untouched)",
             ""]
    for fr in res["folds"][:1]:
        lines.append(f"fold 0 diagnostics (train only): {fr['diagnostics']}")
        lines.append(f"fold 0 calibration: {fr['calibrated']}")
        lines.append("")
    lines.append(f"{'variant':<16} {'folds':>5} {'test Sharpe':>12} {'CI90':>14} "
                 f"{'total ret':>10} {'trades':>7} {'gated':>6} {'z-std':>6} {'costs':>9}")
    for v, a in res["aggregate"].items():
        ci = a.get("sharpe_ci90", (None, None))
        ci_s = f"[{ci[0]},{ci[1]}]" if ci[0] is not None else "n/a"
        zs = a.get("mean_z_std")
        lines.append(f"{v:<16} {a['folds']:>5} {a['mean_test_sharpe']:>12.2f} {ci_s:>14} "
                     f"{a['total_test_return']:>9.1%} {a['trades']:>7} "
                     f"{a['gated_bars']:>6} {(f'{zs:.2f}' if zs else 'n/a'):>6} "
                     f"{a['total_costs']:>9.0f}")
    for v in ("kalman_fixed", "kalman_adaptive"):
        zs = res["aggregate"].get(v, {}).get("mean_z_std")
        if zs is not None and not (0.5 <= zs <= 2.0):
            lines.append(f"WARNING: {v} innovation z-std is {zs:.2f} (far from 1) — its "
                         "entry thresholds are not on the same scale as the baselines; "
                         "treat cross-variant comparison with suspicion.")
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
                led: pd.DataFrame, start: int = 0) -> pd.DataFrame:
    """Trade-level audit: prior state, innovation, signal, posterior, targets,
    fills, turnover, costs, realized P&L — one row per bar.

    `start` is the absolute bar index where the ledger's slice begins (the
    same `start` given to `backtest`). Filter records are absolute-indexed
    while ledger rows and decisions are slice-local; without the offset the
    merge silently produced rows whose filter columns came from the wrong
    bars and whose ledger columns were all NaN (review finding 6)."""
    recs = recs[start:start + len(led)]
    rows = []
    for i, rec in enumerate(recs):
        d = decisions[i] if i < len(decisions) else None
        rows.append({
            "t": i,                    # slice-local: the ledger/decision index
            "t_abs": rec.t,            # absolute filter step, for cross-reference
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
