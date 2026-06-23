"""Markov regime model — Phase 0 standalone module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import yfinance as yf

StateLabel = Literal["bull", "bear", "sideways"]
STATE_ORDER: tuple[StateLabel, ...] = ("bull", "bear", "sideways")
STATE_TO_IDX = {s: i for i, s in enumerate(STATE_ORDER)}


@dataclass
class RegimeFeature:
    state: StateLabel
    p_next: dict[str, float]


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    states: pd.Series
    annual_return: float
    annual_sharpe: float
    max_drawdown: float
    regime_mix: dict[str, float]
    sparse_warnings: list[str]


def label_state(mean_val: float, std_val: float, k: float = 0.5) -> str | None:
    """Single source of truth for the bull/bear/sideways rule (z-score based).

    Causal by construction: caller passes a TRAILING-window mean and std, so the
    label at time t uses only data up to t (no lookahead). Returns None during
    the warmup window where mean/std are undefined.
    """
    if mean_val is None or std_val is None or np.isnan(mean_val) or np.isnan(std_val):
        return None
    if std_val <= 0:
        return "sideways"
    if mean_val > k * std_val:
        return "bull"
    if mean_val < -k * std_val:
        return "bear"
    return "sideways"


def define_states(
    returns: pd.Series,
    window: int = 20,
    k: float = 0.5,
    mode: str = "zscore",
    bull_thresh: float = 0.02,
    bear_thresh: float = -0.02,
) -> pd.Series:
    """Label each day bull / bear / sideways.

    mode="zscore" (default): rolling mean vs +/- k * rolling std — adapts to the
    asset's own volatility so the model actually changes state (fixed +/-2%/day
    thresholds left BTC in 'sideways' ~99% of the time). mode="absolute" keeps
    the old fixed-threshold behavior for backward compatibility.
    """
    rolling = returns.rolling(window=window, min_periods=window).mean()
    if mode == "zscore":
        roll_std = returns.rolling(window=window, min_periods=window).std()
        labels = [label_state(m, s, k) for m, s in zip(rolling, roll_std)]
        states = pd.Series(labels, index=returns.index, dtype=object)
    else:
        states = pd.Series(index=returns.index, dtype=object)
        states[rolling >= bull_thresh] = "bull"
        states[rolling <= bear_thresh] = "bear"
        states[(rolling > bear_thresh) & (rolling < bull_thresh)] = "sideways"
    return states.dropna().astype(str)


def estimate_transition_matrix(states: pd.Series) -> np.ndarray:
    """MLE transition counts; rows sum to 1."""
    n = len(STATE_ORDER)
    counts = np.zeros((n, n))
    labels = states.astype(str).tolist()
    for i in range(len(labels) - 1):
        if labels[i] not in STATE_TO_IDX or labels[i + 1] not in STATE_TO_IDX:
            continue
        counts[STATE_TO_IDX[labels[i]], STATE_TO_IDX[labels[i + 1]]] += 1
    p = np.zeros((n, n))
    for i in range(n):
        row_sum = counts[i].sum()
        if row_sum > 0:
            p[i] = counts[i] / row_sum
        else:
            p[i] = np.ones(n) / n
    return p


def forecast(p: np.ndarray, current_state: StateLabel, n_steps: int = 1) -> np.ndarray:
    """Chapman-Kolmogorov: distribution after n_steps."""
    idx = STATE_TO_IDX[current_state]
    v = np.zeros(len(STATE_ORDER))
    v[idx] = 1.0
    return v @ np.linalg.matrix_power(p, n_steps)


def stationary_distribution(p: np.ndarray) -> np.ndarray:
    """Solve pi @ P = pi."""
    n = p.shape[0]
    a = np.vstack([p.T - np.eye(n), np.ones(n)])
    b = np.zeros(n + 1)
    b[-1] = 1.0
    pi, *_ = np.linalg.lstsq(a, b, rcond=None)
    pi = np.maximum(pi, 0)
    return pi / pi.sum()


def sparse_cell_warnings(states: pd.Series, min_count: int = 20) -> list[str]:
    p = estimate_transition_matrix(states)
    n = len(STATE_ORDER)
    counts = np.zeros((n, n))
    labels = states.astype(str).tolist()
    for i in range(len(labels) - 1):
        if labels[i] in STATE_TO_IDX and labels[i + 1] in STATE_TO_IDX:
            counts[STATE_TO_IDX[labels[i]], STATE_TO_IDX[labels[i + 1]]] += 1
    warnings: list[str] = []
    for i, from_s in enumerate(STATE_ORDER):
        for j, to_s in enumerate(STATE_ORDER):
            if 0 < counts[i, j] < min_count:
                warnings.append(f"Sparse transition {from_s}->{to_s}: {int(counts[i, j])} counts (<{min_count})")
    return warnings


def regime_feature(
    history: pd.DataFrame,
    window: int = 20,
    bull_thresh: float = 0.02,
    bear_thresh: float = -0.02,
    as_of=None,
) -> RegimeFeature:
    """Latest state + one-step forecast from daily close history.

    Point-in-time: when ``as_of`` is given, only history up to and including
    that date is used, so the feature is what would have been known at ``as_of``
    (no lookahead). Without it, the latest available state is returned.
    """
    closes = history["Close"] if "Close" in history.columns else history.squeeze()
    if as_of is not None:
        as_of_date = pd.Timestamp(as_of).date()
        closes = closes[[d <= as_of_date for d in closes.index.date]]
    returns = closes.pct_change().dropna()
    states = define_states(returns, window=window, bull_thresh=bull_thresh, bear_thresh=bear_thresh)
    if len(states) < 2:
        uniform = 1.0 / 3
        return RegimeFeature(state="sideways", p_next={s: uniform for s in STATE_ORDER})
    p = estimate_transition_matrix(states)
    current = states.iloc[-1]
    if current not in STATE_TO_IDX:
        current = "sideways"
    dist = forecast(p, current, n_steps=1)
    return RegimeFeature(
        state=current,  # type: ignore[arg-type]
        p_next={STATE_ORDER[i]: float(dist[i]) for i in range(len(STATE_ORDER))},
    )


def fetch_daily_history(symbol: str, years: int = 3) -> pd.DataFrame:
    ticker = symbol.replace("-USD", "-USD") if "-" in symbol else symbol
    data = yf.download(ticker, period=f"{years}y", interval="1d", progress=False, auto_adjust=True)
    if data.empty:
        raise ValueError(f"No data returned for {symbol}")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def _position_from_state(state: str) -> float:
    """Simple regime strategy: long in bull, flat otherwise."""
    if state == "bull":
        return 1.0
    return 0.0


def walk_forward_backtest(
    history: pd.DataFrame,
    window: int = 20,
    bull_thresh: float = 0.02,
    bear_thresh: float = -0.02,
    fee_pct: float = 0.0,
    slippage_pct: float = 0.0,
    min_train: int = 60,
) -> BacktestResult:
    """Re-estimate P each step using only past data; no lookahead."""
    closes = history["Close"] if "Close" in history.columns else history.squeeze()
    returns = closes.pct_change().dropna()
    cost_rate = (fee_pct + slippage_pct) / 100.0

    equity = 1.0
    equities: list[float] = []
    idx_list: list[pd.Timestamp] = []
    state_series: list[str] = []
    strat_returns: list[float] = []
    prev_position = 0.0

    for t in range(min_train, len(returns)):
        train_returns = returns.iloc[:t]
        train_states = define_states(
            train_returns, window=window, bull_thresh=bull_thresh, bear_thresh=bear_thresh
        )
        if len(train_states) < 10:
            continue
        current_state = train_states.iloc[-1]
        position = _position_from_state(current_state)
        market_r = float(returns.iloc[t])
        turnover = abs(position - prev_position)
        cost = turnover * cost_rate
        strat_r = position * market_r - cost
        equity *= 1.0 + strat_r
        equities.append(equity)
        idx_list.append(returns.index[t])
        state_series.append(current_state)
        strat_returns.append(strat_r)
        prev_position = position

    eq = pd.Series(equities, index=idx_list)
    ret = pd.Series(strat_returns, index=idx_list)
    states = pd.Series(state_series, index=idx_list)

    if len(ret) < 2:
        return BacktestResult(
            equity_curve=eq,
            returns=ret,
            states=states,
            annual_return=0.0,
            annual_sharpe=0.0,
            max_drawdown=0.0,
            regime_mix={},
            sparse_warnings=[],
        )

    ann_factor = np.sqrt(252)
    annual_return = (eq.iloc[-1] ** (252 / len(ret)) - 1) if len(ret) > 0 else 0.0
    sharpe = float(ret.mean() / ret.std() * ann_factor) if ret.std() > 0 else 0.0
    roll_max = eq.cummax()
    dd = (eq - roll_max) / roll_max
    max_dd = float(dd.min()) if len(dd) else 0.0
    mix = states.value_counts(normalize=True).to_dict() if len(states) else {}

    all_states = define_states(returns, window=window, bull_thresh=bull_thresh, bear_thresh=bear_thresh)
    warnings = sparse_cell_warnings(all_states)

    return BacktestResult(
        equity_curve=eq,
        returns=ret,
        states=states,
        annual_return=float(annual_return),
        annual_sharpe=sharpe,
        max_drawdown=max_dd,
        regime_mix={str(k): float(v) for k, v in mix.items()},
        sparse_warnings=warnings,
    )


def run_phase0_report(
    symbol: str = "BTC-USD",
    window: int = 20,
    bull_thresh: float = 0.02,
    bear_thresh: float = -0.02,
    fee_pct: float = 0.3,
    slippage_pct: float = 0.5,
    years: int = 3,
) -> str:
    """CLI report for Phase 0 acceptance gate."""
    history = fetch_daily_history(symbol, years=years)
    returns = history["Close"].pct_change().dropna()
    states = define_states(returns, window=window, bull_thresh=bull_thresh, bear_thresh=bear_thresh)
    p = estimate_transition_matrix(states)
    pi = stationary_distribution(p)
    feat = regime_feature(history, window=window, bull_thresh=bull_thresh, bear_thresh=bear_thresh)

    bt_no_cost = walk_forward_backtest(
        history, window=window, bull_thresh=bull_thresh, bear_thresh=bear_thresh, fee_pct=0, slippage_pct=0
    )
    bt_cost = walk_forward_backtest(
        history,
        window=window,
        bull_thresh=bull_thresh,
        bear_thresh=bear_thresh,
        fee_pct=fee_pct,
        slippage_pct=slippage_pct,
    )

    lines = [
        f"=== Phase 0 Regime Report: {symbol} ===",
        "",
        "Transition matrix P (rows: from bull, bear, sideways):",
        np.array2string(p, precision=4),
        "",
        "Stationary distribution:",
        ", ".join(f"{STATE_ORDER[i]}={pi[i]:.4f}" for i in range(len(STATE_ORDER))),
        "",
        f"Current regime feature: state={feat.state}, p_next={feat.p_next}",
        "",
        "--- Backtest (walk-forward, no lookahead) ---",
        f"No cost  — Sharpe: {bt_no_cost.annual_sharpe:.3f}, Return: {bt_no_cost.annual_return:.2%}, MaxDD: {bt_no_cost.max_drawdown:.2%}",
        f"With cost — Sharpe: {bt_cost.annual_sharpe:.3f}, Return: {bt_cost.annual_return:.2%}, MaxDD: {bt_cost.max_drawdown:.2%}",
        f"Regime mix: {bt_cost.regime_mix}",
        "",
    ]
    if bt_cost.sparse_warnings:
        lines.append("Sparse cell warnings:")
        lines.extend(f"  - {w}" for w in bt_cost.sparse_warnings[:10])
    if bt_cost.annual_sharpe >= bt_no_cost.annual_sharpe and fee_pct + slippage_pct > 0:
        lines.append("NOTE: Costs should typically lower Sharpe vs no-cost run.")
    return "\n".join(lines)
