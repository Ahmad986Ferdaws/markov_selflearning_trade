"""Phase A — daily walk-forward evaluation: regime accuracy + trading comparison.

Everything here is DAILY and CAUSAL. The signal clock equals the trade clock
(one decision per bar), so the comparison actually measures something. No
lookahead: at decision time t we use only returns up to and including t.

Two distinct questions are answered separately (R0):
  1. Prediction accuracy — does the Markov model predict tomorrow's regime?
  2. Trading edge — does acting on it beat buy-and-hold, net of costs?
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.services.data_cache import history_hash
from app.services.regime import STATE_ORDER, STATE_TO_IDX, estimate_transition_matrix, label_state

ANN = math.sqrt(252)
_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Causal states
# --------------------------------------------------------------------------- #
def causal_states(returns: pd.Series, window: int, k: float) -> list[str | None]:
    """Per-day state using only a trailing window (no lookahead). None in warmup."""
    mean = returns.rolling(window=window, min_periods=window).mean()
    std = returns.rolling(window=window, min_periods=window).std()
    return [label_state(m, s, k) for m, s in zip(mean, std)]


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class AccuracyMetrics:
    hit_rate: float
    balanced_accuracy: float
    log_loss: float
    naive_hit_rate: float
    naive_balanced: float  # 1/(classes present): a majority predictor's true balanced acc
    naive_log_loss: float
    persistence_hit_rate: float  # "tomorrow's regime = today's" — the honest bar
    persistence_balanced: float
    n: int
    # Switch-day anatomy: the ONLY days a predictor can beat persistence are the
    # days the regime actually changes (persistence scores 0 on them by
    # construction). These three numbers say where the accuracy really lives.
    n_switch: int = 0        # held-out days where tomorrow != today
    switch_recall: float = 0.0   # model accuracy restricted to those days
    switch_attempts: int = 0     # days the model predicted ANY change at all


@dataclass
class PolicyResult:
    name: str
    total_return: float
    sharpe: float
    max_drawdown: float
    num_trades: int
    total_cost: float


@dataclass
class DailyReport:
    symbol: str
    chosen_window: int
    chosen_k: float
    train_size: int
    test_size: int
    regime_mix: dict
    accuracy: AccuracyMetrics
    policies: list[PolicyResult]
    train_best_score: float
    data_hash: str = ""
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Walk-forward core
# --------------------------------------------------------------------------- #
@dataclass
class PolicyContext:
    """What a policy sees at decision time t (only past/present info)."""
    state: str               # today's regime
    prev_pos: float          # yesterday's position (0..1, long/flat)
    p_next: np.ndarray       # model's forecast distribution over next state
    day_index: int


# A policy maps a context to a target position in [0, 1] (long/flat, v1).
Policy = "callable[[PolicyContext], float]"


def policy_buy_hold(ctx: PolicyContext) -> float:
    return 1.0


def policy_regime_baseline(ctx: PolicyContext) -> float:
    # bull -> long, bear -> flat, sideways -> hold previous
    if ctx.state == "bull":
        return 1.0
    if ctx.state == "bear":
        return 0.0
    return ctx.prev_pos


# The one place policies are registered. Phase C adds the LLM agent here; it then
# flows through the exact same fill/metrics path as every other policy.
DEFAULT_POLICIES: dict = {
    "baseline": policy_regime_baseline,
    "buy_hold": policy_buy_hold,
}


def _walk_forward(
    returns: pd.Series,
    states: list[str | None],
    start: int,
    end: int,
    fee_pct: float,
    slippage_pct: float,
    policies: dict | None = None,
) -> dict:
    """Walk t in [start, end): predict state_{t+1}, trade into r_{t+1}.

    Every policy is a pluggable function evaluated through the SAME loop and the
    SAME downstream metrics (`_policy_metrics`) — one engine, no duplicated fill
    math. P is estimated only from states up to and including the current day t,
    so there is no lookahead.
    """
    policies = policies if policies is not None else DEFAULT_POLICIES
    cost_rate = (fee_pct + slippage_pct) / 100.0

    pred_probs: list[np.ndarray] = []
    actual_idx: list[int] = []
    cur_idx: list[int] = []  # today's state, for the persistence baseline

    positions: dict = {name: [] for name in policies}
    prev_pos: dict = {name: 0.0 for name in policies}
    market_rets: list[float] = []

    r = returns.to_numpy()
    n = len(r)

    for t in range(start, min(end, n - 1)):
        cur = states[t]
        nxt = states[t + 1]
        if cur is None or nxt is None or cur not in STATE_TO_IDX or nxt not in STATE_TO_IDX:
            continue

        # --- prediction (uses only states up to and including t) ---
        observed = [s for s in states[: t + 1] if s in STATE_TO_IDX]
        if len(observed) < 2:
            continue
        p = estimate_transition_matrix(pd.Series(observed))
        dist = p[STATE_TO_IDX[cur]]
        pred_probs.append(dist)
        actual_idx.append(STATE_TO_IDX[nxt])
        cur_idx.append(STATE_TO_IDX[cur])

        # --- positions, decided at t from state_t (one engine, all policies) ---
        for name, fn in policies.items():
            ctx = PolicyContext(state=cur, prev_pos=prev_pos[name], p_next=dist, day_index=t)
            pos = float(fn(ctx))
            positions[name].append(pos)
            prev_pos[name] = pos

        market_rets.append(float(r[t + 1]))

    return {
        "pred_probs": pred_probs,
        "actual_idx": actual_idx,
        "cur_idx": cur_idx,
        "positions": positions,
        "market_rets": market_rets,
        "cost_rate": cost_rate,
        "states_traded": [states[t] for t in range(start, min(end, n - 1))],
    }


def _accuracy(pred_probs, actual_idx, train_freq, cur_idx=None) -> AccuracyMetrics:
    if not pred_probs:
        return AccuracyMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    n_states = len(STATE_ORDER)
    cur_idx = cur_idx if cur_idx is not None else [None] * len(actual_idx)
    hits = 0
    ll = 0.0
    naive_hits = 0
    naive_ll = 0.0
    naive_pred = int(np.argmax(train_freq))
    per_total = np.zeros(n_states)
    per_correct = np.zeros(n_states)
    persist_correct = np.zeros(n_states)
    persist_hits = 0
    n_switch = 0
    switch_hits = 0
    switch_attempts = 0
    for dist, actual, cur in zip(pred_probs, actual_idx, cur_idx):
        pred = int(np.argmax(dist))
        per_total[actual] += 1
        if pred == actual:
            hits += 1
            per_correct[actual] += 1
        ll += -math.log(max(float(dist[actual]), _EPS))
        if naive_pred == actual:
            naive_hits += 1
        naive_ll += -math.log(max(float(train_freq[actual]), _EPS))
        if cur is not None and cur == actual:  # persistence: predict today's state
            persist_hits += 1
            persist_correct[actual] += 1
        if cur is not None:
            if actual != cur:               # a real regime change — the contested days
                n_switch += 1
                if pred == actual:
                    switch_hits += 1
            if pred != cur:                 # the model dared to predict a change
                switch_attempts += 1
    n = len(actual_idx)
    # balanced accuracy = mean per-class recall over classes that actually occur.
    # Predicting only the majority class scores ~1/n_states here, so this metric
    # refuses to reward the degenerate "always sideways" model.
    recalls = [per_correct[c] / per_total[c] for c in range(n_states) if per_total[c] > 0]
    balanced = float(np.mean(recalls)) if recalls else 0.0
    p_recalls = [persist_correct[c] / per_total[c] for c in range(n_states) if per_total[c] > 0]
    persist_balanced = float(np.mean(p_recalls)) if p_recalls else 0.0
    # a majority-class predictor gets recall 1.0 on one class, 0.0 on the rest that
    # occur -> balanced acc = 1/(# classes present), not a fixed 1/3.
    naive_balanced = 1.0 / len(recalls) if recalls else 0.0
    return AccuracyMetrics(
        hit_rate=hits / n,
        balanced_accuracy=balanced,
        log_loss=ll / n,
        naive_hit_rate=naive_hits / n,
        naive_balanced=naive_balanced,
        naive_log_loss=naive_ll / n,
        persistence_hit_rate=persist_hits / n,
        persistence_balanced=persist_balanced,
        n=n,
        n_switch=n_switch,
        switch_recall=(switch_hits / n_switch) if n_switch else 0.0,
        switch_attempts=switch_attempts,
    )


def _policy_metrics(name: str, positions: list[float], market_rets: list[float], cost_rate: float) -> PolicyResult:
    if not positions:
        return PolicyResult(name, 0.0, 0.0, 0.0, 0, 0.0)
    equity = 1.0
    eq = []
    strat_rets = []
    total_cost = 0.0
    trades = 0
    prev = 0.0
    for pos, mret in zip(positions, market_rets):
        turnover = abs(pos - prev)
        cost = turnover * cost_rate
        if turnover > 0:
            trades += 1
        total_cost += cost
        sr = pos * mret - cost
        equity *= 1.0 + sr
        eq.append(equity)
        strat_rets.append(sr)
        prev = pos
    eq_s = pd.Series(eq)
    rets = pd.Series(strat_rets)
    total_return = eq_s.iloc[-1] - 1.0
    sharpe = float(rets.mean() / rets.std() * ANN) if rets.std() > 0 else 0.0
    roll_max = eq_s.cummax()
    dd = ((eq_s - roll_max) / roll_max).replace([np.inf, -np.inf], np.nan).dropna()
    max_dd = float(dd.min()) if len(dd) else 0.0
    return PolicyResult(name, float(total_return), sharpe, max_dd, trades, float(total_cost))


def _train_freq(states: list[str | None], start: int, end: int) -> np.ndarray:
    counts = np.zeros(len(STATE_ORDER))
    for s in states[start:end]:
        if s in STATE_TO_IDX:
            counts[STATE_TO_IDX[s]] += 1
    total = counts.sum()
    return counts / total if total > 0 else np.ones(len(STATE_ORDER)) / len(STATE_ORDER)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def evaluate(
    history: pd.DataFrame,
    symbol: str = "BTC-USD",
    train_frac: float = 0.7,
    grid_windows: tuple[int, ...] = (10, 20, 30),
    grid_k: tuple[float, ...] = (0.3, 0.5, 0.75),
    fee_pct: float = 0.3,
    slippage_pct: float = 0.5,
    min_train: int = 60,
    policies: dict | None = None,
) -> DailyReport:
    """Daily walk-forward eval with a real train/test split (R1, R3-R6).

    Hyperparameters (window, k) are selected on the TRAIN region by prediction
    hit-rate; all reported accuracy and trading metrics are on the held-out TEST
    region the selection never touched. `policies` defaults to baseline +
    buy_hold; Phase C injects the LLM agent here without touching the engine.
    """
    policies = policies if policies is not None else DEFAULT_POLICIES
    closes = history["Close"] if "Close" in history.columns else history.squeeze()
    returns = closes.pct_change().dropna()
    n = len(returns)
    if n < min_train + 30:
        raise ValueError(f"Not enough history: {n} returns")

    split = int(n * train_frac)
    if split <= min_train + 1:
        raise ValueError(
            f"Train region too small: split={split} <= min_train={min_train}. "
            "Increase train_frac or use a longer history."
        )
    if split >= n - 1:
        raise ValueError(f"Test region empty: split={split} >= n-1={n - 1}. Lower train_frac.")
    warnings: list[str] = []

    # --- 1. select (window, k) on TRAIN only ---
    # Two traps to avoid:
    #  (a) raw hit-rate rewards always-predict-majority -> never trades;
    #  (b) balanced accuracy is trivially 1.0 if a config collapses everything to
    #      ONE state (only one class to "balance").
    # So: only configs that produce >=2 well-represented regimes (each >=10% of
    # train days) may compete, ranked by balanced accuracy. If none qualify, the
    # asset is effectively single-regime at these thresholds — an honest finding —
    # and we fall back to the least-collapsed config with a loud warning.
    candidates = []
    for w in grid_windows:
        for k in grid_k:
            states = causal_states(returns, w, k)
            # split-1 so the last predicted target (states[t+1]) stays strictly
            # inside the train region — no test-region label nudges selection.
            wf = _walk_forward(returns, states, min_train, split - 1, fee_pct, slippage_pct)
            tf = _train_freq(states, min_train, split)
            acc = _accuracy(wf["pred_probs"], wf["actual_idx"], tf)
            if acc.n == 0:
                continue
            n_meaningful = int((tf >= 0.10).sum())
            candidates.append(
                {"w": w, "k": k, "score": acc.balanced_accuracy, "max_share": float(tf.max()),
                 "n_meaningful": n_meaningful}
            )

    if not candidates:
        raise ValueError("No valid (window, k) configuration produced predictions on train")

    diverse = [c for c in candidates if c["n_meaningful"] >= 2]
    if diverse:
        best = max(diverse, key=lambda c: c["score"])
    else:
        best = min(candidates, key=lambda c: c["max_share"])
        warnings.append(
            "No (window, k) produced >=2 well-represented regimes on train: this asset is "
            "effectively single-regime at these thresholds. Reporting the least-collapsed config; "
            "treat the regime signal as weak/uninformative (a valid finding)."
        )

    w, k = best["w"], best["k"]
    train_best_score = best["score"]

    # --- 2. report on held-out TEST with the chosen config ---
    states = causal_states(returns, w, k)
    wf = _walk_forward(returns, states, split, n, fee_pct, slippage_pct, policies=policies)
    train_freq = _train_freq(states, min_train, split)
    accuracy = _accuracy(wf["pred_probs"], wf["actual_idx"], train_freq, wf["cur_idx"])

    policy_results = [
        _policy_metrics(name, wf["positions"][name], wf["market_rets"], wf["cost_rate"])
        for name in wf["positions"]
    ]

    # regime mix on test
    traded = [s for s in wf["states_traded"] if s in STATE_TO_IDX]
    mix = {}
    if traded:
        vc = pd.Series(traded).value_counts(normalize=True)
        mix = {str(kk): float(v) for kk, v in vc.items()}

    # --- 3. validation / honesty gates ---
    if mix and max(mix.values()) > 0.80:
        dom = max(mix, key=mix.get)
        warnings.append(f"Regime mix imbalanced: '{dom}' is {mix[dom]:.0%} of test days (>80%).")
    if accuracy.n and accuracy.hit_rate <= accuracy.naive_hit_rate:
        warnings.append(
            f"Model does NOT beat the naive (majority) predictor on test "
            f"(model {accuracy.hit_rate:.1%} vs naive {accuracy.naive_hit_rate:.1%}). "
            "This is a valid, honest finding."
        )
    if accuracy.n and accuracy.balanced_accuracy <= accuracy.persistence_balanced + 0.02:
        warnings.append(
            f"Markov model barely beats persistence ('tomorrow = today'): "
            f"balanced-acc {accuracy.balanced_accuracy:.1%} vs persistence "
            f"{accuracy.persistence_balanced:.1%}. The transition matrix is adding little "
            "beyond regime autocorrelation — high accuracy here is mostly persistence, not skill."
        )
    if train_best_score - accuracy.balanced_accuracy > 0.10:
        warnings.append(
            f"Possible overfit: train balanced-acc {train_best_score:.1%} vs test "
            f"{accuracy.balanced_accuracy:.1%} (gap > 10pp)."
        )
    if len({s for s in traded}) <= 1:
        warnings.append("Constant-regime test window: comparison not meaningful.")

    # Gate 7 — all accuracy is stay-day credit. The only days any predictor can
    # beat persistence are the days the regime actually changes; a model whose
    # switch recall is zero has, by construction, no skill over persistence no
    # matter how high its hit-rate reads — whether it never tried, or tried and
    # missed every time. Say so in plain language.
    if accuracy.n_switch > 0 and accuracy.switch_recall == 0.0:
        if accuracy.switch_attempts == 0:
            warnings.append(
                f"The model never predicted a regime change (0 switch attempts over "
                f"{accuracy.n} days): its accuracy is entirely stay-day credit, and it scored "
                f"0/{accuracy.n_switch} on the days the regime actually moved. A hit-rate "
                "earned only on stay days is persistence, restated."
            )
        else:
            warnings.append(
                f"The model predicted a change on {accuracy.switch_attempts} day(s) and got "
                f"0 of {accuracy.n_switch} actual changes right: all its accuracy is "
                "stay-day credit. A hit-rate earned only on stay days is persistence, restated."
            )

    # Gate 6 — the agent (if present) acted on low-confidence forecasts.
    # p_next is handed to every policy precisely so a confidence-aware agent
    # could flag uncertainty; the rule-based policies ignore it. When the
    # model's own forecast is near-uniform, the most-likely next regime is barely
    # better than a coin-flip, so any agent decision rests on noise. Like every
    # other gate this only APPENDS a caveat — it never overrides the numeric row
    # and never auto-routes to a human.
    if "agent" in policies and wf["pred_probs"]:
        n_states = len(STATE_ORDER)
        conf_floor = 1.0 / n_states + 0.15  # "barely favors any state" threshold
        low_conf = sum(1 for d in wf["pred_probs"] if float(np.max(d)) < conf_floor)
        frac = low_conf / len(wf["pred_probs"])
        if frac >= 0.25:
            warnings.append(
                f"Agent acted on low-confidence forecasts on {frac:.0%} of test days "
                f"(most-likely next regime below {conf_floor:.0%}): treat the agent row "
                "as weak/uninformative on those days (a valid, honest finding)."
            )

    return DailyReport(
        symbol=symbol,
        chosen_window=w,
        chosen_k=k,
        train_size=split - min_train,  # train predictions actually walked
        test_size=accuracy.n,          # test predictions actually evaluated
        regime_mix=mix,
        accuracy=accuracy,
        policies=policy_results,
        train_best_score=train_best_score,
        data_hash=history_hash(history),
        warnings=warnings,
    )


def format_daily_report(r: DailyReport) -> str:
    lines = [
        f"=== Daily Evaluation: {r.symbol} ===",
        f"Train days: {r.train_size}  |  Test days: {r.test_size}  |  "
        f"Chosen window={r.chosen_window}, k={r.chosen_k}",
        f"Data hash: {r.data_hash[:12]} (reproducible)" if r.data_hash else "",
        "",
        "--- Regime mix (test) ---",
        "  " + ", ".join(f"{s}={p:.1%}" for s, p in r.regime_mix.items()) or "  (none)",
        "",
        "--- Prediction accuracy (held-out test) ---",
        f"  Model        hit-rate: {r.accuracy.hit_rate:.1%}   balanced-acc: {r.accuracy.balanced_accuracy:.1%}   log-loss: {r.accuracy.log_loss:.3f}",
        f"  Naive(major) hit-rate: {r.accuracy.naive_hit_rate:.1%}   balanced-acc: {r.accuracy.naive_balanced:.1%}   log-loss: {r.accuracy.naive_log_loss:.3f}",
        f"  Persistence  hit-rate: {r.accuracy.persistence_hit_rate:.1%}   balanced-acc: {r.accuracy.persistence_balanced:.1%}   (tomorrow = today)",
        f"  (n={r.accuracy.n}, train best balanced-acc={r.train_best_score:.1%})",
        "",
        "--- Switch days (the only days skill could exist) ---",
        f"  regime changes: {r.accuracy.n_switch}/{r.accuracy.n} days   "
        f"model recall on them: "
        + (f"{r.accuracy.switch_recall:.1%}" if r.accuracy.n_switch else "n/a")
        + f"   model switch attempts: {r.accuracy.switch_attempts}   "
        f"(persistence scores 0% here by construction)",
        "",
        "--- Trading (test, net of costs) ---",
        f"  {'Policy':<12} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>7} {'Cost':>8}",
    ]
    for p in r.policies:
        lines.append(
            f"  {p.name:<12} {p.total_return:>9.2%} {p.sharpe:>8.2f} "
            f"{p.max_drawdown:>7.2%} {p.num_trades:>7} {p.total_cost:>7.2%}"
        )
    if r.warnings:
        lines.append("")
        lines.append("WARNINGS:")
        lines.extend(f"  - {w}" for w in r.warnings)
    return "\n".join(lines)
