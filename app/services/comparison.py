"""Head-to-head baseline vs agent on identical snapshots.

DEPRECATED (legacy intraday path). Superseded by app/services/evaluation.py, which
is daily, leakage-checked, single-engine, and reproducible. Known issues retained
here only for the legacy /runs API: it re-implements the fill math (diverges from
ledger.apply_trade) and calls the LLM live during "replay" (non-reproducible).
DO NOT wire the Phase-C agent into this module — use evaluation.py's pluggable
policies instead. See docs/13-reframed-plan.md and docs/15-phase-a-results.md.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.config import Settings
from app.models.entities import Run, Snapshot
from app.schemas.trade import TradeIntent
from app.services import agent as agent_service
from app.services.baseline import baseline_decision
from app.services.guards import enforce_decision
from app.services.regime import fetch_daily_history, regime_feature


SECONDS_PER_YEAR = 365 * 24 * 60 * 60


@dataclass
class StrategyMetrics:
    name: str
    total_return: float
    annual_sharpe: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    total_costs: float
    guard_interventions: int
    num_closes: int = 0
    regimes_seen: int = 0


@dataclass
class ComparisonReport:
    baseline: StrategyMetrics
    agent: StrategyMetrics
    headline_agent_minus_baseline: float
    snapshot_count: int
    sharpe_periods_per_year: float = 0.0
    warnings: list[str] = field(default_factory=list)


def _annualization_factor(settings: Settings) -> tuple[float, float]:
    """(periods_per_year, sqrt(periods_per_year)) derived from poll cadence.

    Snapshot returns are sampled every ``poll_interval_seconds``; annualizing
    with sqrt(252) (a *daily*-bar factor) is wrong, so derive it from the real
    cadence instead.
    """
    interval = max(1, settings.poll_interval_seconds)
    periods = SECONDS_PER_YEAR / interval
    return periods, float(np.sqrt(periods))


def _build_regime_lookup(
    snapshots: list[Snapshot],
    benchmark_history: pd.DataFrame,
    settings: Settings,
) -> dict[object, "object"]:
    """Precompute one point-in-time regime feature per unique snapshot date.

    O(unique_days) instead of O(snapshots), and makes the replay point-in-time:
    each snapshot sees the regime as of its own timestamp, not the latest.
    """
    lookup: dict[object, object] = {}
    for snap_row in snapshots:
        key = snap_row.created_at.date()
        if key in lookup:
            continue
        lookup[key] = regime_feature(
            benchmark_history,
            window=settings.regime_window,
            bull_thresh=settings.regime_bull_thresh,
            bear_thresh=settings.regime_bear_thresh,
            as_of=snap_row.created_at,
        )
    return lookup


def _simulate_strategy(
    name: str,
    snapshots: list[Snapshot],
    settings: Settings,
    use_agent: bool,
    regime_lookup: dict,
) -> StrategyMetrics:
    cash = settings.starting_cash
    qty = 0.0
    avg_entry = 0.0
    realized = 0.0
    equities: list[float] = []
    returns: list[float] = []
    wins = 0
    trades = 0
    closes = 0
    costs = 0.0
    interventions = 0
    prev_equity = cash
    states_seen: set[str] = set()
    _, ann_factor = _annualization_factor(settings)

    for snap_row in snapshots:
        metrics = _snapshot_metrics(snap_row)
        feat = regime_lookup[snap_row.created_at.date()]
        states_seen.add(feat.state)
        if use_agent:
            decision = agent_service.get_agent_decision_sync(
                metrics, feat, snap_row.symbol, settings
            )
            intent = TradeIntent(
                action=decision.action,
                percentage=decision.percentage,
                symbol=snap_row.symbol,
                reasoning=decision.reasoning,
            )
        else:
            intent = baseline_decision(feat, snap_row.symbol)

        enforced = enforce_decision(intent, cash, qty, snap_row.price, None, settings)
        if enforced.guard_note or (intent.action != enforced.action):
            interventions += 1

        price = snap_row.price
        if enforced.action == "BUY" and enforced.percentage > 0 and cash > 0:
            notional = cash * (enforced.percentage / 100.0)
            fee_rate = (settings.fee_pct + settings.slippage_pct) / 100.0
            exec_p = price * (1 + fee_rate)
            buy_qty = notional / exec_p
            fee = notional - buy_qty * price
            if qty + buy_qty > 0:
                avg_entry = (avg_entry * qty + exec_p * buy_qty) / (qty + buy_qty)
            cash -= buy_qty * exec_p
            qty += buy_qty
            costs += abs(fee)
            trades += 1
        elif enforced.action == "SELL" and enforced.percentage > 0 and qty > 0:
            sell_qty = qty * (enforced.percentage / 100.0)
            fee_rate = (settings.fee_pct + settings.slippage_pct) / 100.0
            exec_p = price * (1 - fee_rate)
            proceeds = sell_qty * exec_p
            fee = sell_qty * price - proceeds
            realized += proceeds - sell_qty * avg_entry
            cash += proceeds
            qty -= sell_qty
            costs += abs(fee)
            trades += 1
            closes += 1
            if proceeds > sell_qty * avg_entry:
                wins += 1

        equity = cash + qty * price
        if prev_equity > 0:
            r = equity / prev_equity - 1
            returns.append(r)
        equities.append(equity)
        prev_equity = equity

    ret_series = pd.Series(returns)
    eq = pd.Series(equities)
    total_return = (eq.iloc[-1] / settings.starting_cash - 1) if len(eq) else 0.0
    sharpe = (
        float(ret_series.mean() / ret_series.std() * ann_factor)
        if len(ret_series) > 1 and ret_series.std() > 0
        else 0.0
    )
    if len(eq):
        roll_max = eq.cummax()
        dd_series = ((eq - roll_max) / roll_max).replace([np.inf, -np.inf], np.nan).dropna()
        dd = float(dd_series.min()) if len(dd_series) else 0.0
    else:
        dd = 0.0
    win_rate = wins / closes if closes else 0.0

    return StrategyMetrics(
        name=name,
        total_return=float(total_return),
        annual_sharpe=sharpe,
        max_drawdown=float(dd),
        win_rate=float(win_rate),
        num_trades=trades,
        total_costs=float(costs),
        guard_interventions=interventions,
        num_closes=closes,
        regimes_seen=len(states_seen),
    )


def _snapshot_metrics(snapshot: Snapshot) -> dict:
    if snapshot.payload_json:
        try:
            metrics = json.loads(snapshot.payload_json)
            if isinstance(metrics, dict):
                return metrics
        except json.JSONDecodeError:
            pass
    return {
        "symbol": snapshot.symbol,
        "price": snapshot.price,
        "volume": snapshot.volume,
        "timestamp": snapshot.created_at.isoformat(),
    }


def _trade_snapshots_for_run(run: Run, snapshots: list[Snapshot]) -> list[Snapshot]:
    matching = [snapshot for snapshot in snapshots if snapshot.symbol == run.trade_symbol]
    return matching or snapshots


def run_comparison(db: Session, run_id: int, settings: Settings) -> ComparisonReport:
    run = db.get(Run, run_id)
    if run is None:
        raise ValueError(f"Run {run_id} not found")
    snapshots = (
        db.query(Snapshot).filter(Snapshot.run_id == run_id).order_by(Snapshot.created_at).all()
    )
    if not snapshots:
        empty = StrategyMetrics("baseline", 0, 0, 0, 0, 0, 0, 0)
        return ComparisonReport(
            empty,
            StrategyMetrics("agent", 0, 0, 0, 0, 0, 0, 0),
            0.0,
            0,
            warnings=["No snapshots for run; nothing to compare."],
        )

    trade_snapshots = _trade_snapshots_for_run(run, snapshots)
    history = fetch_daily_history(settings.benchmark_symbol, years=3)
    regime_lookup = _build_regime_lookup(trade_snapshots, history, settings)
    baseline = _simulate_strategy("baseline", trade_snapshots, settings, False, regime_lookup)
    agent_m = _simulate_strategy("agent", trade_snapshots, settings, True, regime_lookup)
    periods_per_year, _ = _annualization_factor(settings)
    report = ComparisonReport(
        baseline=baseline,
        agent=agent_m,
        headline_agent_minus_baseline=agent_m.total_return - baseline.total_return,
        snapshot_count=len(trade_snapshots),
        sharpe_periods_per_year=periods_per_year,
    )
    report.warnings = _validate_report(report)
    return report


def _validate_report(report: ComparisonReport) -> list[str]:
    """Fail loud on internally inconsistent / non-meaningful comparisons."""
    warnings: list[str] = []
    seen = max(report.baseline.regimes_seen, report.agent.regimes_seen)
    if seen <= 1:
        warnings.append(
            f"Constant-regime run: only {seen} distinct regime across the replay — "
            "baseline is effectively a fixed decision; the comparison is not meaningful. "
            "Replay snapshots spanning multiple benchmark days."
        )
    for m in (report.baseline, report.agent):
        for fieldname in ("total_return", "annual_sharpe", "max_drawdown", "win_rate"):
            val = getattr(m, fieldname)
            if not math.isfinite(val):
                warnings.append(f"{m.name}.{fieldname} is non-finite ({val}).")
        if report.snapshot_count > 1 and report.sharpe_periods_per_year <= 0:
            warnings.append("Sharpe annualization factor unresolved (check poll_interval_seconds).")
    return warnings


def format_report(report: ComparisonReport) -> str:
    b, a = report.baseline, report.agent
    sharpe_label = (
        f"Sharpe (ann@{report.sharpe_periods_per_year:,.0f}/yr)"
        if report.sharpe_periods_per_year
        else "Sharpe"
    )
    lines = [
        "=== Comparison Report (identical snapshots) ===",
        f"Snapshots: {report.snapshot_count}  |  Distinct regimes: "
        f"baseline={b.regimes_seen} agent={a.regimes_seen}",
        "",
        f"{'Metric':<26} {'Baseline':>12} {'Agent':>12}",
        f"{'Total return':<26} {b.total_return:>11.2%} {a.total_return:>11.2%}",
        f"{sharpe_label:<26} {b.annual_sharpe:>12.3f} {a.annual_sharpe:>12.3f}",
        f"{'Max drawdown':<26} {b.max_drawdown:>11.2%} {a.max_drawdown:>11.2%}",
        f"{'Win rate (per close)':<26} {b.win_rate:>11.2%} {a.win_rate:>11.2%}",
        f"{'Trades':<26} {b.num_trades:>12} {a.num_trades:>12}",
        f"{'Closes':<26} {b.num_closes:>12} {a.num_closes:>12}",
        f"{'Total costs':<26} {b.total_costs:>12.2f} {a.total_costs:>12.2f}",
        f"{'Guard interventions':<26} {b.guard_interventions:>12} {a.guard_interventions:>12}",
        "",
        f"Headline (agent - baseline return): {report.headline_agent_minus_baseline:+.2%}",
    ]
    if report.warnings:
        lines.append("")
        lines.append("⚠ VALIDATION WARNINGS:")
        lines.extend(f"  - {w}" for w in report.warnings)
    return "\n".join(lines)


