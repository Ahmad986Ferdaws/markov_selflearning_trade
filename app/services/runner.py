"""Background polling loop for active runs.

DEPRECATED (legacy 15-second intraday loop). Superseded by the daily evaluation
path (app/services/evaluation.py + evaluate-cli). Retained only for the legacy
/runs API. Known limits: blocking yfinance I/O in async paths, in-memory run
registry that orphans runs on restart, inert request_stop cancel path. Not the
integration point for the Phase-C LLM agent. See docs/13-reframed-plan.md.
"""

from __future__ import annotations

import asyncio
import logging

from sqlalchemy.orm import Session

from app.config import Settings
from app.db import get_session_factory
from app.models.entities import Run
from app.schemas.snapshot import NormalizedSnapshot
from app.services import agent as agent_service
from app.services.baseline import baseline_decision
from app.services.ingestion import get_provider
from app.services.ledger import apply_trade, persist_snapshot
from app.services.regime import fetch_daily_history, regime_feature

logger = logging.getLogger(__name__)

_active_tasks: dict[int, asyncio.Task] = {}
_stop_flags: dict[int, bool] = {}


def _select_trade_snapshot(
    snapshots: list[NormalizedSnapshot],
    trade_symbol: str,
) -> tuple[NormalizedSnapshot | None, str, float | None]:
    for snapshot in snapshots:
        if snapshot.symbol == trade_symbol:
            return snapshot, trade_symbol, snapshot.price
    if snapshots:
        fallback = snapshots[0]
        return fallback, fallback.symbol, fallback.price
    return None, trade_symbol, None


def request_stop(run_id: int) -> None:
    _stop_flags[run_id] = True
    task = _active_tasks.get(run_id)
    if task and not task.done():
        task.cancel()


async def run_loop(run_id: int, settings: Settings) -> None:
    factory = get_session_factory()
    benchmark_history = fetch_daily_history(settings.benchmark_symbol, years=3)

    while not _stop_flags.get(run_id, False):
        db: Session = factory()
        try:
            run = db.get(Run, run_id)
            if run is None or run.status != "running":
                break
            provider = get_provider(run.ingestion_provider)
            watchlist = [s.strip() for s in run.watchlist.split(",") if s.strip()]
            snaps = await provider.poll(watchlist)
            for snap in snaps:
                persist_snapshot(db, run_id, snap)
            trade_snapshot, trade_symbol, trade_price = _select_trade_snapshot(snaps, run.trade_symbol)
            if trade_price is None:
                await asyncio.sleep(settings.poll_interval_seconds)
                continue

            feat = regime_feature(
                benchmark_history,
                window=settings.regime_window,
                bull_thresh=settings.regime_bull_thresh,
                bear_thresh=settings.regime_bear_thresh,
                as_of=trade_snapshot.timestamp if trade_snapshot else None,
            )

            if run.strategy == "agent":
                decision = await agent_service.get_agent_decision(
                    trade_snapshot.truncated_metrics() if trade_snapshot else {},
                    feat,
                    trade_symbol,
                    settings,
                )
                from app.schemas.trade import TradeIntent

                intent = TradeIntent(
                    action=decision.action,
                    percentage=decision.percentage,
                    symbol=trade_symbol,
                    reasoning=decision.reasoning,
                )
            else:
                intent = baseline_decision(feat, trade_symbol)

            apply_trade(db, run, intent, trade_price, settings)
            db.refresh(run)
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("run_loop error run_id=%s", run_id)
        finally:
            db.close()
        await asyncio.sleep(settings.poll_interval_seconds)

    db = factory()
    try:
        run = db.get(Run, run_id)
        if run:
            run.status = "stopped"
            db.commit()
    finally:
        db.close()
    _active_tasks.pop(run_id, None)
    _stop_flags.pop(run_id, None)


def start_run_task(run_id: int, settings: Settings) -> asyncio.Task:
    _stop_flags[run_id] = False
    task = asyncio.create_task(run_loop(run_id, settings))
    _active_tasks[run_id] = task
    return task
