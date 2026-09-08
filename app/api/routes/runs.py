from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.config import Settings
from app.dependencies import get_db_dep, get_settings_dep
from app.exceptions import ApiError
from app.models.entities import Run, Trade
from app.schemas.runs import CreateRunRequest, RunStatusResponse, TradeRecordResponse
from app.services.comparison import format_report, run_comparison
from app.services.ledger import mark_to_market
from app.services.runner import request_stop, start_run_task

router = APIRouter(prefix="/runs", tags=["runs"])


@router.post("", response_model=RunStatusResponse)
async def create_run(
    body: CreateRunRequest,
    db: Session = Depends(get_db_dep),
    settings: Settings = Depends(get_settings_dep),
) -> RunStatusResponse:
    if body.ingestion_provider == "dexscreener":
        raise ApiError("dexscreener provider deferred until after Phase 1 gate", 400)
    if body.strategy == "agent" and not settings.allow_legacy_agent_api:
        raise ApiError(
            "The LLM agent is CLI-only and cannot be triggered through the web "
            "API (it would spend the server's API key with no budget ceiling). "
            "Run it via evaluate-cli, or deliberately set "
            "ALLOW_LEGACY_AGENT_API=true for local experimentation.",
            400,
        )
    trade_symbol = body.trade_symbol or (body.watchlist[0] if body.watchlist else settings.benchmark_symbol)
    if body.trade_symbol and body.watchlist and body.trade_symbol not in body.watchlist:
        raise ApiError(
            f"trade_symbol {body.trade_symbol!r} is not in the watchlist — the run "
            "would trade a symbol it never polls (and previously fell back to "
            "MIXING symbols into one equity curve).",
            422,
        )
    run = Run(
        strategy=body.strategy,
        ingestion_provider=body.ingestion_provider,
        status="running",
        watchlist=",".join(body.watchlist),
        trade_symbol=trade_symbol,
        cash=settings.starting_cash,
        starting_cash=settings.starting_cash,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    # start_run_task (not BackgroundTasks): registers the asyncio task so
    # request_stop's cancel works, resets any sticky stop flag from a previous
    # run with the same id, and frees this request instead of pinning it on an
    # infinite loop (review findings 5+6).
    start_run_task(run.id, settings)
    return _run_to_response(run, 0.0, settings)


@router.get("/{run_id}", response_model=RunStatusResponse)
def get_run(
    run_id: int,
    db: Session = Depends(get_db_dep),
    settings: Settings = Depends(get_settings_dep),
) -> RunStatusResponse:
    run = db.get(Run, run_id)
    if not run:
        raise ApiError(f"Run {run_id} not found", 404)
    last_price = _last_price(db, run)
    unrealized = mark_to_market(run, last_price) if last_price else 0.0
    return _run_to_response(run, unrealized, settings)


@router.get("/{run_id}/trades", response_model=list[TradeRecordResponse])
def list_trades(run_id: int, db: Session = Depends(get_db_dep)) -> list[TradeRecordResponse]:
    run = db.get(Run, run_id)
    if not run:
        raise ApiError(f"Run {run_id} not found", 404)
    trades = db.query(Trade).filter(Trade.run_id == run_id).order_by(Trade.created_at).all()
    return [
        TradeRecordResponse(
            id=t.id,
            symbol=t.symbol,
            intent_action=t.intent_action,
            intent_percentage=t.intent_percentage,
            enforced_action=t.enforced_action,
            enforced_percentage=t.enforced_percentage,
            quantity=t.quantity,
            execution_price=t.execution_price,
            fees_paid=t.fees_paid,
            guard_note=t.guard_note,
            created_at=t.created_at.isoformat(),
        )
        for t in trades
    ]


@router.post("/{run_id}/stop", response_model=RunStatusResponse)
def stop_run(
    run_id: int,
    db: Session = Depends(get_db_dep),
    settings: Settings = Depends(get_settings_dep),
) -> RunStatusResponse:
    run = db.get(Run, run_id)
    if not run:
        raise ApiError(f"Run {run_id} not found", 404)
    request_stop(run_id)
    run.status = "stopped"
    db.commit()
    db.refresh(run)
    last_price = _last_price(db, run)
    unrealized = mark_to_market(run, last_price) if last_price else 0.0
    return _run_to_response(run, unrealized, settings)


@router.get("/{run_id}/comparison")
def get_comparison(
    run_id: int,
    db: Session = Depends(get_db_dep),
    settings: Settings = Depends(get_settings_dep),
) -> dict:
    report = run_comparison(db, run_id, settings)
    return {
        "report_text": format_report(report),
        "headline_agent_minus_baseline": report.headline_agent_minus_baseline,
        "baseline": report.baseline.__dict__,
        "agent": report.agent.__dict__,
        "snapshot_count": report.snapshot_count,
        "sharpe_periods_per_year": report.sharpe_periods_per_year,
        "warnings": report.warnings,
    }


def _last_price(db: Session, run: Run) -> float:
    from app.models.entities import Snapshot

    snap = (
        db.query(Snapshot)
        .filter(Snapshot.run_id == run.id, Snapshot.symbol == run.trade_symbol)
        .order_by(Snapshot.created_at.desc())
        .first()
    )
    if snap:
        return snap.price
    snap = db.query(Snapshot).filter(Snapshot.run_id == run.id).order_by(Snapshot.created_at.desc()).first()
    return snap.price if snap else 0.0


def _run_to_response(run: Run, unrealized: float, settings: Settings) -> RunStatusResponse:
    return RunStatusResponse(
        id=run.id,
        strategy=run.strategy,
        ingestion_provider=run.ingestion_provider,
        status=run.status,
        watchlist=[s.strip() for s in run.watchlist.split(",") if s.strip()],
        trade_symbol=run.trade_symbol,
        cash=run.cash,
        position_qty=run.position_qty,
        unrealized_pnl=unrealized,
        realized_pnl=run.realized_pnl,
    )
