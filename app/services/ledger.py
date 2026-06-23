"""Paper ledger — simulated fills."""

from __future__ import annotations

import json

from sqlalchemy.orm import Session

from app.config import Settings
from app.models.entities import Run, Snapshot, Trade
from app.schemas.snapshot import NormalizedSnapshot
from app.schemas.trade import TradeIntent
from app.services.guards import EnforcedDecision, effective_price, enforce_decision


def persist_snapshot(db: Session, run_id: int, snap: NormalizedSnapshot) -> Snapshot:
    row = Snapshot(
        run_id=run_id,
        symbol=snap.symbol,
        price=snap.price,
        volume=snap.volume,
        source=snap.source,
        payload_json=json.dumps(snap.truncated_metrics()),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def apply_trade(
    db: Session,
    run: Run,
    intent: TradeIntent,
    price: float,
    settings: Settings,
    liquidity_usd: float | None = None,
) -> Trade | None:
    enforced = enforce_decision(
        intent,
        cash=run.cash,
        position_qty=run.position_qty,
        price=price,
        liquidity_usd=liquidity_usd,
        settings=settings,
    )
    if enforced.action == "HOLD":
        if enforced.guard_note or intent.action != "HOLD":
            trade = Trade(
                run_id=run.id,
                symbol=intent.symbol,
                intent_action=intent.action,
                intent_percentage=intent.percentage,
                enforced_action="HOLD",
                enforced_percentage=0.0,
                quantity=0.0,
                execution_price=price,
                fees_paid=0.0,
                guard_note=enforced.guard_note,
            )
            db.add(trade)
            db.commit()
            return trade
        return None

    fees_total = 0.0
    qty = 0.0
    exec_price = price

    if enforced.action == "BUY":
        notional = run.cash * (enforced.percentage / 100.0)
        exec_price, fee_per = effective_price(price, "BUY", settings)
        qty = notional / exec_price
        fees_total = notional - qty * price
        cost = qty * exec_price
        if cost > run.cash:
            cost = run.cash
            qty = cost / exec_price
        if run.position_qty + qty > 0:
            run.avg_entry_price = (
                run.avg_entry_price * run.position_qty + exec_price * qty
            ) / (run.position_qty + qty)
        run.cash -= cost
        run.position_qty += qty

    elif enforced.action == "SELL":
        sell_qty = run.position_qty * (enforced.percentage / 100.0)
        exec_price, _ = effective_price(price, "SELL", settings)
        proceeds = sell_qty * exec_price
        fees_total = sell_qty * price - proceeds
        cost_basis = sell_qty * run.avg_entry_price
        run.realized_pnl += proceeds - cost_basis
        run.cash += proceeds
        run.position_qty -= sell_qty
        qty = sell_qty
        if run.position_qty <= 1e-12:
            run.position_qty = 0.0
            run.avg_entry_price = 0.0

    trade = Trade(
        run_id=run.id,
        symbol=intent.symbol,
        intent_action=intent.action,
        intent_percentage=intent.percentage,
        enforced_action=enforced.action,
        enforced_percentage=enforced.percentage,
        quantity=qty,
        execution_price=exec_price,
        fees_paid=abs(fees_total),
        guard_note=enforced.guard_note,
    )
    db.add(trade)
    db.commit()
    db.refresh(run)
    return trade


def mark_to_market(run: Run, price: float) -> float:
    if run.position_qty <= 0:
        return 0.0
    return (price - run.avg_entry_price) * run.position_qty


def portfolio_value(run: Run, price: float) -> float:
    return run.cash + run.position_qty * price
