"""Hardcoded risk guards — agent cannot override."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.schemas.trade import TradeIntent


@dataclass
class EnforcedDecision:
    action: str
    percentage: float
    guard_note: str | None = None


def enforce_decision(
    intent: TradeIntent,
    cash: float,
    position_qty: float,
    price: float,
    liquidity_usd: float | None,
    settings: Settings,
) -> EnforcedDecision:
    """Clamp or reject intent per docs/03-risk-guards.md."""
    action = intent.action
    pct = intent.percentage
    notes: list[str] = []

    if action == "HOLD" or pct <= 0:
        return EnforcedDecision(action="HOLD", percentage=0.0, guard_note=None)

    if liquidity_usd is not None and liquidity_usd < settings.liquidity_floor_usd:
        return EnforcedDecision(
            action="HOLD",
            percentage=0.0,
            guard_note=f"liquidity {liquidity_usd:.0f} < floor {settings.liquidity_floor_usd:.0f}",
        )

    if action == "BUY":
        # Allocation cap applies to BUY only: it limits cash committed per order.
        # SELL percentage is "% of existing position to exit" and must be able to
        # reach 100% so positions can fully close.
        max_pct = settings.max_allocation_pct
        if pct > max_pct:
            notes.append(f"allocation clamped {pct:.1f}% -> {max_pct:.1f}%")
            pct = max_pct
        notional = cash * (pct / 100.0)
        if notional <= 0 or cash <= 0:
            return EnforcedDecision(
                action="HOLD",
                percentage=0.0,
                guard_note="reject BUY: insufficient cash",
            )
    elif action == "SELL":
        if position_qty <= 0:
            return EnforcedDecision(
                action="HOLD",
                percentage=0.0,
                guard_note="reject SELL: no position (no shorts in v1)",
            )
        # percentage of position to sell
        sell_qty = position_qty * (pct / 100.0)
        if sell_qty <= 0:
            return EnforcedDecision(action="HOLD", percentage=0.0, guard_note="reject SELL: zero size")

    note = "; ".join(notes) if notes else None
    return EnforcedDecision(action=action, percentage=pct, guard_note=note)


def effective_price(price: float, side: str, settings: Settings, liquidity_usd: float | None = None) -> tuple[float, float]:
    """Return (execution_price, fees_paid_per_unit_basis) with fee + slippage."""
    fee_rate = settings.fee_pct / 100.0
    slip_rate = settings.slippage_pct / 100.0
    if side == "BUY":
        exec_price = price * (1 + fee_rate + slip_rate)
    else:
        exec_price = price * (1 - fee_rate - slip_rate)
    fees = abs(price - exec_price)
    return exec_price, fees
