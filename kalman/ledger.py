"""Two-leg cash-and-holdings ledger — the only P&L source in this framework.

The banned shortcut `position * posterior_spread.diff() / spread_std` is not
implemented anywhere: a drifting posterior beta re-prices yesterday's spread
with today's hedge ratio, booking P&L no trader could have earned. Here P&L
exists only as (cash + shares * price) of two REAL legs, marked at closes,
with fills at the NEXT bar's open after each signal.

Execution timeline enforced structurally:
    signal from bar t  ->  target holdings queued  ->  filled at open(t+1)
so bar-t information can never touch bar-t P&L. (Tested, not just asserted.)

Costs — charged on actually TRADED notional per leg, never on signal flips:
    commission_bps + half_spread_bps + slippage_bps  per side,
    borrow_bps_pa on short-leg notional per bar (calendar-aware daily),
    financing on negative cash (rate_bps_pa) per bar.

Sizing (log-price model default): the signal implies dollar exposures
    L1 = +/- gross/2,  L2 = -sign * beta_sign * gross/2  (elasticity-normalized)
translated to SHARES at the fill price. Gross normalization documented:
|L1| + |L2| = gross_target. For the level model, beta is a share ratio and is
translated directly to shares (s2 = -beta * s1).

Hedge-management modes: FREEZE (beta fixed for the trade's life) or REHEDGE
(re-target when |beta_now - beta_at_entry| > rehedge_threshold). Both charge
full turnover costs on every share change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from .strategy import Position


class PriceModel(Enum):
    LOG = "log"      # beta is an elasticity -> dollar exposures -> shares
    LEVEL = "level"  # beta is a share ratio -> shares directly


class HedgeMode(Enum):
    FREEZE = "freeze"
    REHEDGE = "rehedge"


@dataclass
class CostConfig:
    commission_bps: float = 1.0
    half_spread_bps: float = 2.0
    slippage_bps: float = 2.0
    borrow_bps_pa: float = 25.0
    financing_bps_pa: float = 0.0
    bars_per_year: int = 252

    @property
    def per_side_bps(self) -> float:
        return self.commission_bps + self.half_spread_bps + self.slippage_bps


@dataclass
class LedgerConfig:
    capital: float = 100_000.0
    gross_target: float = 100_000.0       # |L1| + |L2| at entry
    max_gross: float = 200_000.0          # hard cap (2x capital default)
    price_model: PriceModel = PriceModel.LOG
    hedge_mode: HedgeMode = HedgeMode.FREEZE
    rehedge_threshold: float = 0.10       # |beta - beta_entry| to re-target
    costs: CostConfig = field(default_factory=CostConfig)


@dataclass
class LedgerRow:
    """One bar of the audit trail (executed state, marked at close)."""

    t: int
    obs_ts: object
    exec_ts: object | None      # when today's fills (from t-1 signal) executed
    s1: float                   # shares leg 1 AFTER today's fill
    s2: float
    cash: float
    equity: float               # cash + s1*c1 + s2*c2 (close marks)
    gross: float
    net: float
    turnover: float             # |traded notional| this bar (both legs)
    costs: float                # all costs charged this bar
    fill_reason: str


def _target_shares(pos: Position, beta: float, o1: float, o2: float,
                   cfg: LedgerConfig) -> tuple[float, float]:
    """Translate a residual position + hedge ratio into explicit shares of
    both legs at the fill prices."""
    if pos is Position.FLAT:
        return 0.0, 0.0
    sign = 1.0 if pos is Position.LONG_RESIDUAL else -1.0
    gross = min(cfg.gross_target, cfg.max_gross)
    if cfg.price_model is PriceModel.LOG:
        # elasticity: dollar-split half/half, hedge leg direction from beta sign
        l1 = sign * gross / 2.0
        l2 = -sign * np.sign(beta if beta != 0 else 1.0) * gross / 2.0
        return l1 / o1, l2 / o2
    # LEVEL: beta is a share ratio; scale so gross notional == gross target
    s1_unit, s2_unit = 1.0, -beta
    unit_gross = abs(s1_unit) * o1 + abs(s2_unit) * o2
    k = gross / unit_gross if unit_gross > 0 else 0.0
    return sign * k * s1_unit, sign * k * s2_unit


def run_ledger(index: pd.DatetimeIndex,
               open1: np.ndarray, open2: np.ndarray,
               close1: np.ndarray, close2: np.ndarray,
               decisions: list,               # strategy.Decision per bar
               betas: np.ndarray,             # posterior beta per bar (for sizing at NEXT open)
               cfg: LedgerConfig) -> pd.DataFrame:
    """Execute decisions with a strict one-bar lag and account both legs.

    decisions[t] and betas[t] are information available AFTER bar t's close;
    they are consumed at bar t+1's open — never earlier.
    """
    n = len(index)
    assert len(open1) == len(open2) == len(close1) == len(close2) == len(decisions) == len(betas) == n

    c = cfg.costs
    per_side = c.per_side_bps / 1e4
    borrow_bar = c.borrow_bps_pa / 1e4 / c.bars_per_year
    fin_bar = c.financing_bps_pa / 1e4 / c.bars_per_year

    s1 = s2 = 0.0
    cash = cfg.capital
    cur_pos = Position.FLAT
    beta_entry = 0.0
    rows: list[LedgerRow] = []

    for t in range(n):
        turnover = costs = 0.0
        exec_ts = None
        reason = ""

        # ---- execute yesterday's decision at TODAY's open ------------------
        if t > 0:
            d = decisions[t - 1]
            b_signal = float(betas[t - 1])            # info from bar t-1 only
            want = d.target
            retarget = False
            if want is not cur_pos:
                retarget = True
                beta_entry = b_signal
            elif want is not Position.FLAT and cfg.hedge_mode is HedgeMode.REHEDGE \
                    and abs(b_signal - beta_entry) > cfg.rehedge_threshold:
                retarget = True
                beta_entry = b_signal
            if retarget:
                o1, o2 = float(open1[t]), float(open2[t])
                if np.isfinite(o1) and np.isfinite(o2) and o1 > 0 and o2 > 0:
                    beta_use = beta_entry if cfg.hedge_mode is HedgeMode.FREEZE else b_signal
                    t1, t2 = _target_shares(want, beta_use, o1, o2, cfg)
                    dn1, dn2 = (t1 - s1) * o1, (t2 - s2) * o2
                    turnover = abs(dn1) + abs(dn2)
                    costs += turnover * per_side
                    cash -= dn1 + dn2 + turnover * per_side
                    s1, s2 = t1, t2
                    cur_pos = want
                    exec_ts = index[t]
                    reason = f"fill:{d.reason}"
                # else: missed fill — position unchanged, retry logic is the
                # next bar's decision; no phantom execution at a bad price.

        # ---- per-bar carrying costs (charged on today's close marks) -------
        c1, c2 = float(close1[t]), float(close2[t])
        short_notional = max(0.0, -s1 * c1) + max(0.0, -s2 * c2)
        carry = short_notional * borrow_bar + max(0.0, -cash) * fin_bar
        cash -= carry
        costs += carry

        equity = cash + s1 * c1 + s2 * c2
        gross = abs(s1 * c1) + abs(s2 * c2)
        rows.append(LedgerRow(t, index[t], exec_ts, s1, s2, cash, equity,
                              gross, s1 * c1 + s2 * c2, turnover, costs, reason))

    df = pd.DataFrame([r.__dict__ for r in rows]).set_index("obs_ts")
    df["ret"] = df["equity"].pct_change().fillna(0.0)
    return df
