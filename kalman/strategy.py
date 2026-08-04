"""Trading state machine: hysteretic z-score rules with hard safety gates.

Timing contract (the single most important property):

    observation_timestamp  — bar t close, when y_t is observed
    signal_timestamp       — same bar t, AFTER close: z_t from the PRE-update
                             innovation of bar t
    order_timestamp        — bar t (order queued after close)
    execution_timestamp    — bar t+1 open (default) — the first permitted
                             execution event after the signal

Same-close execution is prohibited (no close-auction model is implemented,
so it is not offered). The ledger enforces the one-bar lag independently, so
a bug here cannot silently leak same-bar fills.

States: FLAT -> LONG_RESIDUAL (z <= -entry) or SHORT_RESIDUAL (z >= +entry);
exit on |z| <= exit (hysteresis: exit < entry), emergency stop on |z| >= stop,
max holding period, cooldown after any exit, and a no-trade warm-up.

Gates (any -> forced flat / no entry):
  * warm-up not elapsed
  * stale data (missing observation this bar)
  * invalid variance (S_t not finite/positive — surfaced by the filter)
  * extreme beta circuit breaker (|beta| outside bounds)
  * relationship health (caller-supplied boolean from health monitoring)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Position(Enum):
    FLAT = 0
    LONG_RESIDUAL = 1    # long y-leg, short hedge-leg
    SHORT_RESIDUAL = -1  # short y-leg, long hedge-leg


@dataclass
class StrategyConfig:
    entry_z: float = 2.0
    exit_z: float = 0.5              # hysteresis: exit_z < entry_z
    stop_z: float = 4.0              # emergency stop
    warmup: int = 60                 # no-trade warm-up observations
    max_holding: int = 60            # bars; 0 disables
    cooldown: int = 5                # bars after any exit
    beta_min: float = -5.0           # extreme-beta circuit breaker
    beta_max: float = 5.0

    def __post_init__(self) -> None:
        if not (0 <= self.exit_z < self.entry_z <= self.stop_z):
            raise ValueError("need 0 <= exit_z < entry_z <= stop_z")


@dataclass
class Decision:
    t: int
    target: Position
    reason: str
    gated: bool = False


@dataclass
class StateMachine:
    cfg: StrategyConfig
    pos: Position = Position.FLAT
    bars_held: int = 0
    cooldown_left: int = 0
    _seen: int = 0

    def decide(self, t: int, z: float | None, beta: float,
               healthy: bool = True) -> Decision:
        """One decision from bar t's signal. Returns the TARGET position that
        may be executed at the next execution event only."""
        self._seen += 1

        # ---- gates ---------------------------------------------------------
        if self._seen <= self.cfg.warmup:
            return self._flatten(t, "gate:warmup")
        if z is None:                                   # stale / missing bar
            return self._flatten(t, "gate:stale-data")
        if not (self.cfg.beta_min <= beta <= self.cfg.beta_max):
            return self._flatten(t, "gate:extreme-beta")
        if not healthy:
            return self._flatten(t, "gate:health")

        if self.cooldown_left > 0:
            self.cooldown_left -= 1
            return Decision(t, self.pos, "cooldown", gated=True) \
                if self.pos is Position.FLAT else self._exit(t, "cooldown-exit")

        # ---- in a position: exits first ------------------------------------
        if self.pos is not Position.FLAT:
            self.bars_held += 1
            if abs(z) >= self.cfg.stop_z:
                return self._exit(t, "emergency-stop")
            if self.cfg.max_holding and self.bars_held >= self.cfg.max_holding:
                return self._exit(t, "max-holding")
            if abs(z) <= self.cfg.exit_z:
                return self._exit(t, "exit-band")
            return Decision(t, self.pos, "hold")

        # ---- flat: entries --------------------------------------------------
        if z <= -self.cfg.entry_z:
            self.pos, self.bars_held = Position.LONG_RESIDUAL, 0
            return Decision(t, self.pos, "enter-long-residual")
        if z >= self.cfg.entry_z:
            self.pos, self.bars_held = Position.SHORT_RESIDUAL, 0
            return Decision(t, self.pos, "enter-short-residual")
        return Decision(t, Position.FLAT, "no-signal")

    def _exit(self, t: int, reason: str) -> Decision:
        self.pos, self.bars_held = Position.FLAT, 0
        self.cooldown_left = self.cfg.cooldown
        return Decision(t, Position.FLAT, reason)

    def _flatten(self, t: int, reason: str) -> Decision:
        if self.pos is not Position.FLAT:
            self.pos, self.bars_held = Position.FLAT, 0
            self.cooldown_left = self.cfg.cooldown
        return Decision(t, Position.FLAT, reason, gated=True)
