"""Kalman filter cores — implemented directly so the timing stays auditable.

Two models:

  * PairFilter — dynamic hedge ratio for pairs trading.
        state x_t = [beta_t, alpha_t]', random walk (F = I)
        obs   y_t = p1_t,  H_t = [p2_t, 1],  v_t ~ N(0, R)
    The TRADING SIGNAL is the standardized PRE-update innovation
    z_t = e_t / sqrt(S_t) with e_t = y_t - H_t x_{t|t-1}. The posterior state
    (which has already seen y_t) is never used to form the same bar's signal.

  * TrendFilter — local linear trend (level + velocity), dt-aware:
        F = [[1, dt], [0, 1]],  H = [1, 0]
        Q(dt) uses the continuous-time integrated form so irregular intervals
        scale correctly instead of an arbitrary fixed diagonal.

Numerical contract (enforced, not assumed):
  * Joseph-form covariance update, then symmetrization.
  * No explicit matrix inversion anywhere (S_t is scalar in both models).
  * Every S_t is checked > 0; covariances checked symmetric PSD.
  * NaN / inf never propagate silently: a missing observation performs a
    predict-only step and is flagged; a non-finite input raises.
  * beta and alpha (and level/velocity) keep SEPARATE process variances —
    they have different units; Q = qI is not assumed.

Initialization: 'diffuse' (zero state, large prior covariance) or 'ols'
(training-window OLS for the pair model). Both are expected to be paired with
a no-trade warm-up enforced by the strategy layer.

Adaptive Q: Q_t = Q_base * clip((sigma_{t-1}/sigma_ref)^gamma, m_min, m_max)
where sigma is LAGGED realized volatility of the observed series and
sigma_ref comes only from data available at construction (training window).
No centered windows, no backfill.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

_EPS = 1e-12


def _assert_finite(name: str, *vals: float) -> None:
    for v in vals:
        if not np.all(np.isfinite(v)):
            raise ValueError(f"non-finite value reached the filter: {name}")


def _symmetrize(P: np.ndarray) -> np.ndarray:
    P = 0.5 * (P + P.T)
    # PSD guard: clip tiny negative eigenvalues from floating-point drift.
    w, V = np.linalg.eigh(P)
    if w.min() < -1e-9 * max(1.0, abs(w.max())):
        raise ValueError(f"covariance lost positive semidefiniteness: min eig {w.min():.3e}")
    if w.min() < 0:
        P = V @ np.diag(np.clip(w, 0.0, None)) @ V.T
        P = 0.5 * (P + P.T)
    return P


@dataclass
class StepRecord:
    """Everything the spec requires us to retain per timestamp, clearly labeled."""

    t: int
    observed: bool                 # False -> predict-only (missing y)
    x_prior: np.ndarray            # x_{t|t-1}
    P_prior: np.ndarray            # P_{t|t-1}
    innovation: float | None       # e_t = y_t - H_t x_{t|t-1}
    innovation_var: float | None   # S_t
    z: float | None                # standardized innovation e_t / sqrt(S_t)
    gain: np.ndarray | None        # K_t
    x_post: np.ndarray             # x_{t|t}   (== prior when not observed)
    P_post: np.ndarray             # P_{t|t}
    loglik: float | None           # predictive log-likelihood of y_t
    nis: float | None              # normalized innovation squared e_t^2 / S_t

    def state_interval(self, i: int, n_sd: float = 2.0) -> tuple[float, float]:
        sd = math.sqrt(max(self.P_post[i, i], 0.0))
        return (float(self.x_post[i] - n_sd * sd), float(self.x_post[i] + n_sd * sd))


@dataclass
class AdaptiveQ:
    """Causal multiplicative Q scaling from LAGGED realized volatility."""

    sigma_ref: float               # reference vol — training data only
    gamma: float = 1.0
    m_min: float = 0.25
    m_max: float = 4.0
    window: int = 20

    def multiplier(self, past_returns: np.ndarray) -> float:
        """past_returns must END at t-1 (caller guarantees the lag)."""
        if len(past_returns) < self.window or self.sigma_ref <= 0:
            return 1.0
        sig = float(np.std(past_returns[-self.window:], ddof=1))
        if not np.isfinite(sig) or sig <= 0:
            return 1.0
        return float(np.clip((sig / self.sigma_ref) ** self.gamma, self.m_min, self.m_max))


class PairFilter:
    """Dynamic-hedge Kalman filter for one pair. Scalar-observation model, so
    every 'inversion' is a scalar division — nothing to invert."""

    def __init__(
        self,
        q_beta: float,
        q_alpha: float,
        r: float,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
        adaptive: AdaptiveQ | None = None,
    ):
        if q_beta <= 0 or q_alpha <= 0 or r <= 0:
            raise ValueError("q_beta, q_alpha, r must be positive")
        self.Q = np.diag([q_beta, q_alpha]).astype(float)   # separate units
        self.R = float(r)
        self.x = (np.zeros(2) if x0 is None else np.asarray(x0, float).copy())
        self.P = (np.diag([10.0, 10.0]) if P0 is None else np.asarray(P0, float).copy())
        self.adaptive = adaptive
        self._t = -1

    @classmethod
    def diffuse(cls, q_beta: float, q_alpha: float, r: float,
                prior_scale: float = 10.0, adaptive: AdaptiveQ | None = None) -> "PairFilter":
        return cls(q_beta, q_alpha, r, x0=np.zeros(2),
                   P0=np.diag([prior_scale, prior_scale]), adaptive=adaptive)

    @classmethod
    def from_ols(cls, y_train: np.ndarray, p2_train: np.ndarray,
                 q_beta: float, q_alpha: float, r: float,
                 adaptive: AdaptiveQ | None = None) -> "PairFilter":
        """Training-only OLS initialization (state AND a data-scaled prior P)."""
        A = np.column_stack([p2_train, np.ones(len(p2_train))])
        coef, res, *_ = np.linalg.lstsq(A, y_train, rcond=None)
        resid_var = float(res[0] / max(len(y_train) - 2, 1)) if len(res) else float(
            np.var(y_train - A @ coef, ddof=2))
        XtX_inv = np.linalg.inv(A.T @ A)          # 2x2, training-only, one-off
        P0 = _symmetrize(resid_var * XtX_inv * 4.0)  # inflated: it's a prior, not truth
        return cls(q_beta, q_alpha, r, x0=coef, P0=P0, adaptive=adaptive)

    def step(self, y: float | None, p2: float,
             past_returns: np.ndarray | None = None) -> StepRecord:
        """One predict(+update) step. `y=None` -> missing observation
        (predict-only). `past_returns` must end at t-1 (lagged) when adaptive
        Q is enabled."""
        self._t += 1
        _assert_finite("p2", p2)

        Qt = self.Q
        if self.adaptive is not None:
            Qt = self.Q * self.adaptive.multiplier(
                np.asarray([] if past_returns is None else past_returns, float))

        # ---- predict (F = I) ----
        x_prior = self.x.copy()
        P_prior = _symmetrize(self.P + Qt)

        if y is None:
            self.x, self.P = x_prior, P_prior
            return StepRecord(self._t, False, x_prior, P_prior, None, None, None,
                              None, self.x.copy(), self.P.copy(), None, None)

        _assert_finite("y", y)
        H = np.array([p2, 1.0])

        # ---- innovation (PRE-update: this is the signal) ----
        e = float(y - H @ x_prior)
        S = float(H @ P_prior @ H + self.R)
        if not (np.isfinite(S) and S > 0):
            raise ValueError(f"innovation variance not positive: S={S}")
        z = e / math.sqrt(S)

        # ---- update: Joseph form ----
        K = (P_prior @ H) / S                      # scalar S -> no inversion
        self.x = x_prior + K * e
        IKH = np.eye(2) - np.outer(K, H)
        self.P = _symmetrize(IKH @ P_prior @ IKH.T + np.outer(K, K) * self.R)

        ll = -0.5 * (math.log(2.0 * math.pi * S) + e * e / S)
        return StepRecord(self._t, True, x_prior, P_prior, e, S, z,
                          K.copy(), self.x.copy(), self.P.copy(), ll, e * e / S)


class TrendFilter:
    """Local linear trend: state [level, velocity], dt-aware process noise.

    Q(dt) uses the integrated continuous-time form for a white-noise
    acceleration of intensity q_v plus independent level noise q_l:
        Q = [[q_l*dt + q_v*dt^3/3,  q_v*dt^2/2],
             [q_v*dt^2/2,           q_v*dt   ]]
    which reduces to the familiar discrete Q at dt=1 and scales correctly for
    irregular intervals — not an arbitrary fixed diagonal.
    """

    def __init__(self, q_level: float, q_vel: float, r: float,
                 x0: np.ndarray | None = None, P0: np.ndarray | None = None,
                 adaptive: AdaptiveQ | None = None):
        if q_level <= 0 or q_vel <= 0 or r <= 0:
            raise ValueError("q_level, q_vel, r must be positive")
        self.ql, self.qv, self.R = float(q_level), float(q_vel), float(r)
        self.x = (np.zeros(2) if x0 is None else np.asarray(x0, float).copy())
        self.P = (np.diag([10.0, 1.0]) if P0 is None else np.asarray(P0, float).copy())
        self.adaptive = adaptive
        self._t = -1

    def _Q(self, dt: float, mult: float) -> np.ndarray:
        ql, qv = self.ql * mult, self.qv * mult
        return np.array([
            [ql * dt + qv * dt ** 3 / 3.0, qv * dt ** 2 / 2.0],
            [qv * dt ** 2 / 2.0,           qv * dt],
        ])

    def step(self, y: float | None, dt: float = 1.0,
             past_returns: np.ndarray | None = None) -> StepRecord:
        self._t += 1
        if dt <= 0:
            raise ValueError("dt must be positive")
        mult = 1.0
        if self.adaptive is not None:
            mult = self.adaptive.multiplier(
                np.asarray([] if past_returns is None else past_returns, float))

        F = np.array([[1.0, dt], [0.0, 1.0]])
        x_prior = F @ self.x
        P_prior = _symmetrize(F @ self.P @ F.T + self._Q(dt, mult))

        if y is None:
            self.x, self.P = x_prior, P_prior
            return StepRecord(self._t, False, x_prior, P_prior, None, None, None,
                              None, self.x.copy(), self.P.copy(), None, None)

        _assert_finite("y", y)
        H = np.array([1.0, 0.0])
        e = float(y - H @ x_prior)
        S = float(H @ P_prior @ H + self.R)
        if not (np.isfinite(S) and S > 0):
            raise ValueError(f"innovation variance not positive: S={S}")
        z = e / math.sqrt(S)
        K = (P_prior @ H) / S
        self.x = x_prior + K * e
        IKH = np.eye(2) - np.outer(K, H)
        self.P = _symmetrize(IKH @ P_prior @ IKH.T + np.outer(K, K) * self.R)
        ll = -0.5 * (math.log(2.0 * math.pi * S) + e * e / S)
        return StepRecord(self._t, True, x_prior, P_prior, e, S, z,
                          K.copy(), self.x.copy(), self.P.copy(), ll, e * e / S)


def run_pair_filter(y: np.ndarray, p2: np.ndarray, f: PairFilter) -> list[StepRecord]:
    """Batch convenience wrapper — literally the replay loop, so batch and
    one-observation-at-a-time processing agree by construction (and by test)."""
    if len(y) != len(p2):
        raise ValueError("y and p2 must align")
    rets = np.diff(np.asarray(y, float), prepend=np.asarray(y, float)[0])
    out = []
    for t in range(len(y)):
        yt = None if (y[t] is None or not np.isfinite(y[t])) else float(y[t])
        out.append(f.step(yt, float(p2[t]), past_returns=rets[:t]))  # ends at t-1
    return out
