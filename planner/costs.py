"""Residuals and their Jacobians: the toy TrackingProblem that exercises the solver, and RouteProblem."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from planner import bicycle
from planner.route import Centerline, project


KAPPA_MAX = bicycle.kappa_from_steering(np.pi / 3)  # [1/m] the plant's steering limit, as path curvature
SIGMA_MAX = 0.1  # [1/m^2] curvature change per metre: straight to a 7 m radius in 1.4 m
TRACK_OFFSET = 1.461  # [m] rear axle to the Pacifica's geometric center, (front_length - rear_length) / 2
SIGMA_WEIGHT = 50.0  # sigma ~ steer rate / (L v); at 8 m/s this matches P2's 0.1 weight on steer rate in rad/s


def hinge(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0.0)


def wrap_angle(a: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a), np.cos(a))


class TrackingProblem:
    """Track a time-indexed reference (x, y, psi, v)_k with the bicycle.

    Per-knot residuals (p = 11): tracking x, y, psi, v | effort jerk, sigma |
    limits a_hi, a_lo, kappa_hi, kappa_lo, v_neg. Terminal (pN = 9): tracking + limits.
    """

    n, m = bicycle.NX, bicycle.NU
    N_TRACK, N_EFFORT, N_LIMIT = 4, 2, 5  # row counts; rows are laid out in this order
    p, pN = N_TRACK + N_EFFORT + N_LIMIT, N_TRACK + N_LIMIT

    def __init__(
        self,
        ref: np.ndarray,  # (N+1, 4)
        *,
        w_pos: float = 1.0,
        w_psi: float = 1.0,
        w_v: float = 1.0,
        w_jerk: float = 0.1,
        w_sigma: float = SIGMA_WEIGHT,
        w_limit: float = 100.0,
        a_min: float = -4.05,
        a_max: float = 2.40,
        kappa_max: float = KAPPA_MAX,
        u_lb: Sequence[float] = (-4.0, -SIGMA_MAX),
        u_ub: Sequence[float] = (4.0, SIGMA_MAX),
    ) -> None:
        self.ref = np.array(ref, dtype=np.float64)  # copy: callers reuse reference buffers between ticks
        self.N = self.ref.shape[0] - 1
        self.s_track = np.sqrt(np.array([w_pos, w_pos, w_psi, w_v]))
        self.s_effort = np.sqrt(np.array([w_jerk, w_sigma]))
        self.s_limit = float(np.sqrt(w_limit))
        self.a_min, self.a_max, self.kappa_max = a_min, a_max, kappa_max
        self.u_lb = np.array(u_lb, dtype=np.float64)
        self.u_ub = np.array(u_ub, dtype=np.float64)

    step = staticmethod(bicycle.step)
    linearize = staticmethod(bicycle.linearize)

    def _limit_args(self, X: np.ndarray) -> np.ndarray:
        """(K, 6) -> (K, 5) hinge arguments z; the limit is violated where z > 0."""
        v, a, kappa = X[:, 3], X[:, 4], X[:, 5]
        return np.stack([a - self.a_max, self.a_min - a, kappa - self.kappa_max, -self.kappa_max - kappa, -v], axis=1)

    def _state_residuals(self, X: np.ndarray) -> np.ndarray:
        """(K, 6) with the matching rows of self.ref -> (K, pN): tracking then limits."""
        e = X[:, : self.N_TRACK] - self.ref[: X.shape[0]]
        e[:, 2] = wrap_angle(e[:, 2])
        return np.concatenate([self.s_track * e, self.s_limit * hinge(self._limit_args(X))], axis=1)

    def _state_jacobians(self, X: np.ndarray) -> np.ndarray:
        """(K, 6) -> (K, pN, 6). Rows and signs mirror _limit_args; columns are the states v=3, a=4, kappa=5."""
        J = np.zeros((X.shape[0], self.pN, self.n))
        i = np.arange(self.N_TRACK)
        J[:, i, i] = self.s_track  # tracked states are the first N_TRACK states, in order
        active = self.s_limit * (self._limit_args(X) > 0.0)
        t = self.N_TRACK
        J[:, t + 0, 4] = active[:, 0]
        J[:, t + 1, 4] = -active[:, 1]
        J[:, t + 2, 5] = active[:, 2]
        J[:, t + 3, 5] = -active[:, 3]
        J[:, t + 4, 3] = -active[:, 4]
        return J

    def residuals(self, X: np.ndarray, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """X (N+1, 6), U (N, 2) -> r (N, p), rN (pN,)."""
        s = self._state_residuals(X)  # (N+1, pN)
        t = self.N_TRACK
        r = np.concatenate([s[:-1, :t], self.s_effort * U, s[:-1, t:]], axis=1)
        return r, s[-1]

    def residual_jacobians(self, X: np.ndarray, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """X (N+1, 6), U (N, 2) -> Jx (N, p, 6), Ju (N, p, 2), JxN (pN, 6)."""
        Js = self._state_jacobians(X)  # (N+1, pN, 6)
        N, t = self.N, self.N_TRACK
        Jx = np.zeros((N, self.p, self.n))
        Jx[:, :t] = Js[:-1, :t]
        Jx[:, t + self.N_EFFORT :] = Js[:-1, t:]
        Ju = np.zeros((N, self.p, self.m))
        Ju[:, t + 0, 0] = self.s_effort[0]
        Ju[:, t + 1, 1] = self.s_effort[1]
        return Jx, Ju, Js[-1]


class RouteProblem:
    """Follow a route centerline at its speed reference (P3 free driving).

    Per-knot residuals (p = 14): tracking e_y, e_psi, e_v | effort jerk, sigma |
    limits a_hi, a_lo, kappa_hi, kappa_lo, v_neg, lat_hi, lat_lo, yaw_hi, yaw_lo.
    Terminal (pN = 13): tracking + limits + progress, sqrt(w) * (s(x_N) - s_target), zero when s_target is None.
    The progress row exists because tracking cost is summed over time: without it, a car stopped at a bad angle
    finds "wait, then go" cheaper than going now, replans the same every tick, and never moves (P3, tight turns).
    Tracking and progress are measured at a point track_offset ahead of the rear axle (default: the vehicle's
    geometric center). With the rear axle on a 4 m radius centerline, the front bumper swings about 1.8 m outside it.
    Knots project onto a fixed window of centerline segments, so residuals are a pure function of (X, U).
    """

    n, m = bicycle.NX, bicycle.NU
    N_TRACK, N_EFFORT, N_LIMIT = 3, 2, 9  # row counts; rows are laid out in this order
    p, pN = N_TRACK + N_EFFORT + N_LIMIT, N_TRACK + N_LIMIT + 1

    def __init__(
        self,
        centerline: Centerline,
        seg_window: Tuple[int, int],
        N: int = 40,
        *,
        s_target: Optional[float] = None,
        w_progress: float = 0.1,
        track_offset: float = TRACK_OFFSET,
        w_ey: float = 1.0,
        w_psi: float = 1.0,
        w_v: float = 0.5,
        w_jerk: float = 0.1,
        w_sigma: float = SIGMA_WEIGHT,
        w_limit: float = 100.0,
        a_min: float = -4.05,
        a_max: float = 2.40,
        kappa_max: float = KAPPA_MAX,
        a_lat_max: float = 4.0,
        yaw_max: float = 0.8,
        u_lb: Sequence[float] = (-4.0, -SIGMA_MAX),
        u_ub: Sequence[float] = (4.0, SIGMA_MAX),
    ) -> None:
        self.cl, (self.lo, self.hi), self.N = centerline, seg_window, N
        self.s_target, self.track_offset = s_target, track_offset
        self.s_progress = 0.0 if s_target is None else float(np.sqrt(w_progress))
        self.s_track = np.sqrt(np.array([w_ey, w_psi, w_v]))
        self.s_effort = np.sqrt(np.array([w_jerk, w_sigma]))
        self.s_limit = float(np.sqrt(w_limit))
        self.a_min, self.a_max, self.kappa_max = a_min, a_max, kappa_max
        self.a_lat_max, self.yaw_max = a_lat_max, yaw_max
        self.u_lb = np.array(u_lb, dtype=np.float64)
        self.u_ub = np.array(u_ub, dtype=np.float64)

    step = staticmethod(bicycle.step)
    linearize = staticmethod(bicycle.linearize)

    def _point(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(K, 6) -> the tracked point (K, 2), track_offset ahead of the rear axle, and its derivative w.r.t. psi."""
        d = np.column_stack([np.cos(X[:, 2]), np.sin(X[:, 2])])
        return X[:, :2] + self.track_offset * d, self.track_offset * np.column_stack([-d[:, 1], d[:, 0]])

    def _tracking(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(K, 6) -> e (K, 3) = (e_y, e_psi, e_v), its Jacobian w.r.t. the tracked point (K, 3, 2), and that point's
        derivative w.r.t. psi (K, 2)."""
        cl = self.cl
        P, dP_dpsi = self._point(X)
        seg, f = project(cl, self.lo, self.hi, P, X[:, 2])
        t, h = cl.seg_t[seg], cl.seg_len[seg]
        n = np.column_stack([-t[:, 1], t[:, 0]])
        e_y = np.einsum("ki,ki->k", P - cl.xy[seg], n)
        dpsi, dv = cl.psi[seg + 1] - cl.psi[seg], cl.v_ref[seg + 1] - cl.v_ref[seg]
        e_psi = wrap_angle(X[:, 2] - (cl.psi[seg] + f * dpsi))
        e_v = X[:, 3] - (cl.v_ref[seg] + f * dv)
        inside = ((f > 0.0) & (f < 1.0)).astype(np.float64)  # a clipped fraction does not move with p
        df_dp = (inside / h)[:, None] * t
        J = np.stack([n, -dpsi[:, None] * df_dp, -dv[:, None] * df_dp], axis=1)
        return np.column_stack([e_y, e_psi, e_v]), J, dP_dpsi

    def _limit_args(self, X: np.ndarray) -> np.ndarray:
        """(K, 6) -> (K, 9) hinge arguments z; the limit is violated where z > 0."""
        v, a, kappa = X[:, 3], X[:, 4], X[:, 5]
        yaw = v * kappa
        lat = v * yaw
        return np.stack(
            [
                a - self.a_max, self.a_min - a, kappa - self.kappa_max, -self.kappa_max - kappa, -v,
                lat - self.a_lat_max, -self.a_lat_max - lat, yaw - self.yaw_max, -self.yaw_max - yaw,
            ],
            axis=1,
        )

    def _state_residuals(self, X: np.ndarray) -> np.ndarray:
        """(K, 6) -> (K, 12): tracking then limits."""
        e, _, _ = self._tracking(X)
        return np.concatenate([self.s_track * e, self.s_limit * hinge(self._limit_args(X))], axis=1)

    def _state_jacobians(self, X: np.ndarray) -> np.ndarray:
        """(K, 6) -> (K, 12, 6): tracking then limits. Rows mirror _state_residuals and _limit_args."""
        K = X.shape[0]
        _, Jxy, dP_dpsi = self._tracking(X)
        J = np.zeros((K, self.N_TRACK + self.N_LIMIT, self.n))  # the progress row is added in residual_jacobians
        J[:, :3, :2] = self.s_track[None, :, None] * Jxy
        J[:, :3, 2] = self.s_track * np.einsum("kri,ki->kr", Jxy, dP_dpsi)  # psi also moves the tracked point
        J[:, 1, 2] += self.s_track[1]  # e_psi = psi - psi_c
        J[:, 2, 3] = self.s_track[2]  # e_v = v - v_ref
        v, kappa = X[:, 3], X[:, 5]
        dyaw = np.stack([kappa, v], axis=1)  # d yaw / d (v, kappa)
        dlat = np.stack([2 * v * kappa, v**2], axis=1)
        act = self.s_limit * (self._limit_args(X) > 0.0)
        t = self.N_TRACK
        J[:, t + 0, 4] = act[:, 0]
        J[:, t + 1, 4] = -act[:, 1]
        J[:, t + 2, 5] = act[:, 2]
        J[:, t + 3, 5] = -act[:, 3]
        J[:, t + 4, 3] = -act[:, 4]
        J[:, t + 5][:, [3, 5]] = act[:, 5:6] * dlat
        J[:, t + 6][:, [3, 5]] = -act[:, 6:7] * dlat
        J[:, t + 7][:, [3, 5]] = act[:, 7:8] * dyaw
        J[:, t + 8][:, [3, 5]] = -act[:, 8:9] * dyaw
        return J

    def _progress(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """Terminal state (6,) -> progress residual and its Jacobian (6,)."""
        if self.s_target is None:
            return 0.0, np.zeros(self.n)
        P, dP_dpsi = self._point(x[None])
        seg, f = project(self.cl, self.lo, self.hi, P, x[2:3])
        i, f = int(seg[0]), float(f[0])
        ds = self.cl.s[i + 1] - self.cl.s[i]
        J = np.zeros(self.n)
        if 0.0 < f < 1.0:  # a clipped fraction does not move with position
            J[:2] = self.s_progress * ds / self.cl.seg_len[i] * self.cl.seg_t[i]
            J[2] = J[:2] @ dP_dpsi[0]
        return self.s_progress * (self.cl.s[i] + f * ds - self.s_target), J

    def residuals(self, X: np.ndarray, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """X (N+1, 6), U (N, 2) -> r (N, p), rN (pN,)."""
        s = self._state_residuals(X)
        t = self.N_TRACK
        r = np.concatenate([s[:-1, :t], self.s_effort * U, s[:-1, t:]], axis=1)
        return r, np.append(s[-1], self._progress(X[-1])[0])

    def residual_jacobians(self, X: np.ndarray, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """X (N+1, 6), U (N, 2) -> Jx (N, p, 6), Ju (N, p, 2), JxN (pN, 6)."""
        Js = self._state_jacobians(X)
        N, t = self.N, self.N_TRACK
        Jx = np.zeros((N, self.p, self.n))
        Jx[:, :t] = Js[:-1, :t]
        Jx[:, t + self.N_EFFORT :] = Js[:-1, t:]
        Ju = np.zeros((N, self.p, self.m))
        Ju[:, t + 0, 0] = self.s_effort[0]
        Ju[:, t + 1, 1] = self.s_effort[1]
        return Jx, Ju, np.vstack([Js[-1], self._progress(X[-1])[1]])
