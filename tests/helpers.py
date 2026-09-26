"""Shared test fixtures: an LQR problem with a known answer, and the toy arc scenario."""
from __future__ import annotations

import math

import numpy as np

from planner import bicycle
from planner.costs import TrackingProblem


class LQRProblem:
    """Linear dynamics, quadratic cost written as residuals r = [Qs x; Rs u], terminal Qs x. No bounds."""

    def __init__(self, A, B, Qs, Rs, N):
        self.A, self.B, self.Qs, self.Rs, self.N = A, B, Qs, Rs, N
        self.n, self.m = B.shape
        self.u_lb = np.full(self.m, -np.inf)
        self.u_ub = np.full(self.m, np.inf)

    def step(self, x, u):
        return self.A @ x + self.B @ u

    def linearize(self, X, U):
        return np.tile(self.A, (self.N, 1, 1)), np.tile(self.B, (self.N, 1, 1))

    def residuals(self, X, U):
        return np.concatenate([X[:-1] @ self.Qs.T, U @ self.Rs.T], axis=1), self.Qs @ X[-1]

    def residual_jacobians(self, X, U):
        n, m = self.n, self.m
        Jx = np.zeros((self.N, n + m, n))
        Jx[:, :n] = self.Qs
        Ju = np.zeros((self.N, n + m, m))
        Ju[:, n:] = self.Rs
        return Jx, Ju, self.Qs


def random_lqr(seed: int, n: int = 4, m: int = 2, N: int = 20):
    rng = np.random.default_rng(seed)
    A = np.eye(n) + 0.1 * rng.standard_normal((n, n))
    B = rng.standard_normal((n, m))
    Qs = np.diag(rng.uniform(0.5, 2.0, n))
    Rs = np.diag(rng.uniform(0.5, 2.0, m))
    return LQRProblem(A, B, Qs, Rs, N), rng.standard_normal(n)


def riccati(problem: LQRProblem, x0: np.ndarray):
    """Finite-horizon discrete Riccati recursion. Returns optimal cost and the optimal control sequence."""
    A, B = problem.A, problem.B
    Q, R = problem.Qs.T @ problem.Qs, problem.Rs.T @ problem.Rs
    P = Q.copy()
    gains = []
    for _ in range(problem.N):
        K = -np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        P = Q + A.T @ P @ (A + B @ K)
        gains.append(K)
    gains.reverse()
    x, U = x0.copy(), []
    for K in gains:
        U.append(K @ x)
        x = A @ x + B @ U[-1]
    return 0.5 * float(x0 @ P @ x0), np.array(U)


def arc_reference(N: int = 40, v: float = 8.0, yaw_rate: float = 0.2) -> np.ndarray:
    """(N+1, 4) reference produced by the bicycle itself, so it is exactly trackable."""
    delta = math.atan(yaw_rate * bicycle.WHEEL_BASE / v)
    x = np.array([0.0, 0.0, 0.0, v, 0.0, delta])
    states = [x]
    for _ in range(N):
        x = bicycle.step(x, np.zeros(2))
        states.append(x)
    return np.array(states)[:, :4]


def toy_case(lateral_offset: float = 1.0, steer_rate_max: float = 0.15):
    """Ego starts beside the arc with zero steering; a tight steer-rate box forces clamped knots."""
    ref = arc_reference()
    problem = TrackingProblem(ref, u_lb=(-4.0, -steer_rate_max), u_ub=(4.0, steer_rate_max))
    x0 = np.array([0.0, lateral_offset, 0.0, ref[0, 3], 0.0, 0.0])
    return problem, x0, np.zeros((problem.N, problem.m))


def reg_path_case():
    """Zero control-effort weight makes Quu singular, so every backward pass needs regularization, and the
    near bang-bang steps make line searches fail. Exercises the whole reg schedule, which the toy case never does."""
    ref = arc_reference()
    problem = TrackingProblem(ref, w_jerk=0.0, w_steer_rate=0.0)
    x0 = np.array([0.0, 1.0, 0.0, ref[0, 3], 0.0, 0.0])
    return problem, x0, np.zeros((problem.N, problem.m))


def lateral_error(state: np.ndarray, ref_row: np.ndarray) -> float:
    dx, dy, psi_r = state[0] - ref_row[0], state[1] - ref_row[1], ref_row[2]
    return float(-math.sin(psi_r) * dx + math.cos(psi_r) * dy)
