"""Box-constrained Gauss-Newton iLQR. Knows nothing about driving.

References: Tassa, Erez, Todorov 2012 (regularization, value update);
Tassa, Mansard, Todorov 2014 (box-QP, control-limited DDP).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
from scipy.linalg import cho_solve

REG_FACTOR = 1.6
REG_MIN = 1e-6
REG_MAX = 1e10
REG_CONVERGED = 1e-5  # the step test only counts as convergence below this (Tassa: lambda < 1e-5)
ALPHAS = tuple(0.5**i for i in range(8))
ARMIJO = 1e-4

TraceRow = Tuple[float, float, float, float, int]  # cost, reg, alpha, expected_reduction, n_clamped


@dataclass
class Solution:
    X: np.ndarray  # (N+1, n)
    U: np.ndarray  # (N, m)
    cost: float
    iters: int
    status: str  # converged | max_iters | budget | reg_max | non_finite
    solve_time_s: float
    trace: List[TraceRow] = field(default_factory=list)


def _chol_solve(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve (L L') x = b from the lower Cholesky factor. 4x cheaper here than two general solves on L."""
    return cho_solve((L, True), b, check_finite=False)


def boxqp(
    H: np.ndarray, g: np.ndarray, lb: np.ndarray, ub: np.ndarray, x0: np.ndarray, max_iters: int = 100, tol: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Minimize 0.5 x'Hx + g'x subject to lb <= x <= ub. Newton steps on the free set, cut at the first bound hit.

    :param H: (m, m) symmetric. :param g: (m,). :param lb, ub: (m,), may be infinite. :param x0: (m,) warm start.
    :return: x (m,), free (m,) bool mask, L (Cholesky factor of H[free][:, free]), ok.
        x is always feasible. When ok is True, free and L describe the returned x. ok is False if the free block
        is not positive definite or the iteration limit is hit; the caller should then raise regularization.
    """
    x = np.clip(x0, lb, ub)
    for it in range(max_iters + 1):
        grad = H @ x + g
        # Clamped = on a bound with the gradient pushing outward (the KKT sign condition). Everything else is free.
        free = ~(((x <= lb) & (grad > 0)) | ((x >= ub) & (grad < 0)))
        if not free.any():
            return x, free, np.zeros((0, 0)), True
        try:
            L = np.linalg.cholesky(H[np.ix_(free, free)])
        except np.linalg.LinAlgError:
            return x, free, np.zeros((0, 0)), False
        step = -_chol_solve(L, grad[free])

        # Newton decrement: f(x) minus the minimum over the free subspace is exactly -0.5 * grad_f . step.
        # A gradient-norm test is not used: with an ill-conditioned H a tiny gradient can hide a large objective
        # gap, and an absolute threshold is unreachable in float64 once |H| is large.
        gap = -0.5 * float(grad[free] @ step)
        f_scale = 0.5 * float(np.abs(x) @ np.abs(H) @ np.abs(x)) + float(np.abs(g) @ np.abs(x))
        if gap <= tol * f_scale:
            return x, free, L, True
        if it == max_iters:
            return x, free, L, False

        # A free coordinate sitting on a bound whose step points outward cannot move. Hold it for this iteration
        # and re-solve on the rest; H[move][:, move] is PD because it is a principal block of a PD block.
        move = free.copy()
        dx = np.zeros_like(x)
        dx[free] = step
        while True:
            blocked = move & (((x <= lb) & (dx < 0)) | ((x >= ub) & (dx > 0)))
            if not blocked.any():
                break
            move &= ~blocked
            dx[:] = 0.0
            if not move.any():
                dx[free] = -grad[free] / np.diag(H)[free]  # all free coordinates are on a bound: descend inward
                break
            dx[move] = -np.linalg.solve(H[np.ix_(move, move)], grad[move])

        # Exact minimizer along dx (1 for a Newton direction), cut at the first bound reached. The objective is a
        # convex quadratic, so this always decreases it and no backtracking is needed. Backtracking along the
        # clipped arc stalls when a coordinate is a hair inside a bound.
        alpha = -float(grad @ dx) / float(dx @ H @ dx)
        with np.errstate(divide="ignore", invalid="ignore"):
            room = np.where(dx > 0, (ub - x) / dx, np.where(dx < 0, (lb - x) / dx, np.inf))
        alpha = min(alpha, float(room.min()))
        hit = room <= alpha
        x = x + alpha * dx
        x[hit & (dx > 0)] = ub[hit & (dx > 0)]  # land exactly on the bound so the next pass sees it as active
        x[hit & (dx < 0)] = lb[hit & (dx < 0)]
        x = np.clip(x, lb, ub)
    raise AssertionError("unreachable: the loop returns on its last pass")


def _increase_reg(reg: float, dreg: float) -> Tuple[float, float]:
    dreg = max(dreg * REG_FACTOR, REG_FACTOR)
    return max(reg * dreg, REG_MIN), dreg


def _decrease_reg(reg: float, dreg: float) -> Tuple[float, float]:
    dreg = min(dreg / REG_FACTOR, 1.0 / REG_FACTOR)
    return (reg * dreg if reg * dreg > REG_MIN else 0.0), dreg


def _cost(problem, X: np.ndarray, U: np.ndarray) -> float:
    r, rN = problem.residuals(X, U)
    return 0.5 * (float(np.sum(r * r)) + float(np.sum(rN * rN)))


def _rollout(problem, x0: np.ndarray, U_init: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Open-loop rollout of U_init clipped to the box. x0 (n,), U_init (N, m) -> X (N+1, n), U (N, m)."""
    U = np.clip(U_init, problem.u_lb, problem.u_ub)
    X = np.empty((problem.N + 1, problem.n))
    X[0] = x0
    for k in range(problem.N):
        X[k + 1] = problem.step(X[k], U[k])
    return X, U


def _forward(problem, X, U, k_ff, K, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """Closed-loop rollout around (X, U) with controls clipped to the box.

    X (N+1, n), U (N, m), k_ff (N, m), K (N, m, n) -> X_new (N+1, n), U_new (N, m).
    """
    X_new = np.empty_like(X)
    U_new = np.empty_like(U)
    X_new[0] = X[0]
    for k in range(problem.N):
        u = U[k] + alpha * k_ff[k] + K[k] @ (X_new[k] - X[k])
        U_new[k] = np.clip(u, problem.u_lb, problem.u_ub)
        X_new[k + 1] = problem.step(X_new[k], U_new[k])
    return X_new, U_new


class Expansion(NamedTuple):
    """Everything the backward pass needs that depends only on (X, U). Leading axis is the knot, except Vx, Vxx."""

    A: np.ndarray  # (N, n, n)
    B: np.ndarray  # (N, n, m)
    lx: np.ndarray  # (N, n)   Jx'r
    lu: np.ndarray  # (N, m)   Ju'r
    lxx: np.ndarray  # (N, n, n)  Jx'Jx
    luu: np.ndarray  # (N, m, m)  Ju'Ju
    lux: np.ndarray  # (N, m, n)  Ju'Jx
    Vx: np.ndarray  # (n,)     terminal JxN'rN
    Vxx: np.ndarray  # (n, n)   terminal JxN'JxN


def _expand(problem, X: np.ndarray, U: np.ndarray) -> Expansion:
    """Gauss-Newton expansion at (X, U), batched over knots. Computed once per iterate so that regularization
    retries and rejected line searches reuse it."""
    A, B = problem.linearize(X, U)
    r, rN = problem.residuals(X, U)
    Jx, Ju, JxN = problem.residual_jacobians(X, U)
    return Expansion(
        A,
        B,
        np.einsum("kpi,kp->ki", Jx, r),
        np.einsum("kpi,kp->ki", Ju, r),
        np.einsum("kpi,kpj->kij", Jx, Jx),
        np.einsum("kpi,kpj->kij", Ju, Ju),
        np.einsum("kpi,kpj->kij", Ju, Jx),
        JxN.T @ rN,
        JxN.T @ JxN,
    )


def _backward(e: Expansion, U, u_lb, u_ub, reg: float, k_prev) -> Optional[tuple]:
    """One backward pass. U (N, m), u_lb/u_ub (m,), k_prev (N, m) box-QP warm starts.

    Returns k_ff (N, m), K (N, m, n), dV1, dV2, n_clamped, or None if a box-QP fails (caller raises reg).

    The step and the gains use the regularized Quu, Qux (Tassa 2012 eq. 10); the value function update and the
    expected reduction use the unregularized ones (eq. 11).
    """
    A, B, lx, lu, lxx, luu, lux, Vx, Vxx = e
    N, n, m = B.shape
    k_ff = np.zeros((N, m))
    K = np.zeros((N, m, n))
    dV1 = dV2 = 0.0
    n_clamped = 0
    reg_eye = reg * np.eye(n)
    for k in range(N - 1, -1, -1):
        Ak, Bk = A[k], B[k]
        Qx = lx[k] + Ak.T @ Vx
        Qu = lu[k] + Bk.T @ Vx
        Qxx = lxx[k] + Ak.T @ Vxx @ Ak
        Quu = luu[k] + Bk.T @ Vxx @ Bk
        Qux = lux[k] + Bk.T @ Vxx @ Ak
        Vreg = Vxx + reg_eye
        Quu_reg = luu[k] + Bk.T @ Vreg @ Bk
        Qux_reg = lux[k] + Bk.T @ Vreg @ Ak

        kk, free, L, ok = boxqp(Quu_reg, Qu, u_lb - U[k], u_ub - U[k], k_prev[k])
        if not ok:
            return None
        Kk = np.zeros((m, n))  # rows of clamped controls stay zero (Tassa 2014)
        if free.any():
            Kk[free] = -_chol_solve(L, Qux_reg[free])
        n_clamped += int(m - free.sum())

        dV1 += float(kk @ Qu)
        dV2 += 0.5 * float(kk @ Quu @ kk)
        Vx = Qx + Kk.T @ Quu @ kk + Kk.T @ Qu + Qux.T @ kk
        Vxx = Qxx + Kk.T @ Quu @ Kk + Kk.T @ Qux + Qux.T @ Kk
        Vxx = 0.5 * (Vxx + Vxx.T)
        k_ff[k], K[k] = kk, Kk
    return k_ff, K, dV1, dV2, n_clamped


def solve(
    problem,
    x0: np.ndarray,
    U_init: np.ndarray,
    *,
    max_iters: int = 10,
    budget_s: float = 0.080,
    tol_cost: float = 1e-6,
    tol_step: float = 1e-4,
) -> Solution:
    """Solve min 0.5*sum||r_k||^2 + 0.5*||r_N||^2 subject to dynamics and u_lb <= u <= u_ub.

    `problem` is duck-typed: n, m, N, u_lb, u_ub, step, linearize, residuals, residual_jacobians
    (shapes in docs/specs/phase-2-solver.md section 4.1). No state is kept between calls and no input is
    modified. U_init may lie outside the box; the rollout clips it.
    """
    t0 = time.perf_counter()
    N, n, m = problem.N, problem.n, problem.m
    assert x0.shape == (n,) and U_init.shape == (N, m), f"x0 {x0.shape} / U_init {U_init.shape} vs n={n}, N={N}, m={m}"
    X, U = _rollout(problem, x0, U_init)
    J = _cost(problem, X, U)
    if not np.isfinite(J):  # bad input; regularization cannot fix this, so do not spend the budget trying
        return Solution(X=X, U=U, cost=J, iters=0, status="non_finite", solve_time_s=time.perf_counter() - t0)

    reg, dreg = 0.0, 1.0
    k_prev = np.zeros((N, m))
    trace: List[TraceRow] = []
    iters = 0
    expansion = None  # derivatives at (X, U); recomputed only after an accepted step
    while True:
        if iters >= max_iters:
            status = "max_iters"
            break
        if time.perf_counter() - t0 >= budget_s:
            status = "budget"
            break

        if expansion is None:
            expansion = _expand(problem, X, U)
        bw = _backward(expansion, U, problem.u_lb, problem.u_ub, reg, k_prev)
        while bw is None and reg <= REG_MAX:
            reg, dreg = _increase_reg(reg, dreg)
            bw = _backward(expansion, U, problem.u_lb, problem.u_ub, reg, k_prev)
        if bw is None or reg > REG_MAX:
            status = "reg_max"
            break
        k_ff, K, dV1, dV2, n_clamped = bw

        # Heavy regularization shrinks the step, which must not be mistaken for convergence.
        step_norm = float(np.max(np.max(np.abs(k_ff), axis=1) / (np.max(np.abs(U), axis=1) + 1.0)))
        if step_norm < tol_step and reg < REG_CONVERGED:
            status = "converged"
            break

        iters += 1
        accepted = False
        for alpha in ALPHAS:
            X_new, U_new = _forward(problem, X, U, k_ff, K, alpha)
            J_new = _cost(problem, X_new, U_new)
            expected = -(alpha * dV1 + alpha * alpha * dV2)
            if expected > 0.0 and (J - J_new) / expected >= ARMIJO:  # a non-finite J_new compares False
                accepted = True
                break
        if not accepted:
            trace.append((J, reg, 0.0, 0.0, n_clamped))  # reg is always the value the backward pass used
            reg, dreg = _increase_reg(reg, dreg)
            if reg > REG_MAX:
                status = "reg_max"
                break
            continue

        trace.append((J_new, reg, alpha, expected, n_clamped))
        reg, dreg = _decrease_reg(reg, dreg)
        k_prev = k_ff
        expansion = None
        # A small decrease only means convergence after a full step; after a short step it means a short step.
        small = alpha == 1.0 and (J - J_new) < tol_cost * max(1.0, J)
        X, U, J = X_new, U_new, J_new
        if small:
            status = "converged"
            break

    return Solution(X=X, U=U, cost=J, iters=iters, status=status, solve_time_s=time.perf_counter() - t0, trace=trace)
