"""T1 (LQR in one iteration), T5 (active bounds), T6 (determinism), plus the backward pass against an independent
recursion, the regularization path, and the failure statuses."""
import numpy as np
from helpers import lateral_error, random_lqr, reg_path_case, riccati, toy_case

from planner import ilqr
from planner.ilqr import boxqp, solve


def _initial_cost(problem, x0, U):
    X = [x0]
    for u in U:
        X.append(problem.step(X[-1], u))
    r, rN = problem.residuals(np.array(X), U)
    return 0.5 * (np.sum(r * r) + np.sum(rN * rN))


def test_lqr_converges_in_one_iteration_to_riccati():
    for seed in range(5):
        problem, x0 = random_lqr(seed)
        cost_star, U_star = riccati(problem, x0)
        U_init = np.zeros((problem.N, problem.m))
        sol = solve(problem, x0, U_init, max_iters=2, budget_s=np.inf)
        assert sol.status == "converged", f"seed {seed}"
        assert sol.iters == 1, f"seed {seed}"  # the second pass sees a zero step and stops before any line search
        assert abs(sol.cost - cost_star) <= 1e-9 * cost_star, f"seed {seed}"
        assert np.allclose(sol.U, U_star, rtol=0, atol=1e-8), f"seed {seed}"
        # The quadratic model is exact for LQR, so a full step must deliver exactly the predicted reduction.
        _, _, alpha, expected, _ = sol.trace[0]
        assert alpha == 1.0
        assert abs((_initial_cost(problem, x0, U_init) - sol.cost) / expected - 1.0) < 1e-9, f"seed {seed}"


def _reference_backward(A, B, Jx, Ju, r, JxN, rN, U, lb, ub, reg):
    """Spec section 5.2 written knot by knot with plain matrix products, independently of planner.ilqr."""
    N, n, m = B.shape
    Vx, Vxx = JxN.T @ rN, JxN.T @ JxN
    k_ff, K, dV1, dV2 = np.zeros((N, m)), np.zeros((N, m, n)), 0.0, 0.0
    for k in reversed(range(N)):
        lx, lu = Jx[k].T @ r[k], Ju[k].T @ r[k]
        lxx, luu, lux = Jx[k].T @ Jx[k], Ju[k].T @ Ju[k], Ju[k].T @ Jx[k]
        Qx, Qu = lx + A[k].T @ Vx, lu + B[k].T @ Vx
        Qxx = lxx + A[k].T @ Vxx @ A[k]
        Quu = luu + B[k].T @ Vxx @ B[k]
        Qux = lux + B[k].T @ Vxx @ A[k]
        V_reg = Vxx + reg * np.eye(n)
        Quu_reg = luu + B[k].T @ V_reg @ B[k]  # eq. 10: regularized blocks give the step and the gains
        Qux_reg = lux + B[k].T @ V_reg @ A[k]
        kk, free, _, ok = boxqp(Quu_reg, Qu, lb - U[k], ub - U[k], np.zeros(m))
        assert ok
        Kk = np.zeros((m, n))
        Kk[free] = -np.linalg.solve(Quu_reg[np.ix_(free, free)], Qux_reg[free])
        dV1 += kk @ Qu
        dV2 += 0.5 * kk @ Quu @ kk  # eq. 11: unregularized blocks from here on
        Vx = Qx + Kk.T @ Quu @ kk + Kk.T @ Qu + Qux.T @ kk
        Vxx = Qxx + Kk.T @ Quu @ Kk + Kk.T @ Qux + Qux.T @ Kk
        Vxx = 0.5 * (Vxx + Vxx.T)
        k_ff[k], K[k] = kk, Kk
    return k_ff, K, dV1, dV2


class _DenseProblem:
    """Fixed random dense Jacobians, so lux = Ju'Jx is nonzero. Both real problems have lux identically zero."""

    def __init__(self, rng, N=6, n=4, m=3, p=9):
        self.N, self.n, self.m = N, n, m
        self.A, self.B = np.eye(n) + 0.3 * rng.standard_normal((N, n, n)), rng.standard_normal((N, n, m))
        self.Jx, self.Ju = rng.standard_normal((N, p, n)), rng.standard_normal((N, p, m))
        self.r, self.JxN, self.rN = rng.standard_normal((N, p)), rng.standard_normal((5, n)), rng.standard_normal(5)

    def linearize(self, X, U):
        return self.A, self.B

    def residuals(self, X, U):
        return self.r, self.rN

    def residual_jacobians(self, X, U):
        return self.Jx, self.Ju, self.JxN


def test_backward_pass_matches_independent_recursion():
    rng = np.random.default_rng(3)
    n_clamped_seen = 0
    for reg in (0.0, 0.37, 50.0):
        q = _DenseProblem(rng)
        U = rng.uniform(-1, 1, (q.N, q.m))
        lb, ub = np.full(q.m, -1.05), np.full(q.m, 1.05)  # tight around U so some controls clamp
        expansion = ilqr._expand(q, np.zeros((q.N + 1, q.n)), U)
        assert np.abs(expansion.lux).max() > 0.1 and expansion.lux.shape == (q.N, q.m, q.n)
        k_ff, K, dV1, dV2, n_clamped = ilqr._backward(expansion, U, lb, ub, reg, np.zeros((q.N, q.m)))
        k_ref, K_ref, dV1_ref, dV2_ref = _reference_backward(q.A, q.B, q.Jx, q.Ju, q.r, q.JxN, q.rN, U, lb, ub, reg)
        assert np.allclose(k_ff, k_ref, rtol=0, atol=1e-9) and np.allclose(K, K_ref, rtol=0, atol=1e-9), f"reg {reg}"
        assert abs(dV1 - dV1_ref) <= 1e-9 * abs(dV1_ref) and abs(dV2 - dV2_ref) <= 1e-9 * abs(dV2_ref), f"reg {reg}"
        clamped = np.isclose(U + k_ff, lb, rtol=0, atol=1e-12) | np.isclose(U + k_ff, ub, rtol=0, atol=1e-12)
        assert np.all(K[clamped] == 0.0), "gain rows of clamped controls must be zero"
        assert dV1 < 0.0 <= dV2 and -dV1 >= dV2  # guarantees a positive expected reduction for every alpha in (0, 1]
        n_clamped_seen += n_clamped
    assert n_clamped_seen > 0


def test_toy_problem_converges_with_active_bounds():
    problem, x0, U_init = toy_case()
    sol = solve(problem, x0, U_init, max_iters=50, budget_s=np.inf)
    assert sol.status == "converged"
    assert len(sol.trace) == sol.iters

    costs = [row[0] for row in sol.trace if row[2] > 0.0]
    assert all(b <= a for a, b in zip(costs, costs[1:]))

    assert np.all(sol.U >= problem.u_lb) and np.all(sol.U <= problem.u_ub)
    at_bound = (sol.U == problem.u_lb) | (sol.U == problem.u_ub)
    assert at_bound.any(), "sigma box never active: the test is not exercising box-QP"

    assert abs(lateral_error(sol.X[-1], problem.ref[-1])) < 0.1


def test_regularization_path_is_exercised_and_sane():
    problem, x0, U_init = reg_path_case()
    sol = solve(problem, x0, U_init, max_iters=60, budget_s=np.inf)
    assert sol.status in ("converged", "max_iters") and len(sol.trace) == sol.iters
    regs = [row[1] for row in sol.trace]
    accepted = [row for row in sol.trace if row[2] > 0.0]
    assert len(accepted) < len(sol.trace), "no rejected line search: the failure branch is not being exercised"
    assert min(regs) > 0.0 and max(regs) > 1.0, "regularization never had to climb"
    assert regs[-1] <= 1e-5, "regularization must come back down after it has done its job"
    assert all(row[3] > 0.0 for row in accepted), "accepted a step with a non-positive expected reduction"
    costs = [row[0] for row in accepted]
    assert all(b <= a for a, b in zip(costs, costs[1:]))
    assert sol.cost < 0.05 * _initial_cost(problem, x0, U_init)
    assert np.all(sol.U >= problem.u_lb) and np.all(sol.U <= problem.u_ub)


def test_solve_is_deterministic_and_stateless():
    problem, x0, U_init = toy_case()
    a = solve(problem, x0, U_init, max_iters=50, budget_s=np.inf)
    other, x0_other, U_other = reg_path_case()
    solve(other, x0_other, U_other, max_iters=5, budget_s=np.inf)  # an unrelated solve in between must leave no trace
    b = solve(problem, x0, U_init, max_iters=50, budget_s=np.inf)
    assert np.array_equal(a.X, b.X) and np.array_equal(a.U, b.U) and a.trace == b.trace
    assert (a.cost, a.iters, a.status) == (b.cost, b.iters, b.status)


def test_inputs_are_not_modified_and_u_init_outside_the_box_is_clipped():
    problem, x0, _ = toy_case()
    U_init = np.full((problem.N, problem.m), 1e3)
    x0_copy, U_copy, ref_copy = x0.copy(), U_init.copy(), problem.ref.copy()
    sol = solve(problem, x0, U_init, max_iters=50, budget_s=np.inf)
    assert np.array_equal(x0, x0_copy) and np.array_equal(U_init, U_copy) and np.array_equal(problem.ref, ref_copy)
    assert np.all(sol.U >= problem.u_lb) and np.all(sol.U <= problem.u_ub)
    assert sol.trace[0][0] < _initial_cost(problem, x0, np.clip(U_init, problem.u_lb, problem.u_ub))


def test_budget_stops_the_solve():
    problem, x0, U_init = toy_case()
    sol = solve(problem, x0, U_init, max_iters=50, budget_s=0.0)
    assert sol.status == "budget" and sol.iters == 0
    assert sol.X.shape == (problem.N + 1, problem.n)  # still returns the rolled-out seed


def test_max_iters_stops_the_solve():
    problem, x0, U_init = toy_case()
    sol = solve(problem, x0, U_init, max_iters=3, budget_s=np.inf)
    assert sol.status == "max_iters" and sol.iters == 3 and len(sol.trace) == 3


def test_non_finite_input_returns_immediately():
    problem, x0, U_init = toy_case()
    x0[3] = np.nan
    sol = solve(problem, x0, U_init, max_iters=50, budget_s=np.inf)
    assert sol.status == "non_finite" and sol.iters == 0 and sol.solve_time_s < 0.05
    assert np.array_equal(sol.U, U_init), "a caller falling back after non_finite must still get usable controls"


def test_reg_max_is_reported_even_if_the_last_retry_succeeds(monkeypatch):
    """Found by review: a backward pass that first succeeds above REG_MAX used to continue and report converged."""
    problem, x0, U_init = toy_case()
    real = ilqr._backward
    monkeypatch.setattr(ilqr, "_backward", lambda exp, U, lb, ub, reg, k: real(exp, U, lb, ub, reg, k) if reg > ilqr.REG_MAX else None)
    assert solve(problem, x0, U_init, max_iters=50, budget_s=np.inf).status == "reg_max"
