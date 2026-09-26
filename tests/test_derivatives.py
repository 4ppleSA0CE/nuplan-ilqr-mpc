"""T3: analytic Jacobians vs central differences. T4: Euler gap to the exact continuous-time arc."""
import math

import numpy as np

from planner import bicycle
from planner.costs import TrackingProblem

EPS = 1e-6


def _random_states(rng, K, v_lo=0.0):
    return np.column_stack(
        [
            rng.uniform(-50, 50, K),
            rng.uniform(-50, 50, K),
            rng.uniform(-math.pi, math.pi, K),
            rng.uniform(v_lo, 20.0, K),
            rng.uniform(-6.0, 4.0, K),
            rng.uniform(-0.5, 0.5, K),
        ]
    )


def test_bicycle_jacobians_match_central_differences():
    rng = np.random.default_rng(0)
    N = 100
    X, U = _random_states(rng, N + 1), rng.uniform(-1, 1, (N, 2))
    A, B = bicycle.linearize(X, U)
    for k in range(N):
        for i in range(bicycle.NX):
            d = np.zeros(bicycle.NX)
            d[i] = EPS
            fd = (bicycle.step(X[k] + d, U[k]) - bicycle.step(X[k] - d, U[k])) / (2 * EPS)
            assert np.allclose(A[k, :, i], fd, rtol=1e-6, atol=1e-7), f"A: knot {k}, state {i}"
        for i in range(bicycle.NU):
            d = np.zeros(bicycle.NU)
            d[i] = EPS
            fd = (bicycle.step(X[k], U[k] + d) - bicycle.step(X[k], U[k] - d)) / (2 * EPS)
            assert np.allclose(B[k, :, i], fd, rtol=1e-6, atol=1e-7), f"B: knot {k}, control {i}"


def test_residual_jacobians_match_central_differences():
    rng = np.random.default_rng(1)
    N = 100
    X, U = _random_states(rng, N + 1, v_lo=-2.0), rng.uniform(-1, 1, (N, 2))
    ref = X[:, :4] + rng.uniform(-1.0, 1.0, (N + 1, 4))  # heading error stays far from the +-pi wrap
    problem = TrackingProblem(ref, delta_max=0.3, a_min=-3.0, a_max=1.0)  # tight limits so hinges are active
    z = problem._limit_args(X)
    keep = (np.abs(z) > 1e-3).all(axis=1)  # stay off the kink, where the derivative is undefined
    assert (z[keep] > 0).any(axis=0).all(), "every hinge must be active on a compared knot or the test proves nothing"
    assert keep[-1], "terminal knot sits on a kink, so JxN would go unchecked: change the seed"

    Jx, Ju, JxN = problem.residual_jacobians(X, U)
    # Residuals at knot k depend only on (x_k, u_k), so perturbing one component at every knot at once
    # gives that Jacobian column for all knots in a single evaluation pair.
    for i in range(problem.n):
        d = np.zeros_like(X)
        d[:, i] = EPS
        rp, rNp = problem.residuals(X + d, U)
        rm, rNm = problem.residuals(X - d, U)
        assert np.allclose(Jx[keep[:-1], :, i], ((rp - rm) / (2 * EPS))[keep[:-1]], rtol=1e-6, atol=1e-7), f"Jx: state {i}"
        assert np.allclose(JxN[:, i], (rNp - rNm) / (2 * EPS), rtol=1e-6, atol=1e-7), f"JxN: state {i}"
    for i in range(problem.m):
        d = np.zeros_like(U)
        d[:, i] = EPS
        rp, _ = problem.residuals(X, U + d)
        rm, _ = problem.residuals(X, U - d)
        assert np.allclose(Ju[:, :, i], (rp - rm) / (2 * EPS), rtol=1e-6, atol=1e-7), f"Ju: control {i}"


def test_residual_values_and_shapes_match_spec_table():
    """Guards what finite differences cannot: residual order, sqrt(w) scaling, hinge arguments, terminal row, shapes."""
    ref = np.array([[0.0, 0.0, 0.0, 5.0], [1.0, 2.0, 3.0, 4.0]])
    problem = TrackingProblem(ref, w_pos=4.0, w_psi=9.0, w_v=16.0, w_jerk=25.0, w_steer_rate=36.0, w_limit=49.0)
    assert (problem.n, problem.m, problem.N, problem.p, problem.pN) == (6, 2, 1, 11, 9)
    assert np.array_equal(problem.u_lb, [-4.0, -0.5]) and np.array_equal(problem.u_ub, [4.0, 0.5])

    X = np.array([[0.5, -1.0, 0.1 + 2 * math.pi, 7.0, 3.0, 1.2], [1.0, 2.0, 3.0, -1.0, -5.0, -1.3]])
    U = np.array([[2.0, -0.5]])
    d_max = math.pi / 3
    #            x        y         psi (wrapped)  v        jerk     steer     a_hi             a_lo  delta_hi           delta_lo  v_neg
    expected_r = [2 * 0.5, 2 * -1.0, 3 * 0.1, 4 * 2.0, 5 * 2.0, 6 * -0.5, 7 * (3.0 - 2.40), 0.0, 7 * (1.2 - d_max), 0.0, 0.0]
    #             tracking vs the LAST ref row   a_hi  a_lo               delta_hi  delta_lo            v_neg
    expected_rN = [0.0, 0.0, 0.0, 4 * (-1.0 - 4.0), 0.0, 7 * (-4.05 + 5.0), 0.0, 7 * (-d_max + 1.3), 7 * 1.0]

    r, rN = problem.residuals(X, U)
    assert r.shape == (1, 11) and rN.shape == (9,)
    assert np.allclose(r[0], expected_r) and np.allclose(rN, expected_rN)
    Jx, Ju, JxN = problem.residual_jacobians(X, U)
    assert Jx.shape == (1, 11, 6) and Ju.shape == (1, 11, 2) and JxN.shape == (9, 6)

    ref[:] = 99.0  # the problem must own its reference: planners reuse these buffers between ticks
    assert np.allclose(problem.residuals(X, U)[0][0], expected_r)


def test_euler_gap_to_exact_arc():
    """T4. Constant speed and steering trace an exact circle in continuous time; Euler lags it."""
    v, yaw_rate, N = 10.0, 0.3, 40
    x = np.array([0.0, 0.0, 0.0, v, 0.0, math.atan(yaw_rate * bicycle.WHEEL_BASE / v)])
    for _ in range(N):
        x = bicycle.step(x, np.zeros(2))
    T, R = N * bicycle.DT, v / yaw_rate
    exact = np.array([R * math.sin(yaw_rate * T), R * (1.0 - math.cos(yaw_rate * T))])
    gap = float(np.linalg.norm(x[:2] - exact))
    print(f"\nT4 Euler endpoint gap after {T:.1f} s: {gap:.3f} m (heading error {x[2] - yaw_rate * T:+.2e} rad)")
    assert 0.5 < gap < 0.65  # hand estimate 0.56 m; a dynamics regression moves this
