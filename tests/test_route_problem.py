"""C5, C5b: RouteProblem Jacobians vs central differences. C6: residual values by hand. C6b: the solver drives it."""
import math

import numpy as np
import pytest

from planner import bicycle
from planner.costs import RouteProblem
from planner.ilqr import solve
from planner.route import make_centerline, project

EPS = 1e-6
KINK = 1e-3  # stay this far from segment ends and hinge kinks, where the residuals are not differentiable


def _s_curve(v_limit=12.0, westbound=False):
    """S-curve along +x; westbound rotates it by pi, so its headings swing across +-pi."""
    x = np.linspace(0.0, 150.0, 1501)
    pts = np.column_stack([x, 3.0 * np.sin(x / 15.0)]) * (-1.0 if westbound else 1.0)
    return make_centerline(pts, np.full(len(pts), v_limit), a_lat_ref=3.0, b_ref=1.5)


def _straight(v_limit=10.0):
    x = np.linspace(0.0, 100.0, 201)
    return make_centerline(np.column_stack([x, 0 * x]), np.full(len(x), v_limit), a_lat_ref=3.0, b_ref=1.5)


def _smooth_states(problem, rng, K):
    """K states near the S-curve, rejection-sampled away from segment ends and hinge kinks."""
    cl = problem.cl
    rows = []
    while len(rows) < K:
        s = rng.uniform(5.0, 140.0)
        i = int(np.searchsorted(cl.s, s)) - 1
        t = cl.seg_t[i]
        base = cl.xy[i] + (s - cl.s[i]) * t + rng.uniform(-1.5, 1.5) * np.array([-t[1], t[0]])
        x = np.array([*base, np.arctan2(t[1], t[0]) + rng.uniform(-0.3, 0.3),
                      rng.uniform(-2.0, 20.0), rng.uniform(-6.0, 4.0), rng.uniform(-0.5, 0.5)])
        seg, f = project(cl, problem.lo, problem.hi, x[None, :2], x[2:3])
        near_end = min(f[0], 1.0 - f[0]) * cl.seg_len[seg[0]] < KINK
        near_hinge = np.any(np.abs(problem._limit_args(x[None])) < KINK)
        if not (near_end or near_hinge):
            rows.append(x)
    return np.array(rows)


@pytest.mark.parametrize("westbound", [False, True])
def test_c5_jacobians_match_central_differences(westbound):
    cl = _s_curve(westbound=westbound)
    # Tight limits so every hinge fires somewhere in the sample (defaults almost never trip lat, yaw, delta).
    problem = RouteProblem(cl, (0, len(cl.s) - 1), N=100, a_min=-1.0, a_max=1.0, delta_max=0.3,
                           a_lat_max=1.0, yaw_max=0.2)
    rng = np.random.default_rng(5)
    X, U = _smooth_states(problem, rng, 101), rng.uniform(-1.0, 1.0, (100, 2))
    active = problem._limit_args(X) > 0.0
    assert np.all(active.any(axis=0)), "every hinge must be active on at least one compared knot"
    Jx, Ju, JxN = problem.residual_jacobians(X, U)
    for k in range(problem.N + 1):
        for i in range(bicycle.NX):
            Xp, Xm = X.copy(), X.copy()
            Xp[k, i] += EPS
            Xm[k, i] -= EPS
            (rp, rNp), (rm, rNm) = problem.residuals(Xp, U), problem.residuals(Xm, U)
            if k < problem.N:
                fd, an = (rp[k] - rm[k]) / (2 * EPS), Jx[k, :, i]
            else:
                fd, an = (rNp - rNm) / (2 * EPS), JxN[:, i]
            assert np.allclose(an, fd, rtol=1e-6, atol=1e-6), f"knot {k}, state {i}"
    for k in range(problem.N):
        for i in range(bicycle.NU):
            Up, Um = U.copy(), U.copy()
            Up[k, i] += EPS
            Um[k, i] -= EPS
            fd = (problem.residuals(X, Up)[0][k] - problem.residuals(X, Um)[0][k]) / (2 * EPS)
            assert np.allclose(Ju[k, :, i], fd, rtol=1e-6, atol=1e-6), f"knot {k}, control {i}"


def test_c6_residuals_by_hand():
    cl = _straight()  # v_ref = 10 until the braking zone for the route end, which starts at 100 - 100/3 m
    problem = RouteProblem(cl, (0, len(cl.s) - 1), N=1)
    assert (problem.p, problem.pN) == (14, 12)
    assert np.array_equal(problem.u_lb, [-4.0, -0.5]) and np.array_equal(problem.u_ub, [4.0, 0.5])
    X = np.array([[5.0, 0.5, 0.1, 8.0, 3.0, 0.2], [10.0, -0.3, -0.05, 12.0, -5.0, -1.2]])
    U = np.array([[1.5, -0.2]])
    r, rN = problem.residuals(X, U)
    L, h = 3.089, lambda z: max(z, 0.0)

    def limits(v, a, d):
        yaw = v * math.tan(d) / L
        lat = v * yaw
        return [h(a - 2.40), h(-4.05 - a), h(d - math.pi / 3), h(-math.pi / 3 - d), h(-v),
                h(lat - 4.0), h(-4.0 - lat), h(yaw - 0.8), h(-0.8 - yaw)]

    sl = math.sqrt(100.0)
    expect_r = ([1.0 * 0.5, 1.0 * 0.1, math.sqrt(0.5) * (8.0 - 10.0)]
                + [math.sqrt(0.1) * 1.5, math.sqrt(0.1) * -0.2]
                + [sl * z for z in limits(8.0, 3.0, 0.2)])
    expect_rN = [1.0 * -0.3, 1.0 * -0.05, math.sqrt(0.5) * (12.0 - 10.0)] + [sl * z for z in limits(12.0, -5.0, -1.2)]
    assert r.shape == (1, 14) and rN.shape == (12,)
    assert np.allclose(r[0], expect_r, rtol=1e-12, atol=1e-12)
    assert np.allclose(rN, expect_rN, rtol=1e-12, atol=1e-12)
    # The hand case exercises: a_hi and lat_hi at knot 0; a_lo, delta_lo, lat_lo and yaw_lo at the terminal knot.
    assert np.count_nonzero(r[0, 5:]) == 2 and np.count_nonzero(rN[3:]) == 4
    # Heading wrap: a westbound centerline (heading pi) and a knot at -pi + 0.05, which is 0.05 rad left of it.
    west = make_centerline(np.column_stack([np.linspace(100.0, 0.0, 201), np.zeros(201)]), np.full(201, 10.0),
                           a_lat_ref=3.0, b_ref=1.5)
    Xw = np.array([[50.0, 0.0, -math.pi + 0.05, 10.0, 0.0, 0.0]] * 2)
    rw, _ = RouteProblem(west, (0, len(west.s) - 1), N=1).residuals(Xw, U)
    assert rw[0, 1] == pytest.approx(0.05, abs=1e-12)


def test_c6b_solver_pulls_an_offset_start_onto_the_centerline():
    cl = _s_curve()
    problem = RouteProblem(cl, cl.segment_window(0.0, 80.0))
    x0 = np.array([0.0, 1.0, 0.0, 8.0, 0.0, 0.0])  # 1 m left of the centerline at its start
    sol = solve(problem, x0, np.zeros((problem.N, 2)), max_iters=50, budget_s=math.inf)
    assert sol.status == "converged"
    assert np.all(sol.U >= problem.u_lb) and np.all(sol.U <= problem.u_ub)
    r, rN = problem.residuals(sol.X, sol.U)
    assert abs(rN[0]) < 0.1  # terminal lateral error (w_ey = 1)
    assert abs(rN[2]) / math.sqrt(0.5) < 0.5  # terminal speed error, m/s
    # Limits are soft hinges, so a small overshoot is expected: measured 0.0065 m/s^2 on a_max here (the solver
    # trades it against speed error). 0.02 bounds it; a hinge with a flipped sign or a lost Jacobian overshoots far more.
    assert np.max(problem._limit_args(sol.X)) < 0.02


def test_c5b_knots_past_the_window_end_have_exact_jacobians():
    """Past the last segment the fraction clips to 1, so e_psi and e_v stop depending on position. The horizon runs
    past the window near the route end, where v_ref drops to 0, so a Jacobian that ignored the clip would be wrong."""
    cl = _straight()
    problem = RouteProblem(cl, (0, len(cl.s) - 1), N=1)
    X = np.array([[103.0, 0.7, 0.05, 3.0, 0.0, 0.0], [104.0, -0.4, -0.02, 2.0, 0.0, 0.0]])  # 3-4 m past the end
    _check_xy_jacobians(problem, X)


def test_c5b_knots_before_the_window_start_have_exact_jacobians():
    """The same clip at the other end: the window starts inside the braking zone (v_ref falls from 67 m on), and the
    knots sit 2-3 m before it."""
    cl = _straight()
    problem = RouteProblem(cl, cl.segment_window(80.0, 100.0), N=1)
    X = np.array([[77.0, 0.3, 0.02, 6.0, 0.0, 0.0], [78.0, -0.2, 0.01, 6.0, 0.0, 0.0]])
    _check_xy_jacobians(problem, X)


def _check_xy_jacobians(problem, X):
    U = np.zeros((1, 2))
    Jx, _, JxN = problem.residual_jacobians(X, U)
    for k, J in ((0, Jx[0]), (1, JxN)):
        for i in range(2):  # x, y
            Xp, Xm = X.copy(), X.copy()
            Xp[k, i] += EPS
            Xm[k, i] -= EPS
            rp, rNp = problem.residuals(Xp, U)
            rm, rNm = problem.residuals(Xm, U)
            fd = (rp[0] - rm[0]) / (2 * EPS) if k == 0 else (rNp - rNm) / (2 * EPS)
            assert np.allclose(J[:, i], fd, atol=1e-7), f"knot {k}, coordinate {i}"
