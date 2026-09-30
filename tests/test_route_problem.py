"""RouteProblem. C5: Jacobians vs central differences. C5b: exact Jacobians for knots past either end of the
segment window. C6: residual values by hand. C6b: the solver pulls an offset start onto the centerline."""
import math

import numpy as np
import pytest

from planner.costs import KAPPA_MAX, SIGMA_MAX, SIGMA_WEIGHT, TRACK_OFFSET, RouteProblem
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
def test_jacobians_match_central_differences(westbound):
    cl = _s_curve(westbound=westbound)
    # Tight limits so every hinge fires somewhere in the sample (defaults almost never trip lat, yaw, kappa).
    # s_target far ahead: the terminal progress row is active, so its Jacobian is compared too.
    problem = RouteProblem(cl, (0, len(cl.s) - 1), N=100, s_target=200.0, a_min=-1.0, a_max=1.0, kappa_max=0.3,
                           a_lat_max=1.0, yaw_max=0.2)
    rng = np.random.default_rng(5)
    X, U = _smooth_states(problem, rng, 101), rng.uniform(-1.0, 1.0, (100, 2))
    active = problem._limit_args(X) > 0.0
    assert np.all(active.any(axis=0)), "every hinge must be active on at least one compared knot"
    Jx, Ju, JxN = problem.residual_jacobians(X, U)
    # Residuals at knot k depend only on (x_k, u_k), since each knot projects on its own onto a fixed window, so
    # perturbing one component at every knot at once gives that Jacobian column for all knots in one evaluation pair.
    for i in range(problem.n):
        d = np.zeros_like(X)
        d[:, i] = EPS
        rp, rNp = problem.residuals(X + d, U)
        rm, rNm = problem.residuals(X - d, U)
        assert np.allclose(Jx[:, :, i], (rp - rm) / (2 * EPS), rtol=1e-6, atol=1e-7), f"Jx: state {i}"
        assert np.allclose(JxN[:, i], (rNp - rNm) / (2 * EPS), rtol=1e-6, atol=1e-7), f"JxN: state {i}"
    for i in range(problem.m):
        d = np.zeros_like(U)
        d[:, i] = EPS
        rp, _ = problem.residuals(X, U + d)
        rm, _ = problem.residuals(X, U - d)
        assert np.allclose(Ju[:, :, i], (rp - rm) / (2 * EPS), rtol=1e-6, atol=1e-7), f"Ju: control {i}"


def test_residuals_by_hand():
    cl = _straight()  # v_ref = 10 until the braking zone for the route end, which starts at 100 - 100/3 m
    problem = RouteProblem(cl, (0, len(cl.s) - 1), N=1, track_offset=0.0)  # rear axle: hand values stay simple
    assert (problem.p, problem.pN) == (14, 13)
    assert np.array_equal(problem.u_lb, [-4.0, -SIGMA_MAX]) and np.array_equal(problem.u_ub, [4.0, SIGMA_MAX])
    X = np.array([[5.0, 0.5, 0.1, 8.0, 3.0, 0.07], [10.0, -0.3, -0.05, 12.0, -5.0, -0.6]])
    U = np.array([[1.5, -0.2]])
    r, rN = problem.residuals(X, U)
    h = lambda z: max(z, 0.0)  # noqa: E731

    def limits(v, a, k):
        yaw = v * k
        lat = v * yaw
        return [h(a - 2.40), h(-4.05 - a), h(k - KAPPA_MAX), h(-KAPPA_MAX - k), h(-v),
                h(lat - 4.0), h(-4.0 - lat), h(yaw - 0.8), h(-0.8 - yaw)]

    sl = math.sqrt(100.0)
    expect_r = ([1.0 * 0.5, 1.0 * 0.1, math.sqrt(0.5) * (8.0 - 10.0)]
                + [math.sqrt(0.1) * 1.5, math.sqrt(SIGMA_WEIGHT) * -0.2]
                + [sl * z for z in limits(8.0, 3.0, 0.07)])
    expect_rN = ([1.0 * -0.3, 1.0 * -0.05, math.sqrt(0.5) * (12.0 - 10.0)] + [sl * z for z in limits(12.0, -5.0, -0.6)]
                 + [0.0])  # no s_target: the progress row is zero
    assert r.shape == (1, 14) and rN.shape == (13,)
    assert np.allclose(r[0], expect_r, rtol=1e-12, atol=1e-12)
    assert np.allclose(rN, expect_rN, rtol=1e-12, atol=1e-12)
    # The hand case exercises: a_hi and lat_hi at knot 0; a_lo, kappa_lo, lat_lo and yaw_lo at the terminal knot.
    assert np.count_nonzero(r[0, 5:]) == 2 and np.count_nonzero(rN[3:]) == 4
    # Progress: the terminal knot is at x = 10 on the straight centerline, so s = 10; target 25 is 15 m short.
    _, rN_p = RouteProblem(cl, (0, len(cl.s) - 1), N=1, s_target=25.0, track_offset=0.0).residuals(X, U)
    assert rN_p[-1] == pytest.approx(math.sqrt(0.1) * (10.0 - 25.0), abs=1e-12)
    assert np.allclose(rN_p[:-1], rN[:-1], atol=1e-12)
    # Heading wrap: a westbound centerline (heading pi) and a knot at -pi + 0.05, which is 0.05 rad left of it.
    west = make_centerline(np.column_stack([np.linspace(100.0, 0.0, 201), np.zeros(201)]), np.full(201, 10.0),
                           a_lat_ref=3.0, b_ref=1.5)
    Xw = np.array([[50.0, 0.0, -math.pi + 0.05, 10.0, 0.0, 0.0]] * 2)
    rw, _ = RouteProblem(west, (0, len(west.s) - 1), N=1, track_offset=0.0).residuals(Xw, U)
    assert rw[0, 1] == pytest.approx(0.05, abs=1e-12)
    # Default tracked point: the geometric center, TRACK_OFFSET ahead along the heading. Knot 0 sits at y = 0.5 with
    # heading 0.1, so the center is 0.5 + TRACK_OFFSET sin(0.1) left of the x-axis centerline, TRACK_OFFSET cos(0.1)
    # further along it.
    r_c, rN_c = RouteProblem(cl, (0, len(cl.s) - 1), N=1, s_target=25.0).residuals(X, U)
    assert r_c[0, 0] == pytest.approx(0.5 + TRACK_OFFSET * math.sin(0.1), abs=1e-12)
    assert rN_c[-1] == pytest.approx(math.sqrt(0.1) * (10.0 + TRACK_OFFSET * math.cos(-0.05) - 25.0), abs=1e-12)


def test_solver_pulls_an_offset_start_onto_the_centerline():
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


def test_knots_past_the_window_end_have_exact_jacobians():
    """Past the last segment the fraction clips to 1, so e_psi and e_v stop depending on position. The horizon runs
    past the window near the route end, where v_ref drops to 0, so a Jacobian that ignored the clip would be wrong."""
    cl = _straight()
    problem = RouteProblem(cl, (0, len(cl.s) - 1), N=1)
    X = np.array([[103.0, 0.7, 0.05, 3.0, 0.0, 0.0], [104.0, -0.4, -0.02, 2.0, 0.0, 0.0]])  # 3-4 m past the end
    _check_xy_jacobians(problem, X)


def test_knots_before_the_window_start_have_exact_jacobians():
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


def test_progress_gradient_past_the_window_end_and_outside_a_curve():
    """The progress row extrapolates along its segment instead of clipping, so its gradient never vanishes: not past
    the window end, and not outside a curve, where a point can lie past the end of one segment and before the start
    of the next."""
    cl = _straight()
    problem = RouteProblem(cl, (0, len(cl.s) - 1), N=1, s_target=120.0)
    X = np.array([[103.0, 0.7, 0.05, 3.0, 0.0, 0.0], [104.0, -0.4, -0.02, 2.0, 0.0, 0.0]])  # 3-4 m past the end
    _, _, JxN = problem.residual_jacobians(X, np.zeros((1, 2)))
    # The tracked point is TRACK_OFFSET ahead of the rear axle: s = 104 + TRACK_OFFSET cos(psi), on a straight line.
    assert JxN[-1, :3] == pytest.approx(math.sqrt(0.1) * np.array([1.0, 0.0, -TRACK_OFFSET * math.sin(-0.02)]))
    progress = lambda X_: problem.residuals(X_, np.zeros((1, 2)))[1][-1]  # noqa: E731
    for i in range(3):
        Xp, Xm = X.copy(), X.copy()
        Xp[1, i] += EPS
        Xm[1, i] -= EPS
        fd = (progress(Xp) - progress(Xm)) / (2 * EPS)
        assert JxN[-1, i] == pytest.approx(fd, abs=1e-7), f"state {i}"
    th = np.linspace(0.0, np.pi / 2, 400)
    arc = make_centerline(np.column_stack([10 * np.sin(th), 10 - 10 * np.cos(th)]), np.full(400, 10.0),
                          a_lat_ref=3.0, b_ref=1.5)  # left turn, radius 10 m
    p = RouteProblem(arc, (0, len(arc.s) - 1), N=1, s_target=100.0, track_offset=0.0)
    a = np.linspace(0.3, 1.2, 2000)
    Xo = np.column_stack([11 * np.sin(a), 10 - 11 * np.cos(a), a, np.full_like(a, 5.0), 0 * a, 0 * a])  # 1 m outside
    assert all(np.any(p._progress(x)[1] != 0.0) for x in Xo)


def test_progress_is_arc_length_where_a_chord_cuts_a_corner():
    """Arc length runs along the input polyline, so a segment across its corner is shorter than its ds. Progress
    interpolates s over the segment, so it meets cl.s at both ends, as the planner's own s does."""
    cl = make_centerline(np.array([[0.0, 0.0], [10.25, 0.0], [15.25, 5.0]]), np.full(3, 10.0), a_lat_ref=3.0,
                         b_ref=1.5)  # a 45 degree corner a quarter of a sample past s = 10
    i = int(np.argmax(cl.s[1:] - cl.s[:-1] - cl.seg_len))
    assert cl.s[i] == 10.0 and cl.s[i + 1] - cl.s[i] - cl.seg_len[i] > 0.03
    problem = RouteProblem(cl, (0, len(cl.s) - 1), N=1, s_target=0.0, track_offset=0.0, w_progress=1.0)
    t = cl.seg_t[i]
    x = np.array([*(cl.xy[i] + 0.999 * cl.seg_len[i] * t), np.arctan2(t[1], t[0]), 5.0, 0.0, 0.0])
    assert problem._progress(x)[0] == pytest.approx(cl.s[i] + 0.999 * (cl.s[i + 1] - cl.s[i]), abs=1e-9)
