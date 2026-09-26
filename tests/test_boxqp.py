"""T2: box-QP against brute force over all active sets, including the ill-conditioned regime where it used to stall."""
import itertools

import numpy as np

from planner.ilqr import boxqp


def _objective(H, g, x):
    return 0.5 * x @ H @ x + g @ x


def _brute_force(H, g, lb, ub):
    """Best objective over all 3^m choices of (at lb, free, at ub) whose free solution is feasible."""
    m = g.shape[0]
    best = np.inf
    for assign in itertools.product((-1, 0, 1), repeat=m):
        assign = np.array(assign)
        if ((assign < 0) & np.isinf(lb)).any() or ((assign > 0) & np.isinf(ub)).any():
            continue
        x = np.where(assign < 0, lb, np.where(assign > 0, ub, 0.0))
        free = assign == 0
        if free.any():
            rhs = -(g[free] + H[np.ix_(free, ~free)] @ x[~free])
            x[free] = np.linalg.solve(H[np.ix_(free, free)], rhs)
        slack = 1e-9 * (1.0 + np.abs(x))
        if np.all(x >= lb - slack) and np.all(x <= ub + slack):
            best = min(best, _objective(H, g, x))
    return best


def _assert_solution(H, g, lb, ub, x, free, L, rel_tol):
    assert np.all(x >= lb) and np.all(x <= ub)
    grad = H @ x + g
    kkt_free = ~(((x <= lb) & (grad > 0)) | ((x >= ub) & (grad < 0)))
    assert np.array_equal(free, kkt_free), "returned free mask does not describe the returned x"
    assert np.allclose(L @ L.T, H[np.ix_(free, free)], rtol=1e-9, atol=0.0)
    f, f_bf = _objective(H, g, x), _brute_force(H, g, lb, ub)
    assert f - f_bf <= rel_tol * abs(f_bf), f"objective {f} vs brute force {f_bf}"


def test_boxqp_matches_brute_force_well_conditioned():
    rng = np.random.default_rng(0)
    n_lower = n_upper = 0
    for i in range(200):
        m = 2 + i % 5  # 2..6
        M = rng.standard_normal((m, m))
        H, g = M @ M.T + 0.1 * m * np.eye(m), 3.0 * rng.standard_normal(m)
        lb, ub = -rng.uniform(0.0, 1.0, m), rng.uniform(0.0, 1.0, m)
        x, free, L, ok = boxqp(H, g, lb, ub, np.zeros(m))
        assert ok
        _assert_solution(H, g, lb, ub, x, free, L, rel_tol=1e-10)
        n_lower += int((x == lb).any())
        n_upper += int((x == ub).any())
    assert n_lower > 50 and n_upper > 50  # pointless unless both kinds of bound are regularly active


def _hard_qp(rng):
    """Condition number up to 1e8, overall scale 1e-2..1e3, sometimes an infinite bound, adversarial warm starts."""
    m = int(rng.integers(1, 7))
    Q, _ = np.linalg.qr(rng.standard_normal((m, m)))
    eig = np.geomspace(1.0, 10 ** rng.uniform(0, 8), m)
    H = (Q * eig) @ Q.T
    H = 0.5 * (H + H.T) * 10 ** rng.uniform(-2, 3)
    g = rng.standard_normal(m) * 10 ** rng.uniform(-1, 2) * np.sqrt(np.abs(H).max())
    lb, ub = -rng.uniform(0.05, 1.0, m), rng.uniform(0.05, 1.0, m)
    if rng.random() < 0.25:
        (lb if rng.random() < 0.5 else ub)[rng.integers(m)] *= np.inf
    kind = rng.integers(3)
    if kind == 0:
        x0 = np.zeros(m)
    elif kind == 1:  # a corner of the box
        x0 = np.where(rng.random(m) < 0.5, np.where(np.isfinite(lb), lb, -1.0), np.where(np.isfinite(ub), ub, 1.0))
    else:  # far outside the box
        x0 = 5.0 * rng.standard_normal(m)
    return H, g, lb, ub, x0


def test_boxqp_hard_regime_never_fails_or_lies():
    """Backtracking along the clipped arc (the first implementation) returned ok=False on about 4% of these."""
    rng = np.random.default_rng(12345)
    for _ in range(1500):
        H, g, lb, ub, x0 = _hard_qp(rng)
        x, free, L, ok = boxqp(H, g, lb, ub, x0)
        assert ok, f"ok=False on a solvable convex QP: m={len(g)} cond={np.linalg.cond(H):.1e}"
        _assert_solution(H, g, lb, ub, x, free, L, rel_tol=1e-6)


def test_boxqp_near_singular_corner_start_regression():
    """Found by review: backtracking along the clipped arc never converged here."""
    H = np.array([[51675146.34509835, 37823601.90319422], [37823601.90319422, 27684971.238153756]])
    g = np.array([-0.4173522630499486, 0.12443469075900306])
    lb = np.array([-0.5468678014205923, -0.25102398715283614])
    ub = np.array([0.9862291615252186, 0.6638545996488094])
    x, free, L, ok = boxqp(H, g, lb, ub, np.array([lb[0], ub[1]]))
    assert ok
    _assert_solution(H, g, lb, ub, x, free, L, rel_tol=1e-8)


def test_boxqp_is_scale_invariant():
    """iLQR regularization can make H ~1e10. An absolute gradient tolerance fails there and feeds back into reg."""
    rng = np.random.default_rng(1)
    M = rng.standard_normal((3, 3))
    H, g, one = M @ M.T + 0.3 * np.eye(3), 3.0 * rng.standard_normal(3), np.ones(3)
    x_ref, _, _, ok = boxqp(H, g, -one, one, np.zeros(3))
    assert ok
    for c in (1e-6, 1e10):
        x, _, _, ok = boxqp(c * H, c * g, -one, one, np.zeros(3))
        assert ok and np.allclose(x, x_ref, rtol=0, atol=1e-12)


def test_boxqp_unbounded_is_newton_step():
    rng = np.random.default_rng(2)
    M = rng.standard_normal((3, 3))
    H, g, inf = M @ M.T + 0.3 * np.eye(3), rng.standard_normal(3), np.full(3, np.inf)
    x, free, _, ok = boxqp(H, g, -inf, inf, np.zeros(3))
    assert ok and free.all()
    assert np.allclose(x, -np.linalg.solve(H, g), atol=1e-10)


def test_boxqp_reports_failure_on_indefinite_free_block():
    H = np.array([[1.0, 0.0], [0.0, -1.0]])
    _, _, _, ok = boxqp(H, np.ones(2), -np.ones(2), np.ones(2), np.zeros(2))
    assert not ok
