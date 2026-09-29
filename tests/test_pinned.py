"""Pinned reference vectors: guard the numpy solver against drift and feed the P7 C++ parity test.

Regenerate (only when a solver change is intended): PINNED_REGEN=1 .venv/bin/python -m pytest tests/test_pinned.py
"""
import os
from pathlib import Path

import numpy as np
import pytest
from helpers import random_lqr, reg_path_case, toy_case

from planner import bicycle
from planner.ilqr import solve

PINNED = Path(__file__).parent / "pinned"
TOLS = dict(tol_cost=1e-6, tol_step=1e-4)  # passed explicitly and stored, so a C++ port needs no Python defaults


def _cases():
    lqr, x0 = random_lqr(seed=0)
    yield "lqr_unbounded", lqr, x0, np.zeros((lqr.N, lqr.m)), 10, dict(A=lqr.A, B=lqr.B, Qs=lqr.Qs, Rs=lqr.Rs)
    toy, x0, U0 = toy_case()
    toy_params = dict(
        ref=toy.ref, s_track=toy.s_track, s_effort=toy.s_effort, s_limit=toy.s_limit,
        a_min=toy.a_min, a_max=toy.a_max, kappa_max=toy.kappa_max, dt=bicycle.DT, wheel_base=bicycle.WHEEL_BASE,
    )
    yield "toy_active_bounds", toy, x0, U0, 50, toy_params
    yield "toy_max_iters", toy, x0, U0, 3, toy_params
    hard, x0, U0 = reg_path_case()  # the only case with reg > 0 and rejected line searches in its trace
    hard_params = dict(toy_params, s_track=hard.s_track, s_effort=hard.s_effort, s_limit=hard.s_limit)
    yield "toy_reg_path", hard, x0, U0, 60, hard_params


CASES = list(_cases())


@pytest.mark.parametrize("name,problem,x0,U_init,max_iters,params", CASES, ids=[c[0] for c in CASES])
def test_pinned(name, problem, x0, U_init, max_iters, params):
    sol = solve(problem, x0, U_init, max_iters=max_iters, budget_s=np.inf, **TOLS)
    path = PINNED / f"{name}.npz"
    if os.environ.get("PINNED_REGEN"):
        PINNED.mkdir(exist_ok=True)
        np.savez(
            path, x0=x0, U_init=U_init, max_iters=max_iters, u_lb=problem.u_lb, u_ub=problem.u_ub,
            X=sol.X, U=sol.U, cost=sol.cost, iters=sol.iters, status=sol.status, trace=np.array(sol.trace),
            **TOLS, **params,
        )
    ref = np.load(path)
    assert (str(ref["status"]), int(ref["iters"])) == (sol.status, sol.iters), f"{name}: pinned {ref['status']}/{ref['iters']}"
    for key, got in (("X", sol.X), ("U", sol.U)):
        drift = np.abs(ref[key] - got).max()
        assert drift <= 1e-10, f"{name}: {key} drifted by {drift:.2e}"
    assert abs(float(ref["cost"]) - sol.cost) <= 1e-10 * abs(sol.cost), f"{name}: cost {sol.cost} vs pinned {float(ref['cost'])}"
    assert np.allclose(ref["trace"], np.array(sol.trace), rtol=1e-9, atol=1e-12), f"{name}: trace drifted"
