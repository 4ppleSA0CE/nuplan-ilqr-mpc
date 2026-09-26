"""Timing for the numpy solver. Run by hand from the repo root:

    PYTHONPATH=. .venv/bin/python tests/bench_ilqr.py

Cold start = U_init of zeros, as on the first tick. Warm start = the converged plan shifted one knot, solved from
the next tick's state. Two versions: from the plan's own predicted state (a perfect tracker and plant, so a lower
bound on the work) and from that state plus a tracking error. The error size here is an assumption, not a
measurement of nuPlan's tracker; phase 3 measures the real one.
"""
import os

# OpenBLAS threads on 6x6 matrices roughly double the median solve time and blow up the tail (measured).
# Must be set before numpy is imported.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from helpers import arc_reference  # noqa: E402

from planner.costs import TrackingProblem  # noqa: E402
from planner.ilqr import solve  # noqa: E402


def _stats(label: str, sols) -> None:
    ms = [1e3 * s.solve_time_s for s in sols]
    iters = [s.iters for s in sols]
    statuses = {}
    for s in sols:
        statuses[s.status] = statuses.get(s.status, 0) + 1
    print(
        f"{label:<28} time median={np.median(ms):6.1f} ms  p99={np.percentile(ms, 99):6.1f} ms | "
        f"iters median={np.median(iters):4.0f}  p99={np.percentile(iters, 99):4.0f} | {statuses}"
    )


def main(n_runs: int = 200) -> None:
    rng = np.random.default_rng(0)
    ref = arc_reference(N=41)
    now, nxt = TrackingProblem(ref[:-1]), TrackingProblem(ref[1:])
    track_sd = np.array([0.02, 0.05, 0.005, 0.05, 0.0, 0.0])  # x [m], y [m], heading [rad], speed [m/s]
    capped, full, warm, warm_err, excess = [], [], [], [], []
    for _ in range(n_runs):
        x0 = np.array(
            [0.0, rng.uniform(-1.5, 1.5), rng.uniform(-0.2, 0.2), ref[0, 3] + rng.uniform(-3.0, 3.0), 0.0, 0.0]
        )
        U0 = np.zeros((now.N, now.m))
        capped.append(solve(now, x0, U0, max_iters=10, budget_s=np.inf))
        full.append(solve(now, x0, U0, max_iters=200, budget_s=np.inf))
        excess.append((capped[-1].cost - full[-1].cost) / full[-1].cost)
        U_shift = np.vstack([full[-1].U[1:], full[-1].U[-1:]])
        warm.append(solve(nxt, full[-1].X[1], U_shift, max_iters=10, budget_s=np.inf))
        x_err = full[-1].X[1] + track_sd * rng.standard_normal(6)
        warm_err.append(solve(nxt, x_err, U_shift, max_iters=10, budget_s=np.inf))
    print(f"runs={n_runs}  N={now.N}")
    _stats("cold, max_iters=10", capped)
    _stats("cold, to convergence", full)
    _stats("warm, perfect tracking", warm)
    _stats("warm, tracking error", warm_err)
    print("tracking error sd: 2 cm x, 5 cm y, 5 mrad heading, 0.05 m/s (assumed)")
    print(f"cost excess of the 10-iteration cold solve: median={100 * np.median(excess):.1f}%  p99={100 * np.percentile(excess, 99):.1f}%")


if __name__ == "__main__":
    main()
