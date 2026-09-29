"""nuPlan planner: receding-horizon iLQR along the route centerline (P3: free driving, other agents ignored)."""
from __future__ import annotations

import csv
import json
import math
import os
from typing import List, Optional, Type

import numpy as np

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

from planner import bicycle
from planner.costs import TRACK_OFFSET, RouteProblem
from planner.ilqr import solve
from planner.route import build_centerline, progress_target, project

N = 40  # knots: a 4 s plan at bicycle.DT
TICK_US = int(bicycle.DT * 1e6)
PROGRESS_ACCEL = 1.5  # [m/s^2] how fast the progress target lets the car gain speed toward v_ref
LOG_FIELDS = ["tick", "cold", "status", "iters", "solve_ms", "cost", "x", "y", "psi", "v", "a", "kappa", "s", "v_ref",
              "e_y", "e_psi", "center_offset", "err_lon", "err_lat", "err_psi", "err_v"]


class ILQRPlanner(AbstractPlanner):
    """Every 0.1 s: solve RouteProblem from the ego's current state, warm-started from the previous plan shifted one
    knot, and hand the 41 planned rear-axle poses to the simulator's two-stage controller."""

    def __init__(
        self,
        a_lat_ref: float = 3.0,
        b_ref: float = 1.5,
        v_default: float = 15.0,
        max_iters: int = 10,
        max_iters_cold: int = 20,
        budget_s: Optional[float] = None,
        budget_cold_s: Optional[float] = None,
        log_dir: Optional[str] = None,
        weights: Optional[dict] = None,
    ) -> None:
        self._a_lat_ref, self._b_ref, self._v_default = a_lat_ref, b_ref, v_default
        self._max_iters, self._max_iters_cold = max_iters, max_iters_cold
        # None = no wall-clock cap. Scored runs use iteration caps only, so a score depends on the algorithm, not on
        # numpy speed or on how busy the shared machine is, and a rerun reproduces it (spec C8).
        self._budget_s = math.inf if budget_s is None else budget_s
        self._budget_cold_s = math.inf if budget_cold_s is None else budget_cold_s
        self._log_dir = log_dir or os.environ.get("ILQR_LOG_DIR")
        self._weights = dict(weights or {})

    def name(self) -> str:
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks  # type: ignore

    def initialize(self, initialization: PlannerInitialization) -> None:
        self._map_api = initialization.map_api
        self._route_ids = list(initialization.route_roadblock_ids)
        self._cl = None
        self._s = 0.0  # ego arc length along the centerline, tracked tick to tick
        self._U: Optional[np.ndarray] = None
        self._X: Optional[np.ndarray] = None
        self._t_us: Optional[int] = None
        self._log_path: Optional[str] = None
        self._tick = 0

    def compute_planner_trajectory(self, current_input: PlannerInput) -> InterpolatedTrajectory:
        ego: EgoState = current_input.history.current_state[0]
        x0 = np.array([ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading,
                       ego.dynamic_car_state.rear_axle_velocity_2d.x,
                       ego.dynamic_car_state.rear_axle_acceleration_2d.x,
                       bicycle.kappa_from_steering(ego.tire_steering_angle)])
        cl = self._centerline(ego)

        # Where the ego is along the centerline: search the whole route on the first tick, then just around the last
        # position (a tick moves the car a few metres), so a route that loops back cannot capture the ego.
        lo, hi = (0, len(cl.s) - 1) if self._t_us is None else cl.segment_window(self._s - 10.0, self._s + 30.0)
        point = x0[:2] + self._weights.get("track_offset", TRACK_OFFSET) * np.array([math.cos(x0[2]), math.sin(x0[2])])
        seg, f = project(cl, lo, hi, point[None], x0[2:3])  # the same point RouteProblem tracks
        self._s = float(cl.s[seg[0]] + f[0] * (cl.s[seg[0] + 1] - cl.s[seg[0]]))
        v_max = float(np.max(cl.v_ref[lo:]))
        problem = RouteProblem(cl, cl.segment_window(self._s - 10.0, self._s + max(x0[3], v_max) * N * bicycle.DT + 20.0),
                               N=N, s_target=progress_target(cl, self._s, x0[3], N, bicycle.DT, PROGRESS_ACCEL),
                               **self._weights)

        # Log timestamps jitter around 0.1 s, so a step within 20 ms of it counts as the next tick.
        warm = self._U is not None and abs(ego.time_us - self._t_us - TICK_US) <= TICK_US // 5
        err = self._tracking_error(x0) if warm else [math.nan] * 4
        if warm:
            # The tracker realizes poses only; the plant's own acceleration and steering lag and differ from the plan's.
            # At 13 m/s a 0.004 1/m curvature mismatch bends a 4 s rollout by 5 m, so carry the plan's a and kappa.
            x0[4:] = self._X[1, 4:]
        U_init = np.vstack([self._U[1:], self._U[-1:]]) if warm else np.zeros((N, bicycle.NU))
        sol = solve(problem, x0, U_init,
                    max_iters=self._max_iters if warm else self._max_iters_cold,
                    budget_s=self._budget_s if warm else self._budget_cold_s)
        self._U, self._X, self._t_us = sol.U, sol.X, ego.time_us

        r, _ = problem.residuals(sol.X, sol.U)
        e = r[0, :3] / problem.s_track  # knot 0: e_y, e_psi, e_v
        self._write_log(ego, [self._tick, int(not warm), sol.status, sol.iters, 1e3 * sol.solve_time_s, sol.cost,
                              *x0, self._s, x0[3] - e[2], e[0], e[1], self._center_offset(cl, ego), *err])
        self._tick += 1

        params = ego.car_footprint.vehicle_parameters
        states: List[EgoState] = [
            EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(x[0], x[1], x[2]),
                rear_axle_velocity_2d=StateVector2D(x[3], 0.0),
                rear_axle_acceleration_2d=StateVector2D(x[4], 0.0),
                tire_steering_angle=bicycle.steering_from_kappa(x[5]),
                time_point=TimePoint(ego.time_us + k * TICK_US),
                vehicle_parameters=params,
            )
            for k, x in enumerate(sol.X)
        ]
        return InterpolatedTrajectory(states)

    def _centerline(self, ego: EgoState):
        if self._cl is None:
            self._cl, self._route_info = build_centerline(
                self._map_api, self._route_ids, ego, a_lat_ref=self._a_lat_ref, b_ref=self._b_ref,
                v_default=self._v_default)
        return self._cl

    def _tracking_error(self, x0: np.ndarray) -> List[float]:
        """Ego now vs knot 1 of the previous plan (where the plan said it would be): longitudinal and lateral error in
        the planned pose's frame, heading error, speed error."""
        p = self._X[1]
        d = x0[:2] - p[:2]
        c, s = math.cos(p[2]), math.sin(p[2])
        return [c * d[0] + s * d[1], -s * d[0] + c * d[1],
                math.remainder(x0[2] - p[2], 2 * math.pi), x0[3] - p[3]]

    def _center_offset(self, cl, ego: EgoState) -> float:
        """Signed lateral offset of the vehicle's geometric center from the centerline (the rear axle is what tracks)."""
        c = np.array([[ego.center.x, ego.center.y]])
        seg, _ = project(cl, *cl.segment_window(self._s - 10.0, self._s + 10.0), c, np.array([ego.center.heading]))
        t = cl.seg_t[seg[0]]
        return float(np.dot(c[0] - cl.xy[seg[0]], [-t[1], t[0]]))

    def _write_log(self, ego: EgoState, row: list) -> None:
        """Append one row per tick. Only the path is kept: the simulator pickles the planner, and open files do not
        pickle."""
        if not self._log_dir:
            return
        if self._log_path is None:
            os.makedirs(self._log_dir, exist_ok=True)
            # Named by the scenario's first timestamp; eval/summarize.py maps it back to the scenario token.
            base = os.path.join(self._log_dir, str(ego.time_us))
            with open(base + ".route.json", "w") as f:
                json.dump(self._route_info, f)
            self._log_path = base + ".csv"
            with open(self._log_path, "w", newline="") as f:
                csv.writer(f, lineterminator="\n").writerow(LOG_FIELDS)
        with open(self._log_path, "a", newline="") as f:
            csv.writer(f, lineterminator="\n").writerow(row)
