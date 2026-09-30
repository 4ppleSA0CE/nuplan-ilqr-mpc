"""build_centerline start-lane choice, joins and extension, and ILQRPlanner ticks, on mocked map objects: needs the
devkit installed, not the dataset."""
import csv
import dataclasses
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

pytest.importorskip("nuplan")

from nuplan.common.actor_state.ego_state import EgoState  # noqa: E402
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint  # noqa: E402
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters  # noqa: E402
from nuplan.common.maps.maps_datatypes import SemanticMapLayer  # noqa: E402
from shapely.geometry import LineString  # noqa: E402

import planner.ilqr_planner as ip  # noqa: E402
from planner import bicycle, route  # noqa: E402
from planner.costs import TRACK_OFFSET  # noqa: E402
from planner.route import build_centerline  # noqa: E402

PARAMS = get_pacifica_parameters()


class Path_:
    """baseline_path: poses with the heading of the chord that leaves each point."""

    def __init__(self, pts):
        pts = np.asarray(pts, dtype=np.float64)
        d = np.diff(pts, axis=0)
        h = np.arctan2(d[:, 1], d[:, 0])
        self.pts = pts
        self.discrete_path = [NS(x=p[0], y=p[1], heading=hh) for p, hh in zip(pts, np.append(h, h[-1]))]
        self.length = float(np.sum(np.linalg.norm(d, axis=1)))

    def get_nearest_pose_from_position(self, pt):
        return self.discrete_path[int(np.argmin(np.linalg.norm(self.pts - [pt.x, pt.y], axis=1)))]


class Lane:
    def __init__(self, id, rb, pts, limit=10.0):
        self.id, self.rb, self.speed_limit_mps = id, rb, limit
        self.baseline_path = Path_(pts)
        self.polygon = LineString(pts).buffer(1.75)
        self.outgoing_edges = []

    def get_roadblock_id(self):
        return self.rb


class Map:
    """Lanes under the ego and within 10 m are given, not computed: the tests pick which polygons contain it."""

    def __init__(self, lanes, under, near=()):
        self.under, self.near, self.blocks = under, list(near), {}
        for lane in lanes:
            self.blocks.setdefault(lane.rb, NS(id=lane.rb, interior_edges=[])).interior_edges.append(lane)

    def get_all_map_objects(self, point, layer):
        return list(self.under) if layer == SemanticMapLayer.LANE else []

    def get_proximal_map_objects(self, point, radius, layers):
        return {SemanticMapLayer.LANE: self.near}

    def get_map_object(self, i, layer):
        return self.blocks.get(i) if layer == SemanticMapLayer.ROADBLOCK else None


def line(x0, y0, x1, y1, n=41):
    return np.column_stack([np.linspace(x0, x1, n), np.linspace(y0, y1, n)])


def right_turn(x0=0.0, radius=15.0):
    """Quarter circle starting at (x0, 0) heading east, ending at (x0 + radius, -radius) heading south."""
    th = np.linspace(0.0, math.pi / 2, 41)
    return np.column_stack([x0 + radius * np.sin(th), -radius + radius * np.cos(th)])


def ego(x, y, psi, v=0.0, a=0.0, steer=0.0, t_us=0, center=False):
    """EgoState at a rear-axle pose, or at a center pose with center=True."""
    if center:
        x, y = x - PARAMS.rear_axle_to_center * math.cos(psi), y - PARAMS.rear_axle_to_center * math.sin(psi)
    return EgoState.build_from_rear_axle(StateSE2(x, y, psi), StateVector2D(v, 0.0), StateVector2D(a, 0.0), steer,
                                         TimePoint(t_us), PARAMS)


def build(m, ids, e):
    return build_centerline(m, ids, e, a_lat_ref=3.0, b_ref=1.5, v_default=15.0)


def test_start_lane_skips_oncoming_and_prefers_the_route():
    # Ego heading east, center 0.2 m over the divider: only the westbound lane's polygon contains it. It must start
    # on the eastbound lane, found by the within-10 m fallback.
    west = Lane("W", "RBW", line(100, 0.2, -100, 0.2))
    east = Lane("E", "RBE", line(-100, -3.3, 100, -3.3))
    cl, info = build(Map([west, east], [west], near=[west, east]), ["RBE"], ego(0.0, 0.0, 0.0, center=True))
    assert info["lanes"][0] == "E" and info["on_route"] and abs(cl.psi[0]) < 1e-9

    # Intersection entry: straight (on route) and right-turn (off route) connectors share their start pose. The ego,
    # 1 m in and yawed 0.08 rad right, is closer in heading to the turn, but the route decides.
    straight, right = Lane("S", "RC_S", line(0, 0, 30, 0)), Lane("R", "RC_R", right_turn())
    after, south = Lane("A", "RB_A", line(30, 0, 400, 0)), Lane("D", "RB_D", line(15, -15, 15, -400))
    straight.outgoing_edges, right.outgoing_edges = [after], [south]
    lanes = [straight, right, after, south]
    e = ego(1.0, 0.0, -0.08, center=True)
    for under in ([straight, right], [right, straight]):
        _, info = build(Map(lanes, under), ["RC_S", "RB_A"], e)
        assert info["lanes"] == ["S", "A"] and info["on_route"] and info["complete"]
    # Neither connector on the route: the closest heading wins, and the join goes down the turn to reach it.
    _, info = build(Map(lanes, [straight, right]), ["RB_D"], e)
    assert info["lanes"] == ["R", "D"] and info["joined"] and not info["on_route"]


def test_join_extension_and_default_speed_limit():
    # The ego's lane is two lanes short of the route: joined within JOIN_DEPTH, then the route is searched to its end.
    l0, l1 = Lane("L0", "X0", line(0, 0, 20, 0)), Lane("L1", "X1", line(20, 0, 40, 0))
    l2, l3 = Lane("L2", "R1", line(40, 0, 60, 0)), Lane("L3", "R2", line(60, 0, 400, 0))
    l0.outgoing_edges, l1.outgoing_edges, l2.outgoing_edges = [l1], [l2], [l3]
    _, info = build(Map([l0, l1, l2, l3], [l0]), ["R1", "R2"], ego(5.0, 0.0, 0.0))
    assert info["lanes"] == ["L0", "L1", "L2", "L3"] and info["joined"] and info["complete"] and not info["extended"]

    # The route breaks 20 m ahead (its next roadblock is not reachable): follow the road, taking the successor that
    # turns least. That one has no speed limit.
    e0 = Lane("E0", "R0", line(0, 0, 20, 0), limit=8.0)
    left = Lane("TL", "X", right_turn(20.0) * [1.0, -1.0])
    ahead = Lane("TS", "X", line(20, 0, 400, 0, n=381), limit=None)
    e0.outgoing_edges = [left, ahead, Lane("TR", "X", right_turn(20.0))]
    far = Lane("Z", "R9", line(0, 500, 100, 500))
    cl, info = build(Map([e0, far, *e0.outgoing_edges], [e0]), ["R0", "R9"], ego(5.0, 0.0, 0.0))
    assert info["lanes"] == ["E0", "TS"] and info["extended"] and not info["complete"] and info["no_limit"] == 1
    assert np.all(cl.v_limit[cl.s < 19.0] == 8.0) and np.all(cl.v_limit[cl.s > 22.0] == 15.0)  # v_default


def test_join_search_visits_each_lane_once_on_a_cyclic_graph():
    """Six lanes, each leading to all the others, none on the route: without the visited set the search would
    expand 5^k paths at depth k."""
    expanded = []

    class Counting(Lane):
        @property
        def outgoing_edges(self):
            expanded.append(self.id)
            return self._out

        @outgoing_edges.setter
        def outgoing_edges(self, v):
            self._out = v

    lanes = [Counting(f"C{i}", "X", line(0, 0, 10, 0)) for i in range(6)]
    for lane in lanes:
        lane.outgoing_edges = [o for o in lanes if o is not lane]
    assert route._path_to_route(lanes[0], {"R"}) == []
    assert len(expanded) == len(set(expanded)) <= 6


def test_steering_curvature_and_track_offset_match_the_pacifica():
    assert bicycle.WHEEL_BASE == PARAMS.wheel_base and TRACK_OFFSET == PARAMS.rear_axle_to_center
    for delta in (-0.6, -0.1, 0.0, 0.3, 1.0):
        kappa = bicycle.kappa_from_steering(delta)
        assert kappa == pytest.approx(math.tan(delta) / PARAMS.wheel_base, abs=1e-15)
        assert bicycle.steering_from_kappa(kappa) == pytest.approx(delta, abs=1e-12)


def test_planner_ticks(tmp_path, monkeypatch):
    """Trajectory shape and timing, warm vs cold ticks, carry-over of the plan's a and kappa, the standstill clamp,
    and a cold solve after a non-finite one. The per-tick log records what the solver started from."""
    lane = Lane("L", "R", line(-10, 0, 400, 0, n=411))
    planner = ip.ILQRPlanner(log_dir=str(tmp_path))
    planner.initialize(NS(map_api=Map([lane], [lane]), route_roadblock_ids=["R"]))
    force_status = []
    real_solve = ip.solve

    def solve(*args, **kwargs):
        sol = real_solve(*args, **kwargs)
        return dataclasses.replace(sol, status=force_status.pop()) if force_status else sol

    monkeypatch.setattr(ip, "solve", solve)

    def tick(t_us, v=5.0, a=0.4, steer=0.02):
        e = ego(5.0 * t_us * 1e-6, 0.0, 0.0, v=v, a=a, steer=steer, t_us=t_us)
        traj = planner.compute_planner_trajectory(NS(history=NS(current_state=(e, None))))
        with open(next(tmp_path.glob("*.csv"))) as f:
            return traj, list(csv.DictReader(f))[-1]

    t0 = 1_000_000
    traj, row = tick(t0, steer=0.3)
    states = traj.get_sampled_trajectory()
    assert len(states) == 41 and [s.time_us for s in states] == [t0 + 100_000 * k for k in range(41)]
    assert states[0].tire_steering_angle == pytest.approx(0.3, abs=1e-12)  # steering -> kappa -> steering
    assert row["cold"] == "1" and float(row["kappa"]) == pytest.approx(bicycle.kappa_from_steering(0.3))

    t = t0
    plant = (0.4, bicycle.kappa_from_steering(0.02))
    for dt_us, cold in ((115_000, 0), (85_000, 0), (121_000, 1), (100_000, 0), (79_000, 1), (300_000, 1)):
        t += dt_us
        prev = planner._X[1].copy()
        _, row = tick(t)
        assert row["cold"] == str(cold), dt_us
        # The plant's a and kappa are logged as they are; a warm tick starts from the plan's instead.
        assert (float(row["a_plant"]), float(row["kappa_plant"])) == pytest.approx(plant)
        assert (float(row["a"]), float(row["kappa"])) == pytest.approx(plant if cold else tuple(prev[4:]))

    # Standstill: a carried deceleration is dropped, so the plan does not reverse; above 0.3 m/s it is kept.
    for v, a_start in ((0.1, 0.0), (1.0, -3.0)):
        planner._X[1, 4] = -3.0
        t += 100_000
        traj, row = tick(t, v=v)
        assert row["cold"] == "0" and float(row["a"]) == a_start
        if v < 0.3:
            assert min(s.dynamic_car_state.rear_axle_velocity_2d.x for s in traj.get_sampled_trajectory()) >= 0.0

    # A non-finite solve is no warm start: the next tick is cold even at exactly 0.1 s.
    t += 200_000
    force_status.append("non_finite")
    _, row = tick(t)
    assert row["status"] == "non_finite"
    _, row = tick(t + 100_000)
    assert row["cold"] == "1"
