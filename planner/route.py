"""Route centerline: resampling, speed reference, projection. No nuPlan import."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

DS = 0.5  # [m] resampling step
KAPPA_WINDOW_M = 5.0  # [m] curvature smoothing and widening window, for the speed reference
MIN_ROUTE_M = 1.0  # [m] shorter routes are rejected: they leave fewer than 3 samples


@dataclass
class Centerline:
    """Piecewise-linear centerline, M >= 3 samples. Segment i joins samples i and i+1."""

    s: np.ndarray  # (M,) arc length along the cleaned input polyline
    xy: np.ndarray  # (M, 2)
    psi: np.ndarray  # (M,) unwrapped heading
    kappa: np.ndarray  # (M,) unsigned curvature, smoothed then widened; for the speed reference only
    v_limit: np.ndarray  # (M,) lane speed limit
    v_ref: np.ndarray  # (M,) speed reference
    seg_t: np.ndarray = field(init=False)  # (M-1, 2) unit tangents
    seg_len: np.ndarray = field(init=False)  # (M-1,) chord lengths

    def __post_init__(self) -> None:
        d = np.diff(self.xy, axis=0)
        self.seg_len = np.linalg.norm(d, axis=1)
        if np.any(self.seg_len <= 0.0):
            raise ValueError("centerline has a zero-length segment")
        self.seg_t = d / self.seg_len[:, None]

    def segment_window(self, s_lo: float, s_hi: float) -> Tuple[int, int]:
        """Segments [lo, hi) that overlap arc length [s_lo, s_hi]. Always at least one segment."""
        lo = int(np.clip(np.searchsorted(self.s, s_lo, side="right") - 1, 0, len(self.s) - 2))
        hi = int(np.clip(np.searchsorted(self.s, s_hi, side="left"), lo + 1, len(self.s) - 1))
        return lo, hi


def kept_indices(points: np.ndarray) -> np.ndarray:
    """Indices of the input points to keep: drop a point within 1 cm of the last kept one, or one that steps
    backward along the last kept direction. Lanes share end points, and a lane can start slightly behind where the
    previous one ended; either would otherwise make a zero-length or reversed segment."""
    kept = [0]
    for i in range(1, len(points)):
        d = points[i] - points[kept[-1]]
        if np.hypot(*d) < 0.01:
            continue
        if len(kept) >= 2 and np.dot(d, points[kept[-1]] - points[kept[-2]]) <= 0.0:
            continue
        kept.append(i)
    return np.asarray(kept)


def resample(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Polyline (K, 2) -> s (M,), xy (M, 2) at arc-length spacing DS along the kept points, last gap in
    (DS/2, 1.5 DS]; and src (M,) int, the index into points of the last kept point at or before each sample."""
    points = np.asarray(points, dtype=np.float64)
    k = kept_indices(points)
    pts = points[k]
    s_in = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    length = s_in[-1]
    if length < MIN_ROUTE_M:
        raise ValueError(f"route is {length:.3f} m long; at least {MIN_ROUTE_M} m is needed")
    s = np.arange(0.0, length, DS)
    if length - s[-1] < DS / 2:
        s = s[:-1]
    s = np.append(s, length)
    xy = np.column_stack([np.interp(s, s_in, pts[:, 0]), np.interp(s, s_in, pts[:, 1])])
    src = k[np.clip(np.searchsorted(s_in, s, side="right") - 1, 0, len(k) - 1)]
    return s, xy, src


def headings(xy: np.ndarray) -> np.ndarray:
    """(M, 2) -> (M,) unwrapped heading: mean of the adjacent segment headings, one-sided at the ends."""
    d = np.diff(xy, axis=0)
    theta = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    return np.concatenate([[theta[0]], 0.5 * (theta[:-1] + theta[1:]), [theta[-1]]])


def curvature(xy: np.ndarray) -> np.ndarray:
    """(M, 2) -> (M,) unsigned curvature for the speed reference.

    Raw: turning angle between adjacent segments over their mean length, copied to the two end samples. Then a
    centered moving average over KAPPA_WINDOW_M, so a kink at a lane join is spread out instead of braking the car
    to walking pace. The average cuts the curvature by up to half over the first and last half-window of a curve,
    so a centered moving maximum over twice that width follows it, and the curve speed holds on the whole arc.
    """
    d = np.diff(xy, axis=0)
    h = np.linalg.norm(d, axis=1)
    theta = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    inner = np.diff(theta) / (0.5 * (h[:-1] + h[1:]))
    kappa = np.abs(np.concatenate([inner[:1], inner, inner[-1:]]))
    n = max(1, int(round(KAPPA_WINDOW_M / np.median(h)))) | 1  # odd width keeps both filters centered
    avg = np.convolve(np.pad(kappa, n // 2, mode="edge"), np.ones(n) / n, mode="valid")
    w = 2 * n - 1  # twice the average's reach, so every sample of a curve sees an average taken fully inside it
    return sliding_window_view(np.pad(avg, w // 2, mode="edge"), w).max(axis=1)


def speed_reference(
    s: np.ndarray, kappa: np.ndarray, v_limit: np.ndarray, a_lat_ref: float, b_ref: float
) -> np.ndarray:
    """min(limit, curve speed), braking at b_ref into curves and to 0 at the route end."""
    v = np.minimum(v_limit, np.sqrt(a_lat_ref / np.maximum(kappa, 1e-6)))
    v[-1] = 0.0
    for i in range(len(s) - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2.0 * b_ref * (s[i + 1] - s[i])))
    return v


def make_centerline(points: np.ndarray, v_limit_pts: np.ndarray, *, a_lat_ref: float, b_ref: float) -> Centerline:
    """Polyline (K, 2) with a speed limit per input point (K,) -> Centerline. Each sample takes the limit of the
    last kept input point at or before it, so a limit changes where the lane that carries it starts."""
    s, xy, src = resample(points)
    v_limit = np.asarray(v_limit_pts, dtype=np.float64)[src]
    kappa = curvature(xy)
    return Centerline(s, xy, headings(xy), kappa, v_limit, speed_reference(s, kappa, v_limit, a_lat_ref, b_ref))


def progress_target(cl: Centerline, s0: float, v0: float, n: int, dt: float, a_up: float) -> float:
    """Arc length reached after n steps of dt from s0 at speed v0, following v_ref but gaining speed at no more than
    a_up, so the target never asks for more than a comfortable acceleration can deliver. Capped at the route end."""
    s, v = s0, v0
    for _ in range(n):
        v = min(float(np.interp(s, cl.s, cl.v_ref)), v + a_up * dt)
        s += max(v, 0.0) * dt
    return min(s, float(cl.s[-1]))


def project(cl: Centerline, lo: int, hi: int, P: np.ndarray, psi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest segment in [lo, hi) for each point, among segments pointing within 90 degrees of the point's heading.

    P (K, 2), psi (K,) -> seg (K,) int segment index, f (K,) fraction along it in [0, 1].
    A point that faces away from every segment in the window falls back to the plain nearest segment.
    """
    c, t, h = cl.xy[lo:hi], cl.seg_t[lo:hi], cl.seg_len[lo:hi]
    d = P[:, None, :] - c[None, :, :]  # (K, W, 2)
    f = np.clip(np.einsum("kwi,wi->kw", d, t) / h, 0.0, 1.0)
    dist2 = np.sum((d - f[..., None] * h[None, :, None] * t[None, :, :]) ** 2, axis=2)
    facing = np.cos(psi)[:, None] * t[None, :, 0] + np.sin(psi)[:, None] * t[None, :, 1] > 0.0  # |wrap| < pi/2
    j = np.where(facing.any(axis=1), np.argmin(np.where(facing, dist2, np.inf), axis=1), np.argmin(dist2, axis=1))
    return lo + j, f[np.arange(len(P)), j]



EXTEND_M = 300.0  # [m] a broken or early-ending route is extended by following the road to this far ahead of the ego
JOIN_DEPTH = 6  # lanes searched from the ego's lane to reach the route when the ego is not on it


def build_centerline(map_api, route_roadblock_ids, ego_state, *, a_lat_ref: float, b_ref: float, v_default: float):
    """nuPlan route -> (Centerline, info). nuPlan route lists are noisy: some start several roadblocks behind the ego,
    some have gaps, and a few never come near the ego at all. So:

    1. Start on the lane (or lane connector) under the ego center whose heading is closest to the ego's.
    2. If that lane's roadblock is on the route, drop the roadblocks before it. Otherwise search JOIN_DEPTH lanes
       ahead for one that is, and prepend the lanes that lead there.
    3. Breadth-first search through the remaining route's lanes to its last roadblock, as IDMPlanner does. If it
       cannot get there, keep the longest route found.
    4. If the route was not completed and ends less than EXTEND_M ahead of the ego, follow the road from its last
       lane, taking the successor that turns least.

    info: on_route, joined, complete, extended (bool), no_limit (lanes without a speed limit), lanes (ids used)."""
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer  # nuPlan stays out of the module import
    from nuplan.planning.simulation.planner.utils.breadth_first_search import BreadthFirstSearch

    layers = (SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR)
    point, heading = ego_state.center.point, ego_state.center.heading

    def heading_gap(lane) -> float:
        return abs(math.remainder(lane.baseline_path.get_nearest_pose_from_position(point).heading - heading, 2 * math.pi))

    under = [o for layer in layers for o in map_api.get_all_map_objects(point, layer)]
    if under:
        start = min(under, key=heading_gap)
    else:
        near = [o for objs in map_api.get_proximal_map_objects(point, 10.0, list(layers)).values() for o in objs]
        if not near:
            raise ValueError("no lane within 10 m of the ego")
        foot = ego_state.car_footprint.geometry
        start = min(near, key=lambda o: (heading_gap(o) > math.pi / 2, o.polygon.distance(foot)))

    ids = list(route_roadblock_ids)
    prefix = _path_to_route(start, set(ids)) if start.get_roadblock_id() not in ids else [start]
    lanes, complete = [start], False
    if prefix:
        j = ids.index(prefix[-1].get_roadblock_id())
        blocks = [map_api.get_map_object(i, SemanticMapLayer.ROADBLOCK)
                  or map_api.get_map_object(i, SemanticMapLayer.ROADBLOCK_CONNECTOR) for i in ids[j:]]
        blocks = [b for b in blocks if b is not None]
        found, complete = BreadthFirstSearch(prefix[-1], [e.id for b in blocks for e in b.interior_edges]).search(
            blocks[-1], len(blocks))
        lanes = prefix[:-1] + found

    def centerline(lanes_):
        pts = [(p.x, p.y) for lane in lanes_ for p in lane.baseline_path.discrete_path]
        lim = [lane.speed_limit_mps or v_default for lane in lanes_ for _ in lane.baseline_path.discrete_path]
        return make_centerline(np.array(pts), np.array(lim), a_lat_ref=a_lat_ref, b_ref=b_ref)

    cl = centerline(lanes)
    ahead = cl.s[-1] - cl.s[int(np.argmin(np.linalg.norm(cl.xy - np.array([point.x, point.y]), axis=1)))]
    extended = bool(not complete and ahead < EXTEND_M)
    if extended:
        length = ahead
        while length < EXTEND_M and lanes[-1].outgoing_edges:
            end = lanes[-1].baseline_path.discrete_path[-1].heading
            nxt = min(lanes[-1].outgoing_edges,
                      key=lambda e: abs(math.remainder(e.baseline_path.discrete_path[0].heading - end, 2 * math.pi)))
            lanes.append(nxt)
            length += nxt.baseline_path.length
        cl = centerline(lanes)
    info = dict(on_route=start.get_roadblock_id() in ids, joined=len(prefix) > 1, complete=bool(complete),
                extended=extended, no_limit=sum(1 for lane in lanes if not lane.speed_limit_mps),
                lanes=[lane.id for lane in lanes])
    return cl, info


def _path_to_route(start, route_ids):
    """Shortest lane path from start (inclusive) to a lane whose roadblock is in route_ids, at most JOIN_DEPTH lanes
    ahead; [] if there is none."""
    frontier, seen = [[start]], {start.id}
    for _ in range(JOIN_DEPTH):
        nxt = []
        for path in frontier:
            for e in path[-1].outgoing_edges:
                if e.id in seen:
                    continue
                if e.get_roadblock_id() in route_ids:
                    return path + [e]
                seen.add(e.id)
                nxt.append(path + [e])
        frontier = nxt
    return []
