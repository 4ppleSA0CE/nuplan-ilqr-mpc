"""Route centerline: resampling, speed reference, projection. Only build_centerline() imports nuPlan."""
from __future__ import annotations

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


def resample(points: np.ndarray, ds: float = DS) -> Tuple[np.ndarray, np.ndarray]:
    """Polyline (K, 2) -> s (M,), xy (M, 2) at arc-length spacing ds along the kept points; last gap in
    (ds/2, 1.5 ds]."""
    pts = np.asarray(points, dtype=np.float64)
    pts = pts[kept_indices(pts)]
    s_in = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    length = s_in[-1]
    if length < MIN_ROUTE_M:
        raise ValueError(f"route is {length:.3f} m long; at least {MIN_ROUTE_M} m is needed")
    s = np.arange(0.0, length, ds)
    if length - s[-1] < ds / 2:
        s = s[:-1]
    s = np.append(s, length)
    xy = np.column_stack([np.interp(s, s_in, pts[:, 0]), np.interp(s, s_in, pts[:, 1])])
    return s, xy


def headings(xy: np.ndarray) -> np.ndarray:
    """(M, 2) -> (M,) unwrapped heading: mean of the adjacent segment headings, one-sided at the ends."""
    d = np.diff(xy, axis=0)
    theta = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    return np.concatenate([[theta[0]], 0.5 * (theta[:-1] + theta[1:]), [theta[-1]]])


def curvature(xy: np.ndarray, window_m: float = KAPPA_WINDOW_M) -> np.ndarray:
    """(M, 2) -> (M,) unsigned curvature for the speed reference.

    Raw: turning angle between adjacent segments over their mean length, copied to the two end samples. Then a
    centered moving average over window_m, so a kink at a lane join is spread out instead of braking the car to
    walking pace. The average halves the curvature over the first and last half-window of a curve, so a centered
    moving maximum over twice that width follows it, and the curve speed holds on the whole arc.
    """
    d = np.diff(xy, axis=0)
    h = np.linalg.norm(d, axis=1)
    theta = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    inner = np.diff(theta) / (0.5 * (h[:-1] + h[1:]))
    kappa = np.abs(np.concatenate([inner[:1], inner, inner[-1:]]))
    n = max(1, int(round(window_m / np.median(h)))) | 1  # odd width keeps both filters centered
    avg = np.convolve(np.pad(kappa, n // 2, mode="edge"), np.ones(n) / n, mode="valid")
    w = 2 * n - 1  # twice the average's reach, so every sample of a curve sees an average taken fully inside it
    return sliding_window_view(np.pad(avg, w // 2, mode="edge"), w).max(axis=1)


def speed_reference(
    s: np.ndarray, kappa: np.ndarray, v_limit: np.ndarray, a_lat_ref: float, b_ref: float
) -> np.ndarray:
    """min(limit, curve speed), braking at b_ref into curves and to 0 at the route end."""
    v = np.minimum(v_limit, np.sqrt(a_lat_ref / np.maximum(np.abs(kappa), 1e-6)))
    v[-1] = 0.0
    for i in range(len(s) - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2.0 * b_ref * (s[i + 1] - s[i])))
    return v


def make_centerline(
    points: np.ndarray, v_limit_pts: np.ndarray, *, a_lat_ref: float, b_ref: float, ds: float = DS
) -> Centerline:
    """Polyline (K, 2) with a speed limit per input point (K,) -> Centerline. Each sample takes the limit of the
    last kept input point at or before it, so a limit changes where the lane that carries it starts."""
    points = np.asarray(points, dtype=np.float64)
    s, xy = resample(points, ds)
    k = kept_indices(points)
    s_kept = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points[k], axis=0), axis=1))])
    idx = k[np.clip(np.searchsorted(s_kept, s, side="right") - 1, 0, len(k) - 1)]
    v_limit = np.asarray(v_limit_pts, dtype=np.float64)[idx]
    kappa = curvature(xy)
    return Centerline(s, xy, headings(xy), kappa, v_limit, speed_reference(s, kappa, v_limit, a_lat_ref, b_ref))


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
