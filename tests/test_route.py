"""C1-C4, C3b: centerline resampling, projection, speed reference, loop-back guards. No nuPlan."""
import numpy as np
import pytest

from planner import route
from planner.costs import wrap_angle
from planner.route import make_centerline, project, resample

A_LAT, B_REF = 3.0, 1.5


def _polyline_length(p):
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def _irregular(s_max, rng, lo=0.3, hi=1.1):
    """Irregular arc-length samples in [0, s_max], spacing lo-hi m, with the end point included."""
    s = np.cumsum(rng.uniform(lo, hi, int(s_max / lo) + 2))
    return np.concatenate([[0.0], s[s < s_max], [s_max]])


def _arc(radius, s):
    """Counter-clockwise arc around the origin starting at (radius, 0), heading +y."""
    th = s / radius
    return np.column_stack([radius * np.cos(th), radius * np.sin(th)])


def _straight_arc_straight(radius=15.0, arc_len=20.0, straight=100.0, ds_in=0.7):
    """East for `straight` m, left arc of `arc_len` m, then `straight` m in the new direction."""
    a = np.column_stack([np.arange(0.0, straight, ds_in), np.zeros(int(np.ceil(straight / ds_in)))])
    th = np.arange(0.0, arc_len, ds_in) / radius
    b = np.column_stack([straight + radius * np.sin(th), radius - radius * np.cos(th)])
    th_end = arc_len / radius
    end = np.array([straight + radius * np.sin(th_end), radius - radius * np.cos(th_end)])
    heading = np.array([np.cos(th_end), np.sin(th_end)])
    c = end + np.arange(0.0, straight + 1e-9, ds_in)[:, None] * heading
    return np.vstack([a, b, c])


# Rigid transforms of a test route: the result must not depend on turn side or on where the heading crosses +-pi.
# "rotated" starts at heading pi - 0.3, so a left turn crosses +-pi.
_ROT = np.pi - 0.3
TRANSFORMS = {
    "as_is": lambda p: p,
    "mirrored": lambda p: p * np.array([1.0, -1.0]),
    "rotated": lambda p: p @ np.array([[np.cos(_ROT), np.sin(_ROT)], [-np.sin(_ROT), np.cos(_ROT)]]),
}


def test_c1_resample_spacing_length_and_duplicates():
    rng = np.random.default_rng(1)
    line, x = _irregular(37.3, rng), _irregular(80.0, rng)
    cases = {
        "straight": np.column_stack([line, 0.5 * line]),
        "arc": _arc(12.0, _irregular(30.0, rng)),
        "s_curve": np.column_stack([x, 3.0 * np.sin(x / 15.0)]),
    }
    for name, pts in cases.items():
        pts = np.insert(pts, 3, pts[3], axis=0)  # a duplicate point, as where two lanes join
        s, xy = resample(pts)
        gaps = np.diff(s)
        assert np.allclose(gaps[:-1], route.DS, atol=1e-9), name
        assert route.DS / 2 < gaps[-1] <= 1.5 * route.DS + 1e-9, name
        assert s[-1] == pytest.approx(_polyline_length(pts), rel=1e-6), name
        assert np.all(np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-3), name
        assert np.allclose(xy[0], pts[0]) and np.allclose(xy[-1], pts[-1]), name


def test_c1_lane_join_that_steps_backward_is_dropped():
    lane_a = np.column_stack([np.arange(0.0, 50.01, 1.0), np.zeros(51)])
    lane_b = np.column_stack([np.arange(49.7, 150.0, 1.0), np.zeros(101)])  # starts 0.3 m behind lane a's end
    pts = np.vstack([lane_a, lane_b])
    cl = make_centerline(pts, np.full(len(pts), 10.0), a_lat_ref=A_LAT, b_ref=B_REF)
    assert np.allclose(cl.psi, 0.0)
    assert np.all(cl.kappa < 1e-9)
    assert np.min(cl.v_ref[cl.s < 100.0]) == 10.0  # no dip at the join (braking for the route end starts at 117 m)


def test_c1_too_short_route_is_rejected():
    with pytest.raises(ValueError):
        make_centerline(np.array([[0.0, 0.0], [0.6, 0.0]]), np.full(2, 10.0), a_lat_ref=A_LAT, b_ref=B_REF)
    with pytest.raises(ValueError):
        make_centerline(np.zeros((4, 2)), np.full(4, 10.0), a_lat_ref=A_LAT, b_ref=B_REF)


def test_c2_projection_on_an_arc():
    radius = 40.0
    # Fine input spacing: the input polyline's own sagitta would otherwise dominate (1.1 m chords: 3.8e-3 m).
    # 100 m of arc turns 2.5 rad from heading pi/2, so the heading crosses +-pi at th = pi/2.
    pts = _arc(radius, _irregular(100.0, np.random.default_rng(2), lo=0.05, hi=0.1))
    cl = make_centerline(pts, np.full(len(pts), 10.0), a_lat_ref=A_LAT, b_ref=B_REF)
    th = np.linspace(0.2, 2.3, 421)  # every 0.2 m along the arc, away from both ends, across the +-pi crossing
    for offset in (-1.0, 0.0, 0.4, 1.5):  # positive offset = toward the center = left of travel
        r = radius - offset
        P = np.column_stack([r * np.cos(th), r * np.sin(th)])
        psi = th + np.pi / 2  # the arc's true tangent at each point
        seg, f = project(cl, 0, len(cl.s) - 1, P, psi)
        e_y = np.einsum("ki,ki->k", P - cl.xy[seg], np.column_stack([-cl.seg_t[seg, 1], cl.seg_t[seg, 0]]))
        assert np.allclose(e_y, offset, atol=1e-3)  # chord sagitta at ds = 0.5, R = 40 is 7.8e-4 m
        e_psi = wrap_angle(psi - (cl.psi[seg] + f * (cl.psi[seg + 1] - cl.psi[seg])))
        assert np.allclose(e_psi, 0.0, atol=1e-3)  # sample headings must sit at the samples, not half a step off
        s_star = cl.s[seg] + f * (cl.s[seg + 1] - cl.s[seg])
        assert np.all(np.diff(s_star) > 0)


@pytest.mark.parametrize("name", TRANSFORMS)
def test_c3_speed_reference(name):
    radius, v_lim = 15.0, 15.0  # braking 15 -> 6.7 m/s at 1.5 m/s^2 takes 60 m, so the straight is 100 m
    pts = TRANSFORMS[name](_straight_arc_straight(radius=radius))
    cl = make_centerline(pts, np.full(len(pts), v_lim), a_lat_ref=A_LAT, b_ref=B_REF)
    v = cl.v_ref
    assert np.all(v <= cl.v_limit + 1e-12)
    assert v[-1] == 0.0
    decel = (v[:-1] ** 2 - v[1:] ** 2) / (2.0 * np.diff(cl.s))
    assert np.max(decel) <= B_REF + 1e-9
    on_arc = (cl.s >= 100.0) & (cl.s <= 120.0)
    assert np.all(v[on_arc] ** 2 / radius <= A_LAT * (1 + 1e-9))  # curve speed holds on the whole arc
    mid_arc = np.argmin(np.abs(cl.s - 110.0))
    assert v[mid_arc] == pytest.approx(np.sqrt(A_LAT * radius), rel=0.05)
    assert v[np.argmin(np.abs(cl.s - 5.0))] == v_lim  # far enough before the curve to be at the limit


def test_c3_speed_limit_changes_where_the_next_lane_starts():
    x = np.arange(0.0, 200.01, 1.0)
    limit = np.where(x < 50.0, 15.0, 8.0)  # the lane starting at x = 50 has a lower limit
    cl = make_centerline(np.column_stack([x, 0 * x]), limit, a_lat_ref=A_LAT, b_ref=B_REF)
    assert np.all(cl.v_limit[cl.s < 50.0] == 15.0) and np.all(cl.v_limit[cl.s >= 50.0] == 8.0)
    assert np.all(cl.v_ref <= cl.v_limit)


def test_c3b_route_starting_in_a_curve_is_slowed_from_the_first_sample():
    radius = 15.0  # starts mid-curve: curvature must not be lost at the route ends
    arc = _arc(radius, np.arange(0.0, 30.0, 0.2))
    heading = np.array([-np.sin(30.0 / radius), np.cos(30.0 / radius)])
    pts = np.vstack([arc, arc[-1] + np.arange(0.2, 60.0, 0.2)[:, None] * heading])
    cl = make_centerline(pts, np.full(len(pts), 15.0), a_lat_ref=A_LAT, b_ref=B_REF)
    assert cl.v_ref[0] == pytest.approx(np.sqrt(A_LAT * radius), rel=0.05)


def _hairpin():
    """East along y = 0 for 30 m, left U-turn of radius 1.5, west along y = 3: two branches 3 m apart."""
    a = np.column_stack([np.linspace(0.0, 30.0, 61), np.zeros(61)])
    th = np.linspace(0.0, np.pi, 20)[1:-1]
    b = np.column_stack([30.0 + 1.5 * np.sin(th), 1.5 - 1.5 * np.cos(th)])
    c = np.column_stack([np.linspace(30.0, 0.0, 61), np.full(61, 3.0)])
    return np.vstack([a, b, c])


def _block_loop():
    """East along y = 0, around a block, and east again along y = 2: same direction, 2 m apart, ~140 m later."""
    a = np.column_stack([np.linspace(0.0, 40.0, 81), np.zeros(81)])
    th = np.linspace(0.0, np.pi, 40)[1:-1]
    b = np.column_stack([40.0 + 11.0 * np.sin(th), 11.0 - 11.0 * np.cos(th)])
    c = np.column_stack([np.linspace(40.0, 0.0, 81), np.full(81, 22.0)])
    d = np.column_stack([-10.0 * np.sin(th), 12.0 + 10.0 * np.cos(th)])
    e = np.column_stack([np.linspace(0.0, 40.0, 81), np.full(81, 2.0)])
    return np.vstack([a, b, c, d, e])


def _plain_nearest(cl, lo, hi, p):
    """Nearest segment ignoring heading: what project() would return without the gate."""
    c, t, h = cl.xy[lo:hi], cl.seg_t[lo:hi], cl.seg_len[lo:hi]
    f = np.clip(np.einsum("wi,wi->w", p - c, t) / h, 0.0, 1.0)
    return lo + int(np.argmin(np.linalg.norm(p - (c + (f * h)[:, None] * t), axis=1)))


@pytest.mark.parametrize(
    "p, psi, branch_y",
    [
        ((10.0, 1.8), 0.0, 0.0),  # eastbound, nearer the westbound branch
        ((10.0, 1.2), np.pi - 0.01, 3.0),  # westbound, nearer the eastbound branch; heading just below +pi
        ((10.0, 1.2), -np.pi + 0.01, 3.0),  # the same heading written just above -pi
    ],
)
def test_c4_heading_gate_picks_the_branch_going_our_way(p, psi, branch_y):
    pts = _hairpin()
    cl = make_centerline(pts, np.full(len(pts), 10.0), a_lat_ref=A_LAT, b_ref=B_REF)
    whole = (0, len(cl.s) - 1)
    seg, _ = project(cl, *whole, np.array([p]), np.array([psi]))
    assert cl.xy[seg[0], 1] == pytest.approx(branch_y)
    assert cl.xy[_plain_nearest(cl, *whole, np.array(p)), 1] != pytest.approx(branch_y)  # the case needs the gate


def test_c4_facing_away_from_the_whole_window_falls_back_to_nearest():
    pts = _hairpin()
    cl = make_centerline(pts, np.full(len(pts), 10.0), a_lat_ref=A_LAT, b_ref=B_REF)
    lo, hi = cl.segment_window(0.0, 25.0)  # eastbound segments only
    p = np.array([10.0, 0.4])
    seg, f = project(cl, lo, hi, p[None], np.array([np.pi]))  # facing west
    assert seg[0] == _plain_nearest(cl, lo, hi, p) and 0.0 <= f[0] <= 1.0


def test_c4_window_keeps_a_knot_off_a_later_pass():
    pts = _block_loop()
    cl = make_centerline(pts, np.full(len(pts), 10.0), a_lat_ref=A_LAT, b_ref=B_REF)
    P, psi = np.array([[10.0, 1.2]]), np.array([0.0])  # 1.2 m from the first pass, 0.8 m from the later one
    seg, _ = project(cl, *cl.segment_window(0.0, 60.0), P, psi)
    assert abs(cl.xy[seg[0], 1]) < 1e-9  # first pass
    seg, _ = project(cl, 0, len(cl.s) - 1, P, psi)  # window off: the later pass wins
    assert cl.xy[seg[0], 1] == pytest.approx(2.0)


def test_segment_window_bounds():
    x = np.arange(0.0, 10.01, 0.5)
    cl = make_centerline(np.column_stack([x, 0 * x]), np.full(len(x), 10.0), a_lat_ref=A_LAT, b_ref=B_REF)
    assert cl.segment_window(-5.0, 100.0) == (0, len(cl.s) - 1)
    assert cl.segment_window(2.2, 3.1) == (4, 7)  # segments [2.0,2.5] ... [3.0,3.5]
    lo, hi = cl.segment_window(50.0, 60.0)  # past the end: still one segment
    assert hi - lo == 1
