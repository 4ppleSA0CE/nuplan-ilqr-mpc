"""C7: build_centerline on every scenario in eval/scenarios.csv, on the real maps. Needs the nuPlan data, so it runs
in the Thor container (docker/run.sh python -m pytest tests/test_centerline_maps.py -s) and skips elsewhere."""
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("nuplan")
MINI = Path(os.environ.get("NUPLAN_DATA_ROOT", "/nonexistent")) / "nuplan-v1.1/splits/mini"
pytestmark = pytest.mark.skipif(not any(MINI.glob("*.db")), reason="nuPlan mini split not present")

REACH_M = 250.0  # [m] a 15 s scenario cannot drive further than this, so only this much route is checked
MAX_EGO_OFFSET_M = 3.5  # [m] one lane width: the ego must start in the lane the centerline follows


def test_centerline_on_every_listed_scenario():
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    from eval.scenarios import listed, load
    from planner.route import DS, build_centerline

    rows = listed()
    scenarios = load([r["token"] for r in rows], sorted({r["log_name"] for r in rows}))
    assert len(scenarios) == len(rows)
    failures, kinds = [], Counter()
    for s in scenarios:
        ego = s.initial_ego_state
        cl, info = build_centerline(s.map_api, s.get_route_roadblock_ids(), ego, a_lat_ref=3.0, b_ref=1.5,
                                    v_default=15.0)
        kinds.update(k for k in ("on_route", "joined", "complete", "extended") if info[k])
        rear = np.array([ego.rear_axle.x, ego.rear_axle.y])
        k0 = int(np.argmin(np.linalg.norm(cl.xy - rear, axis=1)))
        reach = cl.xy[k0:][cl.s[k0:] - cl.s[k0] <= REACH_M]
        problems = []
        if np.linalg.norm(np.diff(cl.xy, axis=0), axis=1).max() > 1.5 * DS:
            problems.append("gap in the centerline")
        if np.abs(np.diff(cl.psi)).max() >= 0.2:
            problems.append("heading jumps by 0.2 rad or more between samples")
        if np.linalg.norm(cl.xy[k0] - rear) > MAX_EGO_OFFSET_M:
            problems.append("ego starts more than a lane width from the centerline")
        if any(not s.map_api.is_in_layer(Point2D(*p), SemanticMapLayer.DRIVABLE_AREA) for p in reach):
            problems.append(f"centerline leaves the drivable area within {REACH_M:.0f} m")
        if problems:
            failures.append(f"{s.token} ({s.scenario_type}): {'; '.join(problems)}")
    print(f"\nC7: {len(scenarios)} scenarios; route handling: {dict(kinds)}")
    assert not failures, "\n".join(failures)
