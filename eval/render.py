"""Bird's-eye GIF of one simulated scenario from its nuPlan simulation log: lanes, other agents, the ego and its plan.

    docker/run.sh python eval/render.py <experiment_name> <token> [<token> ...]

Writes $NUPLAN_EXP_ROOT/renders/<experiment_name>/<token>.gif. Everything drawn comes from the saved log.
"""
import glob
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from nuplan.common.actor_state.state_representation import Point2D  # noqa: E402
from nuplan.common.maps.maps_datatypes import SemanticMapLayer  # noqa: E402
from nuplan.planning.simulation.simulation_log import SimulationLog  # noqa: E402

EXP = os.environ["NUPLAN_EXP_ROOT"]
VIEW_M = 30.0  # half-width of the view around the ego
STEP = 2  # draw every second 0.1 s tick


def render(experiment: str, token: str) -> str:
    path = glob.glob(f"{EXP}/exp/{experiment}/*/*/simulation_log/*/*/*/{token}/{token}.msgpack.xz")[0]
    log = SimulationLog.load_data(Path(path))
    samples = log.simulation_history.data
    map_api = log.scenario.map_api
    ego_xy = [(s.ego_state.center.x, s.ego_state.center.y) for s in samples]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=90)

    def draw(i: int) -> None:
        ax.clear()
        s = samples[i]
        ego = s.ego_state
        cx, cy = ego.center.x, ego.center.y
        near = map_api.get_proximal_map_objects(Point2D(cx, cy), VIEW_M * 1.5,
                                                [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR])
        for layer, objs in near.items():
            for o in objs:
                ax.add_patch(Polygon(list(o.polygon.exterior.coords), closed=True, fc="#e8e8e8", ec="#b0b0b0", lw=0.5))
                b = o.baseline_path.linestring.xy
                ax.plot(b[0], b[1], color="#c8c8c8", lw=0.5, ls="--")
        ax.plot(*zip(*ego_xy[: i + 1]), color="#1f77b4", lw=1.5, alpha=0.6)  # driven so far
        for obj in s.observation.tracked_objects.tracked_objects:
            ax.add_patch(Polygon(list(obj.box.geometry.exterior.coords), closed=True, fc="#999999", ec="#555555", lw=0.8))
        plan = s.trajectory.get_sampled_trajectory()
        ax.plot([p.rear_axle.x for p in plan], [p.rear_axle.y for p in plan], color="#d62728", lw=2.0)
        ax.add_patch(Polygon(list(ego.car_footprint.geometry.exterior.coords), closed=True, fc="#1f77b4", ec="#0b3c61"))
        ax.set_xlim(cx - VIEW_M, cx + VIEW_M)
        ax.set_ylim(cy - VIEW_M, cy + VIEW_M)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{log.scenario.scenario_type}  {token[:8]}   t = {i / 10:4.1f} s   v = {ego.dynamic_car_state.speed:4.1f} m/s",
                     fontsize=9)

    out = f"{EXP}/renders/{experiment}/{token}.gif"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    FuncAnimation(fig, draw, frames=range(0, len(samples), STEP)).save(out, writer=PillowWriter(fps=10 // STEP))  # real time
    plt.close(fig)
    return out


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    for tok in sys.argv[2:]:
        print(render(sys.argv[1], tok))
