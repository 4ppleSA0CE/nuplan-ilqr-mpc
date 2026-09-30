"""Summarize one closed-loop run of the iLQR planner: scores, the P3 gate, and the planner's per-tick logs.

    docker/run.sh python eval/summarize.py <experiment_name>

<experiment_name> is the name scripts/sim.sh prints (ilqr_<split>_<mode>_<commit>_<time>). Every number printed here
comes from that run's files; the script fails if any scenario of the split is missing, so a planner crash cannot
silently drop a scenario from the average.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from eval.scenarios import listed, load

EXP = os.environ["NUPLAN_EXP_ROOT"]


def main(experiment: str) -> None:
    split = experiment.split("_")[1]
    rows = [r for r in listed() if r["split"] == split]
    tokens = {r["token"] for r in rows}

    parquet = glob.glob(f"{EXP}/exp/{experiment}/*/*/aggregator_metric/*.parquet")
    assert len(parquet) == 1, f"expected one aggregator file for {experiment}, found {parquet}"
    df = pd.read_parquet(parquet[0])
    final = df[df.scenario == "final_score"].iloc[0]
    per = df[df.scenario.isin(tokens)]
    missing = tokens - set(per.scenario)
    assert not missing, f"{len(missing)} scenarios of the {split} split have no score: {sorted(missing)[:5]}..."

    starts = {s.initial_ego_state.time_us: s.token for s in load(sorted(tokens), sorted({r["log_name"] for r in rows}))}
    log_dir = f"{EXP}/ilqr_logs/{experiment}"
    logs = {starts[int(os.path.basename(f)[:-4])]: pd.read_csv(f) for f in glob.glob(f"{log_dir}/*.csv")}
    assert set(logs) == tokens, f"planner logs missing for {len(tokens - set(logs))} scenarios"
    routes = [json.load(open(f)) for f in glob.glob(f"{log_dir}/*.route.json")]

    n = len(per)
    print(f"# {experiment}: {n} scenarios")
    print(f"score (CLS) {100 * final.score:.2f}")
    for m in ("no_ego_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
              "ego_is_making_progress", "ego_is_comfortable"):
        print(f"  {m}: {int((per[m] == 1).sum())}/{n} = {100 * (per[m] == 1).mean():.1f}%")
    for m in ("ego_progress_along_expert_route", "speed_limit_compliance", "time_to_collision_within_bound"):
        print(f"  {m}: mean {per[m].mean():.3f}")
    gate = (per.drivable_area_compliance == 1).mean()
    print(f"P3 gate (drivable-area multiplier = 1 on >= 95%): {100 * gate:.1f}% -> {'PASS' if gate >= 0.95 else 'FAIL'}")
    off = per[per.drivable_area_compliance < 1]
    for _, r in off.iterrows():
        print(f"  off-road: {r.scenario} ({r.scenario_type})")

    print("\nper scenario type: n, mean score, drivable-area passes")
    for t, g in per.groupby("scenario_type"):
        print(f"  {t:58s} n={len(g):2d}  score {100 * g.score.mean():5.1f}  drivable {int((g.drivable_area_compliance == 1).sum())}/{len(g)}")

    print("\nroute handling:", {k: sum(bool(r[k]) for r in routes) for k in ("on_route", "joined", "complete", "extended")},
          f"| lanes without a speed limit in {sum(r['no_limit'] > 0 for r in routes)} routes")

    ticks = pd.concat(logs.values())
    for label, t in (("cold", ticks[ticks.cold == 1]), ("warm", ticks[ticks.cold == 0])):
        print(f"\n{label} ticks: {len(t)}; status {t.status.value_counts().to_dict()}")
        print(f"  iterations median {t.iters.median():.0f}, p99 {t.iters.quantile(0.99):.0f}; "
              f"solve ms median {t.solve_ms.median():.1f}, p99 {t.solve_ms.quantile(0.99):.1f} (numpy, this run's load)")
    warm = ticks[ticks.cold == 0]
    print("\ntracking error, ego now vs knot 1 of the previous plan (warm ticks), |.| median / p95 / p99:")
    for c in ("err_lon", "err_lat", "err_psi", "err_v"):
        a = warm[c].abs()
        print(f"  {c}: {a.median():.2e} / {a.quantile(0.95):.2e} / {a.quantile(0.99):.2e}")
    print(f"vehicle center to centerline |.|: p99 {ticks.center_offset.abs().quantile(0.99):.2f} m, "
          f"max {ticks.center_offset.abs().max():.2f} m")
    print(f"ticks with a solve ending in 'budget': {int((ticks.status == 'budget').sum())} (0 means the run is deterministic)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else sys.exit(__doc__))
