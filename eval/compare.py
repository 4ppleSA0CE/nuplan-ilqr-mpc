"""Compare closed-loop runs scenario by scenario: mean score, multiplier passes, and a paired bootstrap 95% CI of
each run's mean score minus the first run's (PRD section 4).

    docker/run.sh python eval/compare.py <reference_experiment> <experiment> [<experiment> ...]

Experiment names are the ones scripts/sim.sh prints (<planner>_<split>_<mode>_<commit>_<time>). All runs must cover
every scenario of the same split; the script fails otherwise. The devkit's final score is the scenario-count-weighted
mean of the per-type means, which is the plain per-scenario mean, so the CI is on the same number (asserted).
"""
import glob
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd

MULTIPLIERS = ("no_ego_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
               "ego_is_making_progress")


def paired_bootstrap(ref: np.ndarray, other: np.ndarray, n: int = 10000, seed: int = 0) -> Tuple[float, float, float]:
    """Paired per-scenario scores -> (mean of other - ref, 2.5th and 97.5th percentile of the resampled mean)."""
    d = np.asarray(other, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def load(experiment: str) -> pd.DataFrame:
    """Per-scenario rows of one run's aggregator file, indexed by token, checked against the committed split."""
    from eval.scenarios import listed  # nuPlan import stays out of paired_bootstrap's tests

    split = experiment.split("_")[1]
    tokens = {r["token"] for r in listed() if r["split"] == split}
    files = glob.glob(f"{os.environ['NUPLAN_EXP_ROOT']}/exp/{experiment}/*/*/aggregator_metric/*.parquet")
    assert len(files) == 1, f"expected one aggregator file for {experiment}, found {files}"
    df = pd.read_parquet(files[0])
    per = df[df.scenario.isin(tokens)].set_index("scenario")
    assert set(per.index) == tokens, f"{experiment}: {len(tokens - set(per.index))} scenarios of {split} have no score"
    final = float(df[df.scenario == "final_score"].iloc[0].score)
    assert abs(per.score.mean() - final) < 1e-9, f"{experiment}: final score {final} is not the per-scenario mean"
    return per


def main(experiments) -> None:
    runs = [load(e) for e in experiments]
    ref = runs[0]
    assert all(set(r.index) == set(ref.index) for r in runs), "runs cover different scenarios"
    n = len(ref)
    for name, r in zip(experiments, runs):
        r = r.loc[ref.index]
        passes = "  ".join(f"{m.split('_')[-1]} {int((r[m] == 1).sum())}/{n}" for m in MULTIPLIERS)
        line = f"{name}: CLS {100 * r.score.mean():.2f}  {passes}  comfortable {int((r.ego_is_comfortable == 1).sum())}/{n}"
        if r is not ref:
            d, lo, hi = paired_bootstrap(ref.score.values, r.score.values)
            line += f"  vs ref {100 * d:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]"
        print(line)


if __name__ == "__main__":
    main(sys.argv[1:]) if len(sys.argv) > 2 else sys.exit(__doc__)
