#!/usr/bin/env bash
# Closed-loop run of a planner on a committed scenario split. Runs inside the container:
#   docker/run.sh scripts/sim.sh <tuning|heldout> <nonreactive|reactive> [extra Hydra overrides...]
#   docker/run.sh env PLANNER=idm_planner|pdm_closed_planner scripts/sim.sh ...   baselines (default: ilqr_planner)
#   (set inside the container: docker/run.sh does not forward the host environment)
# Per-tick planner logs (iLQR only) go to $NUPLAN_EXP_ROOT/ilqr_logs/<experiment>/, simulator output to the usual
# experiment dir. The experiment name starts with ilqr, idm or pdm, then the split: eval/compare.py relies on that.
set -euo pipefail
cd "$(dirname "$0")/.."

split="${1:?split: tuning | heldout}"
agents="${2:?agents: nonreactive | reactive}"
shift 2
planner="${PLANNER:-ilqr_planner}"

searchpath="pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments,file://$PWD/config"
if [ "$planner" = pdm_closed_planner ]; then
  searchpath="pkg://tuplan_garage.planning.script.config.common,pkg://tuplan_garage.planning.script.config.simulation,$searchpath"
fi

experiment="${planner%%_*}_${split}_${agents}_$(git rev-parse --short HEAD 2>/dev/null || echo nogit)_$(date +%Y%m%d_%H%M%S)"
export ILQR_LOG_DIR="$NUPLAN_EXP_ROOT/ilqr_logs/$experiment"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"  # threads are pure overhead on 6x6 matrices

# exit_on_failure: the devkit default (false) silently drops a scenario whose planner raised from the aggregate.
"${PYTHON:-python}" "${DEVKIT:-external/nuplan-devkit}/nuplan/planning/script/run_simulation.py" \
  "+simulation=closed_loop_${agents}_agents" \
  "planner=${planner}" \
  scenario_builder=nuplan_mini \
  "scenario_filter=${split}" \
  "hydra.searchpath=[${searchpath}]" \
  "experiment_name=${experiment}" \
  worker=sequential \
  exit_on_failure=true \
  "$@"
[ "$planner" = ilqr_planner ] && echo "planner logs: $ILQR_LOG_DIR"
echo "experiment: $experiment"
