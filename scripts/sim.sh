#!/usr/bin/env bash
# Closed-loop run of the iLQR planner on a committed scenario split. Runs inside the container:
#   docker/run.sh scripts/sim.sh <tuning|heldout> <nonreactive|reactive> [extra Hydra overrides...]
# Per-tick planner logs go to $NUPLAN_EXP_ROOT/ilqr_logs/<experiment>/, simulator output to the usual experiment dir.
set -euo pipefail
cd "$(dirname "$0")/.."

split="${1:?split: tuning | heldout}"
agents="${2:?agents: nonreactive | reactive}"
shift 2

experiment="ilqr_${split}_${agents}_$(git rev-parse --short HEAD 2>/dev/null || echo nogit)_$(date +%Y%m%d_%H%M%S)"
export ILQR_LOG_DIR="$NUPLAN_EXP_ROOT/ilqr_logs/$experiment"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"  # threads are pure overhead on 6x6 matrices

# exit_on_failure: the devkit default (false) silently drops a scenario whose planner raised from the aggregate.
"${PYTHON:-python}" "${DEVKIT:-external/nuplan-devkit}/nuplan/planning/script/run_simulation.py" \
  "+simulation=closed_loop_${agents}_agents" \
  planner=ilqr_planner \
  scenario_builder=nuplan_mini \
  "scenario_filter=${split}" \
  "hydra.searchpath=[pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments,file://$PWD/config]" \
  "experiment_name=${experiment}" \
  worker=sequential \
  exit_on_failure=true \
  "$@"
echo "planner logs: $ILQR_LOG_DIR"
