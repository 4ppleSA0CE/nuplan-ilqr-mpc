#!/usr/bin/env bash
# Stage 1 install check: one closed-loop run of a stock planner on nuPlan mini.
# Usage: scripts/stage1_sim.sh <simple_planner|idm_planner> <nonreactive|reactive>
set -euo pipefail
cd "$(dirname "$0")/.."

export NUPLAN_DATA_ROOT="${NUPLAN_DATA_ROOT:-$HOME/nuplan/dataset}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$NUPLAN_DATA_ROOT/maps}"
export NUPLAN_EXP_ROOT="${NUPLAN_EXP_ROOT:-$HOME/nuplan/exp}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"  # threads are pure overhead on 6x6 matrices

planner="${1:?planner: simple_planner | idm_planner}"
agents="${2:?agents: nonreactive | reactive}"

[ -f "$NUPLAN_MAPS_ROOT/nuplan-maps-v1.0.json" ] || { echo "maps missing under $NUPLAN_MAPS_ROOT"; exit 1; }
ls "$NUPLAN_DATA_ROOT"/nuplan-v1.1/splits/mini/*.db >/dev/null 2>&1 || { echo "mini .db files missing under $NUPLAN_DATA_ROOT/nuplan-v1.1/splits/mini"; exit 1; }

# worker=sequential: clean per-scenario wall-clock, and ray's default fan-out is risky on 16 GB.
time "${PYTHON:-.venv/bin/python}" "${DEVKIT:-external/nuplan-devkit}/nuplan/planning/script/run_simulation.py" \
  "+simulation=closed_loop_${agents}_agents" \
  "planner=${planner}" \
  scenario_builder=nuplan_mini \
  scenario_filter=one_of_each_scenario_type \
  worker=sequential
