#!/usr/bin/env bash
# Run a command in the project container. The container sees only PROJECT_ROOT: the repo (rw), the dataset (ro)
# and the experiment outputs (rw). No network, HOME in a throwaway tmpfs, files owned by the calling user.
# Usage: docker/run.sh [command...]     (default: bash)
#   CPUSET=0-3 docker/run.sh ...        pin to cores, required for runtime measurements (PRD R6)
#   REPO=$PROJECT_ROOT/proto docker/run.sh ...   mount a scratch copy of the repo instead (prototyping)
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/nuplan-ilqr-mpc}"
tty=(-i); [ -t 0 ] && [ -t 1 ] && tty=(-it)
pin=(); [ -n "${CPUSET:-}" ] && pin=(--cpuset-cpus "$CPUSET")

exec docker run --rm "${tty[@]}" "${pin[@]}" \
  --network none \
  --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:exec,size=16g -e HOME=/tmp \
  --shm-size 8g --memory "${MEMORY:-48g}" \
  -v "${REPO:-$PROJECT_ROOT/repo}:/work" \
  -v "$PROJECT_ROOT/data:/data/nuplan:ro" \
  -v "$PROJECT_ROOT/exp:/data/exp" \
  "${IMAGE:-nuplan-ilqr-mpc}" "${@:-bash}"
