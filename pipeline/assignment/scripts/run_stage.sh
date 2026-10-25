#!/usr/bin/env bash
set -euo pipefail
umask 077
BUNDLE_ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
RUN_DIR=$(cd "${1:?Usage: run_stage.sh RUN_DIR offline|glm|gemini|smoke}" && pwd)
STAGE=${2:?Specify offline, glm, gemini, or smoke}
case "$STAGE" in offline|glm|gemini|smoke) ;; *) echo "Unknown stage" >&2; exit 2 ;; esac
cd "$RUN_DIR"
command -v "$PYTHON_BIN" >/dev/null
"$PYTHON_BIN" -c 'import sys; sys.exit(0 if __debug__ else "Python -O is not supported")'
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKER_TARGET=2
if [[ "$STAGE" != smoke ]]; then
WORKER_TARGET=$("$PYTHON_BIN" -c 'import json; print(json.load(open("work/pipeline/glm_worker_target.json"))["workers"])')
fi
export GLM_FILTER_WORKERS=${GLM_FILTER_WORKERS:-$WORKER_TARGET}
export GLM_FILTER_INITIAL_WORKERS=${GLM_FILTER_INITIAL_WORKERS:-$GLM_FILTER_WORKERS}
export GLM_FILTER_RAMP_SECONDS=0 GLM_TRACK_COSTS=0 GLM_DEFER_TRANSPORT=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$OMP_NUM_THREADS
if [[ "$STAGE" == smoke ]]; then
  : "${PIPELINE_SECRET_FILE:?Set an absolute path to an authorized credentials JSON}"
  exec 9>.smoke_submission.lock
  flock -n 9
  exec "$PYTHON_BIN" "$BUNDLE_ROOT/scripts/glm_smoke.py" --output-dir "$RUN_DIR/result"
fi
test -f config.json
exec 9>work/.bundle_submission.lock
flock -n 9
"$PYTHON_BIN" work/run_glm_only.py --offline
if [[ "$STAGE" == offline ]]; then exit 0; fi
: "${PIPELINE_SECRET_FILE:?Set an absolute path to an authorized credentials JSON}"
if [[ "$STAGE" == glm ]]; then
  "$PYTHON_BIN" work/run_glm_only.py
  "$PYTHON_BIN" work/validate_glm_complete.py
else
  "$PYTHON_BIN" work/validate_glm_complete.py
  "$PYTHON_BIN" work/run_reciprocal.py
  "$PYTHON_BIN" work/export_assignments.py
fi
