#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT_DIR}/.venv/bin/python"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

"${PYTHON}" "${ROOT_DIR}/codes/cluster_activity_taxonomy.py" \
  --input "${ROOT_DIR}/phrases/clean_normalize/activities_medium_cleaned.csv" \
  --dropped-input "${ROOT_DIR}/phrases/clean_normalize/dropped_phrases_medium.csv" \
  --output-dir "${ROOT_DIR}/phrases/clustering"
