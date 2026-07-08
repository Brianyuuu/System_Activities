#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

"$PROJECT_DIR/.venv/bin/python" "$SCRIPT_DIR/clean_normalize_medium.py" \
  --comparison-dir "$PROJECT_DIR/phrases/comparison" \
  --output-dir "$PROJECT_DIR/phrases/clean_normalize" \
  --spacy-model en_core_web_sm \
  --embedding-model sentence-transformers/all-mpnet-base-v2
