#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

: "${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY in the environment}"

"$PROJECT_DIR/.venv/bin/python" "$SCRIPT_DIR/extract_llm_activities.py" \
  --input-dir "$PROJECT_DIR/10k" \
  --output-dir "$PROJECT_DIR/llm_phrases" \
  --audit-dir "$PROJECT_DIR/llm_phrases/audit" \
  --model "${DEEPSEEK_MODEL:-deepseek-v4-pro}"
