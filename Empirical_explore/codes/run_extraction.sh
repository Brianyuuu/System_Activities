#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

"$PROJECT_DIR/.venv/bin/python" "$SCRIPT_DIR/extract_dependency_phrases.py" \
  --input-dir "$PROJECT_DIR/10k" \
  --output-dir "$PROJECT_DIR/dependency_phrases" \
  --model en_core_web_sm
