#!/usr/bin/env python3
"""Check immutable package files and prompt parity without API calls."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'MANIFEST.json').read_text())
for name, expected in manifest['sha256'].items():
    path = root / name
    assert path.is_file() and not path.is_symlink(), name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, 'Changed file: ' + name
source = root / 'code/production/pipeline_filter.py'
tree = ast.parse(source.read_text())
prompt = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'PROMPT' for t in n.targets))
assert (root / 'prompts/glm_high_recall.txt').read_text() == prompt + '\n'
assert (root / 'prompts/parent_first_prompts.json').read_bytes() == (root / 'code/production/parent_first_prompts.json').read_bytes()
for path in (root / 'code').rglob('*.py'):
    ast.parse(path.read_text())
print(json.dumps({'verified_files': len(manifest['sha256']), 'prompt_parity': True,
                  'api_calls': 0}, indent=2))
