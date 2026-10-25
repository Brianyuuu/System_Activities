#!/usr/bin/env python3
"""Run the synthetic verification suite; never invoke live inference or submit jobs."""
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if not __debug__:raise RuntimeError('Run tests without Python -O')
manifest=json.loads((ROOT/'docs/release_manifest.json').read_text())
for name,digest in manifest['sha256'].items():
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
        raise RuntimeError('Distributed file changed: '+name)
print('PASS release manifest')
for base in ['scripts','tests','pipeline']:
    for path in (ROOT/base).rglob('*.py'):ast.parse(path.read_text())
commands=[['pipeline/assignment/scripts/verify_bundle.py'],
          ['pipeline/assignment/tests/test_input_adapter.py'],
          ['pipeline/assignment/tests/test_portable_pipeline.py'],
          ['-m','unittest','discover','-s','tests','-v']]
# Import each executable extraction stage with --help to catch missing packaged dependencies.
for path in sorted((ROOT/'pipeline/extraction/template/runtime/pipeline_source').glob('*.py')):
    if path.name!='item1_boundary_v5_5.py':commands.append([str(path),'--help'])
commands += [['pipeline/extraction/template/pipeline_latest/02_filter_cross_industry_focal.py','--help'],
             ['pipeline/extraction/template/code/strict_runtime/05_decompose_mechanism_phrases.py','--help']]
for command in commands:
    result=subprocess.run([sys.executable,*command],cwd=ROOT,text=True,capture_output=True)
    if result.returncode:
        print(result.stdout,result.stderr);raise SystemExit(result.returncode)
    print('PASS',command[0],flush=True)
print('OFFLINE_SUITE_PASSED; live API calls and scheduler submissions: 0')
