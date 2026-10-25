#!/usr/bin/env python3
"""Validate and run one approved filing. Paid calls occur only without --offline."""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

def verify(run, index):
    if not __debug__:
        raise RuntimeError('Python -O disables required validation')
    contract = json.loads((run / 'inputs/releases/v1/contract.json').read_text())
    files = dict(contract['files'])
    files.update(json.loads((run / 'state/runtime_snapshot.json').read_text()))
    for name, digest in files.items():
        if hashlib.sha256((run / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Changed frozen run file: ' + name)
    with (run / 'inputs/releases/v1/approved_manifest.csv').open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    if not 0 <= index < len(rows):
        raise ValueError('Report index is outside the approved manifest')
    for name in ['raw', 'item1']:
        if hashlib.sha256(Path(rows[index][name + '_path']).read_bytes()).hexdigest() != rows[index][name + '_sha256']:
            raise ValueError('Input hash mismatch: ' + name)
    return rows[index]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--index', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', '0')))
    p.add_argument('--offline', action='store_true')
    a = p.parse_args(); run = a.run.resolve()
    row = verify(run, a.index)
    if a.offline:
        print(json.dumps({'pass': True, 'accession': row['accession_number'], 'model_calls': 0}))
        return
    if not os.environ.get('LITELLM_API_KEY'):
        raise ValueError('Set LITELLM_API_KEY in the process environment')
    env = dict(os.environ, CASINO_RELEASE='v1', SLURM_ARRAY_TASK_ID=str(a.index),
               SLURM_JOB_ID=os.environ.get('SLURM_JOB_ID', 'manual'))
    subprocess.run([sys.executable, str(run / 'code/run_report.py')], cwd=run, env=env, check=True)

if __name__ == '__main__':
    main()
