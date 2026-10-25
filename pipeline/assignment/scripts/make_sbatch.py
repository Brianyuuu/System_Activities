#!/usr/bin/env python3
"""Generate a Slurm script after relocation. This command never submits jobs."""
import argparse
import json
import shlex
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--stage', choices=['offline', 'smoke', 'glm', 'gemini'], required=True)
    p.add_argument('--secret-file', type=Path)
    p.add_argument('--account', required=True)
    p.add_argument('--partition', default='genoa-std-mem')
    p.add_argument('--qos', required=True)
    p.add_argument('--cpus', type=int)
    p.add_argument('--mem')
    p.add_argument('--time')
    a = p.parse_args()
    run = a.run.resolve()
    if not run.is_dir():
        p.error('Run directory must already exist.')
    if a.stage != 'offline' and a.secret_file is None:
        p.error('--secret-file is required for an API stage; its contents are never copied.')
    if a.secret_file and not a.secret_file.is_absolute():
        p.error('--secret-file must be an absolute remote path.')
    for value in [str(run), str(BUNDLE), a.account, a.partition, a.qos]:
        if any(c.isspace() for c in value):
            p.error('Use paths and scheduler identifiers without whitespace on PARCC.')
    small = a.stage in ('offline', 'smoke')
    cpus = a.cpus or (1 if a.stage == 'smoke' else 4 if a.stage == 'gemini' else 8)
    mem = a.mem or ('2G' if a.stage == 'smoke' else '22G' if a.stage == 'gemini' else '44G')
    wall = a.time or ('00:30:00' if small else '48:00:00')
    for value in [mem, wall]:
        if any(c.isspace() for c in value): p.error('Invalid resource value.')
    (run / 'logs').mkdir(exist_ok=True)
    path = run / (a.stage + '.sbatch')
    if path.exists(): p.error('Script exists; review it or choose a new run. No overwrite.')
    lines = ['#!/usr/bin/env bash', f'#SBATCH --job-name={run.name[:30]}-{a.stage}',
             f'#SBATCH --account={a.account}', f'#SBATCH --partition={a.partition}',
             f'#SBATCH --qos={a.qos}', '#SBATCH --nodes=1', '#SBATCH --ntasks=1',
             f'#SBATCH --cpus-per-task={cpus}', f'#SBATCH --mem={mem}', f'#SBATCH --time={wall}',
             '#SBATCH --no-requeue', f'#SBATCH --output={run}/logs/{a.stage}_%j.out',
             f'#SBATCH --error={run}/logs/{a.stage}_%j.err', 'set -euo pipefail', 'umask 077']
    if a.secret_file:
        lines.append('export PIPELINE_SECRET_FILE=' + shlex.quote(str(a.secret_file)))
    lines += ['export PYTHON_BIN=' + shlex.quote(str(Path(__import__('sys').executable).resolve())),
              'exec bash ' + shlex.quote(str(BUNDLE / 'scripts/run_stage.sh')) + ' ' +
              shlex.quote(str(run)) + ' ' + a.stage]
    path.write_text('\n'.join(lines) + '\n')
    print(json.dumps({'script': str(path), 'submitted': False, 'cpus': cpus,
                      'memory': mem, 'walltime': wall}, indent=2))

if __name__ == '__main__': main()
