#!/usr/bin/env python3
"""Prepare an isolated primary-Item-1 run from an explicit, verified accession mapping."""
import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ('company_id ticker fiscal_year cik accession form report_period_end filing_date '
            'company_name_at_year registrant_names identity_basis fiscal_year_basis '
            'verification_scope included_in_has10k panel_has10k industry').split()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')

def write_csv(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def prepare(mapping, output, confirm_identity):
    if not __debug__:
        raise RuntimeError('Do not run the workflow with Python -O: validation assertions are required.')
    if not confirm_identity:
        raise ValueError('Resolve historical company identities first; --confirm-identity records that review.')
    mapping, output = mapping.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError('Use a fresh directory; existing run state must not be overwritten.')
    with mapping.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('Empty accession mapping')
    panel, accessions = {}, {}
    for row in rows:
        missing = [k for k in REQUIRED if not row.get(k)]
        if missing:
            raise ValueError('Missing mapping fields: ' + ', '.join(missing))
        if not re.fullmatch(r'[A-Za-z0-9_-]+', row['company_id']):
            raise ValueError('company_id must be a safe filename identifier')
        if not re.fullmatch(r'\d{10}-\d{2}-\d{6}', row['accession']):
            raise ValueError('Expected a hyphenated SEC accession')
        if row['included_in_has10k'] != '1' or row['panel_has10k'] != '1':
            raise ValueError('Mapping must contain only the positively verified target scope')
        if not 1900 <= int(row['fiscal_year']) <= 2200:
            raise ValueError('Invalid fiscal year')
        key = row['company_id'], row['fiscal_year']
        if row['accession'] in accessions and accessions[row['accession']] != key:
            raise ValueError('One accession cannot represent multiple company-years')
        accessions[row['accession']] = key
        target = dict(company_id=key[0], year=key[1], ICB_new=row['industry'], has10k_dummy='1')
        if key in panel and panel[key] != target:
            raise ValueError('Conflicting industry for the same company-year')
        panel[key] = target
        for name in ['sec_primary_document_url', 'sec_full_submission_url']:
            row.setdefault(name, '')
        if row.get('local_raw_path'):
            raw = Path(row['local_raw_path'])
            raw = raw if raw.is_absolute() else mapping.parent / raw
            if not row.get('raw_sha256') or sha(raw) != row['raw_sha256']:
                raise ValueError('Local source SHA-256 does not match the supplied mapping')
            row['local_raw_path'] = str(raw.resolve())
    shutil.copytree(ROOT / 'pipeline/extraction/template', output)
    for directory in ['state', 'inputs', 'source/inputs', 'source/raw', 'source/evidence',
                      'source/audits/source_packets', 'logs', 'outputs']:
        (output / directory).mkdir(parents=True, exist_ok=True)
    shutil.copy2(mapping, output / 'inputs/mapping_original.csv')
    source = output / 'source'
    write_csv(source / 'inputs/panel_scope.csv', list(panel.values()), list(next(iter(panel.values()))))
    fields = list(dict.fromkeys(k for row in rows for k in row))
    write_csv(source / 'inputs/mapping_scope.csv', rows, fields)
    dump(source / 'inputs/panel_provenance.json', {
        'scope_sha256': sha(source / 'inputs/panel_scope.csv'),
        'mapping_scope_sha256': sha(source / 'inputs/mapping_scope.csv'),
        'operator_confirmed_identity': True})
    dump(output / 'inputs/extracted_accessions.json', {})
    config = json.loads((output / 'config.json').read_text())
    settings = config['latest_phrase_pipeline']
    for name in ['clean_manifest', 'output_root', 'prompt_root', 'source_root']:
        settings[name] = str(output / settings[name])
    dump(output / 'config.json', config)
    runtime = {str(p.relative_to(output)): sha(p) for directory in ['runtime', 'pipeline_latest', 'code']
               for p in (output / directory).rglob('*') if p.is_file() and p.suffix in {'.py', '.txt', '.md'}}
    dump(output / 'state/runtime_snapshot.json', runtime)
    subprocess.run([sys.executable, str(output / 'code/prepare_from_mapping.py')], cwd=output, check=True)
    subprocess.run([sys.executable, str(output / 'code/build_item1_release.py'),
                    '--source-root', str(source), '--output-root', str(output), '--release', 'v1'], check=True)
    summary = json.loads((output / 'audits/item1_only_build.json').read_text())
    dump(output / 'preparation.json', {'mapping_sha256': sha(mapping), 'summary': summary,
                                     'model_calls': 0, 'submitted': False})
    return summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--confirm-identity', action='store_true')
    args = parser.parse_args()
    print(json.dumps(prepare(args.mapping, args.output, args.confirm_identity), indent=2))
