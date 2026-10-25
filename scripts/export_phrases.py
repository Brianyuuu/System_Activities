#!/usr/bin/env python3
"""Export a fully completed extraction release, preserving zero-phrase firm-years."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

def export(run, output, industry):
    if output.exists():
        raise FileExistsError('Choose a fresh export directory')
    with (run / 'inputs/releases/v1/approved_manifest.csv').open(encoding='utf-8-sig') as stream:
        manifest = list(csv.DictReader(stream))
    contract_path = run / 'inputs/releases/v1/contract.json'
    contract = json.loads(contract_path.read_text())
    for name, digest in contract['files'].items():
        if hashlib.sha256((run / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Changed release file: ' + name)
    with (run / 'outputs/coverage_inventory.csv').open(encoding='utf-8-sig') as stream:
        coverage = [r for r in csv.DictReader(stream) if r['industry'] == industry]
    if not coverage:
        raise ValueError('No coverage rows for the exact source industry name')
    manifest = [r for r in manifest if r['industry'] == industry]
    phrases, panel, seen = [], [], set()
    for row in manifest:
        for name in ['raw', 'item1']:
            if hashlib.sha256(Path(row[name+'_path']).read_bytes()).hexdigest() != row[name+'_sha256']:
                raise ValueError('Changed source input: ' + name)
        key = row['company_id'] + '_' + row['accession_number']
        receipt = json.loads((run / 'state' / (key + '.json')).read_text())
        if receipt.get('status') != 'complete' or not receipt.get('technical_qa_pass'):
            raise ValueError('Incomplete report: ' + key)
        if (receipt.get('source_contract_sha256') != hashlib.sha256(contract_path.read_bytes()).hexdigest()
                or receipt.get('item1_sha256') != row['item1_sha256']):
            raise ValueError('Report receipt does not match this source release')
        source = Path(receipt['enriched_output'])
        if hashlib.sha256(source.read_bytes()).hexdigest() != receipt['enriched_sha256']:
            raise ValueError('Changed phrase export: ' + key)
        with source.open(encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != receipt['phrases']:
            raise ValueError('Phrase row count mismatch')
        # Use the verified economic-entity ID as the canonical firm key, not a reused ticker.
        fy = row['company_id'], row['year']
        if fy in seen:
            raise ValueError('Resolve multiple reports in one company-year before export')
        seen.add(fy)
        for item in rows:
            item['source_ticker'] = item['ticker']
            item['ticker'] = row['company_id']
            item['final_phrase'] = item['atom_mechanism_phrase']
        phrases.extend(rows)
        panel.append(dict(firm=fy[0], year=fy[1], complete='1', expected_source_rows=len(rows)))
    for row in coverage:
        fy = row['company_id'], row['panel_year']
        if fy not in seen:
            panel.append(dict(firm=fy[0], year=fy[1], complete='0', expected_source_rows=0))
            seen.add(fy)
    output.mkdir(parents=True)
    def write(name, rows, fields):
        with (output / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    write('phrases.csv', phrases, list(dict.fromkeys(k for row in phrases for k in row)) or
          ['ticker', 'year', 'final_phrase', 'original_sentence'])
    write('firm_years.csv', panel, ['firm', 'year', 'complete', 'expected_source_rows'])
    (output / 'validation.json').write_text(json.dumps({'reports':len(manifest),'firm_years':len(panel),'phrases':len(phrases),
        'technical_qa_pass':True,'semantic_certification':False,
        'industry':industry, 'scope':'approved release phrases; held reports retained as missing coverage'},indent=2))

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--industry',required=True)
    a=p.parse_args();export(a.run.resolve(),a.output.resolve(),a.industry)
