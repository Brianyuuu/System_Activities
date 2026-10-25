#!/usr/bin/env python3
"""Provide Merci's decomposition input column without performing early cleaning."""
import argparse, csv
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
with args.input.open(encoding='utf-8-sig', newline='') as handle:
    reader = csv.DictReader(handle); fields = list(reader.fieldnames or []); rows = list(reader)
if 'replaced_activity_phrase' not in fields: fields.append('replaced_activity_phrase')
for row in rows:
    row['replaced_activity_phrase'] = row.get('activity_phrase', '')
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open('w', encoding='utf-8', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
print(f'Wrote {len(rows)} rows')
