#!/usr/bin/env python3
"""Check industry-independent phrase adaptation and source preservation without network."""
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    originals = read(BUNDLE / 'examples/phrases.csv')
    with tempfile.TemporaryDirectory(prefix='phrase_adapter_test_') as directory:
        root = Path(directory)
        reference_units = None
        reference_phrases = None
        cases = [
            ('canonical', 'final_phrase', None),
            ('mechanism', 'atom_mechanism_phrase', None),
            ('activity', 'activity_phrase', None),
            ('custom', 'custom_phrase', 'custom_phrase'),
            ('precedence', 'atom_mechanism_phrase', None),
            ('override', 'activity_phrase', 'activity_phrase'),
            ('canonical_priority', 'final_phrase', None),
            ('missing', 'custom_phrase', None),
            ('unknown', 'final_phrase', 'nonexistent'),
            ('blank', 'atom_mechanism_phrase', None),
            ('conflict', 'final_phrase', 'activity_phrase'),
        ]
        results = []
        for name, column, explicit in cases:
            rows = [dict(row) for row in originals]
            for row in rows:
                phrase = row.pop('final_phrase')
                row[column] = phrase
                if name in ('precedence', 'canonical_priority', 'conflict'):
                    row['activity_phrase'] = 'a different source representation'
                if name in ('override', 'canonical_priority'):
                    row['atom_mechanism_phrase'] = 'another source representation'
            if name == 'blank':
                rows[0][column] = ''
                for row in rows:
                    row['activity_phrase'] = 'fallback must not mask missing data'
            source = root / f'{name}.csv'
            with source.open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            run = root / name
            command = [sys.executable, '-B', str(BUNDLE / 'scripts/prepare_run.py'),
                       '--source', str(source), '--taxonomy', str(BUNDLE / 'examples/taxonomy.csv'),
                       '--industry', 'arbitrary_sector_42', '--identity-version', 'synthetic_test',
                       '--output', str(run), '--workers', '2', '--budget-cap-usd', '1',
                       '--confirm-verified-source']
            if explicit:
                command += ['--phrase-column', explicit]
            prepared = subprocess.run(command, text=True, capture_output=True)
            if name in ('missing', 'unknown', 'blank', 'conflict'):
                assert prepared.returncode != 0, name
                assert not run.exists(), name
                results.append(name + ': rejected before run creation')
                continue
            assert prepared.returncode == 0, prepared.stderr
            config = json.loads(prepared.stdout)
            assert config['source_phrase_column'] == column
            manifest = json.loads((run / 'preparation_manifest.json').read_text())
            assert manifest['config']['source_phrase_column'] == column
            assert (run / 'inputs/source_original.csv').read_bytes() == source.read_bytes()
            adapted = read(run / 'inputs/phrases.csv')
            for original, item in zip(rows, adapted):
                assert all(item[key] == value for key, value in original.items())
                assert item['final_phrase'] == original[column]
            units = json.loads((run / 'work/pipeline/units.json').read_text())
            phrases = [row['final_phrase'] for row in adapted]
            if reference_units is None:
                reference_units, reference_phrases = units, phrases
            assert units == reference_units, name
            assert phrases == reference_phrases, name
            offline = subprocess.run([sys.executable, '-B', str(run / 'work/run_glm_only.py'),
                                      '--offline'], text=True, capture_output=True)
            assert offline.returncode == 0, offline.stderr
            results.append(name + ': adapted, source preserved, offline validation passed')
        print(json.dumps({'pass': True, 'real_api_calls': 0, 'checks': results}, indent=2))


if __name__ == '__main__':
    main()
