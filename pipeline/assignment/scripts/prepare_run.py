#!/usr/bin/env python3
"""Prepare an isolated run using bundled code and explicit verified source data. No API calls."""
import argparse
import ast
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]
WORDS = r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*"
PHRASE_COLUMNS = ('final_phrase', 'atom_mechanism_phrase', 'activity_phrase')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def readcsv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

def writecsv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def prepare(args):
    if not __debug__:
        raise RuntimeError('Python -O disables required validation')
    source, taxonomy, output = args.source.resolve(), args.taxonomy.resolve(), args.output.resolve()
    if not args.confirm_verified_source:
        raise ValueError('Verify report identity, fiscal year, source text and provenance first; then use --confirm-verified-source.')
    if not re.fullmatch(r'[a-z][a-z0-9_]*', args.industry):
        raise ValueError('Use a lowercase industry identifier with letters, digits and underscores.')
    if output.exists():
        raise FileExistsError('Run directory already exists. Resume it, or choose a genuinely new run; never overwrite it.')
    if not 1 <= args.workers <= 768 or args.budget_cap_usd <= 0:
        raise ValueError('Workers must be 1..768 and the paid API budget must be positive.')
    rows, tax = readcsv(source), readcsv(taxonomy)
    if not rows or not tax:
        raise ValueError('Source and taxonomy must be nonempty.')
    requested_column = getattr(args, 'phrase_column', None)
    phrase_column = requested_column or next((name for name in PHRASE_COLUMNS if name in rows[0]), None)
    if not phrase_column or phrase_column not in rows[0]:
        raise ValueError('Provide final_phrase, atom_mechanism_phrase or activity_phrase, '
                         'or select an existing column with --phrase-column.')
    for line, row in enumerate(rows, 2):
        if not (row.get(phrase_column) or '').strip():
            raise ValueError(f'CSV row {line}: empty phrase in selected column {phrase_column!r}.')
        if 'final_phrase' in row and row['final_phrase'] != row[phrase_column]:
            raise ValueError(f'CSV row {line}: selected phrase conflicts with existing final_phrase; '
                             'resolve the source columns before preparing a run.')
        if 'original_sentence' not in row or not row.get('year'):
            raise ValueError('Each source row needs original_sentence, year and ticker or firm_id.')
        if not (row.get('ticker') or row.get('firm_id')):
            raise ValueError('Missing ticker/firm_id; ticker identity must already be verified.')
        if row.get('status') and row['status'] != 'fully_verified':
            raise ValueError('Existing non-verified status cannot be overwritten.')
        if row.get('industry') and row['industry'] != args.industry:
            raise ValueError('Existing industry field conflicts with the requested industry.')
    if len({row['parent_id'] for row in tax}) != len(tax):
        raise ValueError('Duplicate parent IDs.')
    parents = []
    for row in tax:
        label = row['parent_label'].strip()
        if not label:
            raise ValueError('Empty parent label.')
        scope = row.get('inclusion_rule', '').strip() or label
        exclusion = row.get('exclusion', '').strip()
        if exclusion:
            scope += '\nExclusions: ' + exclusion
        parents.append(dict(parent_id=row['parent_id'], parent_label=label,
                            parent_scope=scope, parent_exclusion=exclusion))
    adapted, grouped = [], {}
    for line, row in enumerate(rows, 2):
        item = dict(row)
        item['final_phrase'] = row[phrase_column]
        if item.get('source_original_row') and item['source_original_row'] != str(line):
            raise ValueError('Existing source_original_row conflicts with actual CSV order.')
        if not item.get('ticker'):
            item['ticker'] = item['firm_id'].removeprefix('ECON::')
        item.update(source_original_row=str(line), status='fully_verified', industry=args.industry)
        adapted.append(item)
        phrase = re.sub(r'\s+', ' ', item['final_phrase']).strip()
        # GLM production uses <=4, distinct from Gemini's later strict <4 policy.
        short = len(re.findall(WORDS, phrase)) <= 4
        context = row['original_sentence'] if short else ''
        unit = grouped.setdefault((phrase, context), dict(
            phrase=phrase, aliases=[phrase], kind='short' if short else 'long',
            original_sentence=context, source_rows=[], industries=[args.industry]))
        unit['source_rows'].append(line)
    units = sorted(grouped.values(), key=lambda unit: (unit['phrase'], unit['original_sentence']))
    for index, unit in enumerate(units):
        unit['context_texts'] = [f'Phrase: {p}\nContext: {unit["original_sentence"]}' for p in unit['aliases']] if unit['kind'] == 'short' else []
        unit.update(unit_index=index, unit_id=f'{args.industry.upper()}{index:06}',
                    canonical_id='C' + hashlib.sha256(unit['phrase'].encode()).hexdigest()[:16])
    mapped = [line for unit in units for line in unit['source_rows']]
    assert sorted(mapped) == list(range(2, len(rows) + 2))
    prompt_source = BUNDLE / 'code/production/pipeline_filter.py'
    tree = ast.parse(prompt_source.read_text())
    prompt = next(ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'PROMPT' for t in node.targets))
    assert prompt + '\n' == (BUNDLE / 'prompts/glm_high_recall.txt').read_text(), 'Prompt copy drift'
    assert (BUNDLE / 'prompts/parent_first_prompts.json').read_bytes() == (BUNDLE / 'code/production/parent_first_prompts.json').read_bytes()
    for subdir in ['inputs/embedding_cache', 'work/pipeline', 'outputs/pipeline', 'logs']:
        (output / subdir).mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output / 'inputs/source_original.csv')
    shutil.copy2(taxonomy, output / 'inputs/final_taxonomy_enrich.csv')
    writecsv(output / 'inputs/phrases.csv', adapted)
    writecsv(output / 'inputs/parents.csv', parents)
    for src in (BUNDLE / 'code/production').iterdir():
        if src.is_file() and src.suffix in ('.py', '.json'):
            shutil.copy2(src, output / 'work' / src.name)
    dump(output / 'work/pipeline/units.json', units)
    dump(output / 'work/pipeline/glm_worker_target.json', {'workers': args.workers})
    dump(output / 'work/pipeline/gemini_candidate_policy.json', {
        'k': 30, 'unconditional_for_topk': True, 'topk_universe': 'all frozen taxonomy parents',
        'tie_break': 'ascending frozen parent index'})
    config = dict(
        run_id=args.run_id or output.name, industry=args.industry, source_rows=len(rows),
        assignment_units=len(units), parents=len(parents), threshold=0.33,
        budget_cap_usd=args.budget_cap_usd, gemini_enabled=False,
        source_sha256=sha(output / 'inputs/phrases.csv'), source_original_sha256=sha(source),
        taxonomy_sha256=sha(output / 'inputs/parents.csv'), frozen_sha256=sha(taxonomy),
        taxonomy_original_sha256=sha(taxonomy), identity_version=args.identity_version,
        source_phrase_column=phrase_column,
        firms=len({row['ticker'] for row in adapted}),
        firm_years=len({(row['ticker'], row['year']) for row in adapted}),
        budget_scope='Embedding and optional Gemini only; internal GLM dollar cost unknown.',
        recall_policy=getattr(args, 'recall_policy', 'legacy'),
        max_parents_per_unit=1000 if getattr(args, 'recall_policy', 'legacy') == 'top1000' else None,
        threshold_operator='>' if getattr(args, 'recall_policy', 'legacy') == 'top1000' else '>=',
        pipeline='Explicit recall policy; GLM high recall; optional Gemini kept-union-top30 reciprocal review')
    dump(output / 'config.json', config)
    files = [p for directory in ['inputs', 'work'] for p in (output / directory).rglob('*') if p.is_file()]
    dump(output / 'preparation_manifest.json', {
        'prepared_without_api': True, 'source_verified_by_operator': True,
        'source_argument': str(source), 'taxonomy_argument': str(taxonomy),
        'hashes': {str(p.relative_to(output)): sha(p) for p in files},
        'unit_policy': 'normalized phrase + exact context when word count <=4; longer phrases omit context',
        'config': config})
    return config

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recall-policy', choices=['legacy', 'top1000'], default='top1000',
                        help='top1000: score > .33 intersect global top1000; legacy: all scores >= .33')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--phrase-column', help='Source phrase column; default precedence: '
                        'final_phrase, atom_mechanism_phrase, activity_phrase. Original columns are preserved.')
    parser.add_argument('--taxonomy', type=Path, default=BUNDLE / 'inputs/frozen/final_taxonomy_enrich.csv')
    parser.add_argument('--industry', required=True)
    parser.add_argument('--identity-version', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-id')
    parser.add_argument('--workers', type=int, default=512)
    parser.add_argument('--budget-cap-usd', type=float, required=True)
    parser.add_argument('--confirm-verified-source', action='store_true')
    args = parser.parse_args()
    print(json.dumps(prepare(args), indent=2))

if __name__ == '__main__':
    main()
