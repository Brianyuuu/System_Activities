"""Build immutable, primary-Item-1-only inputs from reviewed source packets.

No reference discovery, reference downloads, or reference-completeness gate.
Identity, fiscal-year, primary-document and Item 1 boundary decisions remain.
This command never submits a job or invokes a model.
"""
import argparse
import collections
import csv
import hashlib
import json
import re
from pathlib import Path

SCOPE = 'primary_item1_only'
METHOD = 'SEC_official_reviewed_primary_Item1_only'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def reviewed_packets(source):
    decision_path = source / 'audits/manual_source_decisions.json'
    decisions = {x['accession']: x for x in json.loads(decision_path.read_text())} if decision_path.exists() else {}
    packets = []
    for path in sorted((source / 'audits/source_packets').glob('*.json')):
        packet = json.loads(path.read_text())
        decision = decisions.get(packet.get('accessionNumber',''))
        if decision:
            assert decision['raw_sha256'] == packet['raw_sha256']
            assert decision['actual_period_end'] == packet.get('actual_period_end')
            if decision.get('reporting_name'):
                normalize = lambda s: re.sub(r'\W', '', s.lower())
                assert normalize(decision['reporting_name']) in normalize(packet['cover_excerpt'])
                packet['reporting_name'] = decision['reporting_name']
            packet['source_issues'] = [x for x in packet.get('source_issues', []) if x not in decision['resolved_issues']]
            packet['manual_review_decision'] = decision
            if not packet['source_issues']:
                packet['source_status'] = 'evidence_ready'
        packets.append(packet)
    return packets


def select_packet(row, packets):
    candidates = [p for p in packets if p['company_id'] == row['company_id'] and str(p.get('panel_year',p.get('reportDate','')[:4])) == row['year']]
    transitions = [p for p in candidates if re.search(r'(?i)\[\s*X\s*\]\s*TRANSITION REPORT', p.get('cover_excerpt', ''))]
    if len(candidates) > 1 and len(candidates) - len(transitions) == 1:
        candidates = [p for p in candidates if p not in transitions]
    if len(candidates) != 1:
        return None, 'source_candidate_not_unique' if candidates else 'no_downloaded_original_for_panel_year'
    packet = candidates[0]
    if packet.get('source_status') == 'existing_run_claimed':
        return None, 'existing_run_claimed:' + packet['accessionNumber']
    if packet.get('source_status') == 'previously_extracted':
        return None, 'previously_extracted:' + packet['accessionNumber']
    if packet.get('source_status') != 'evidence_ready' or packet.get('source_issues'):
        return None, ';'.join(packet.get('source_issues', [])) or 'source_review_pending'
    return packet, ''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--release', required=True)
    parser.add_argument('--published-source-root', type=Path, help='Original remote source root when building from a local snapshot')
    parser.add_argument('--published-root', type=Path, help='Remote destination paths to record in a locally prepared release')
    args = parser.parse_args()
    source, output = args.source_root.resolve(), args.output_root.resolve()
    published_source = args.published_source_root or source
    published = args.published_root or output
    assert output != source, 'Use a separate project; never overwrite a running source project'
    base = output / 'inputs/releases' / args.release
    assert not base.exists(), 'Immutable release already exists'
    assert not (output / 'inputs/releases').exists() or not list((output / 'inputs/releases').iterdir()), 'Use a fresh output project to avoid duplicate filing assignments'
    panel = [x for x in csv.DictReader((source / 'inputs/panel_scope.csv').open(encoding='utf-8-sig')) if x['has10k_dummy'] == '1']
    keys = [(x['company_id'], x['year']) for x in panel]
    assert len(keys) == len(set(keys)), 'Duplicate panel keys'
    packets = reviewed_packets(source)
    rows, coverage, evidence = [], [], []
    for target in panel:
        packet, reason = select_packet(target, packets)
        record = {'company_id': target['company_id'], 'panel_year': target['year'], 'industry':target['ICB_new'], 'status': 'source_review_pending', 'reason': reason, 'source_scope': SCOPE}
        if packet is None:
            if reason.startswith('existing_run_claimed:'):
                record.update(status='existing_run_claimed',accession=reason.split(':',1)[1])
            if reason.startswith('previously_extracted:'):
                record.update(status='previously_extracted_preserved',accession=reason.split(':',1)[1])
            coverage.append(record)
            continue
        body_path = Path(packet['item1_body_path'])
        if not body_path.is_relative_to(source):
            body_path = source / body_path.relative_to(published_source)
        assert sha(body_path) == packet['item1_body_sha256'], body_path
        body = body_path.read_text().strip()
        assert 500 <= len(body.split()) <= 50000, 'Unverified Item 1 body extent'
        accession = packet['accessionNumber']
        relative_text = Path('evidence') / f'{accession}_primary_item1.txt'
        text_path = output / relative_text
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(body)
        part = {'kind': 'primary_Item1', 'source_url': packet['source_url'], 'source_path': packet['primary_text_path'], 'source_sha256': packet['primary_text_sha256'], 'text_path': str(published / relative_text), 'text_sha256': sha(text_path), 'words': len(body.split())}
        relative_parts = Path('audits/approved_parts') / f'{accession}.json'
        write_json(output / relative_parts, [part])
        metadata = {'accession_number': accession, 'sample_id': packet['company_id'], 'ticker': packet['canonical_ticker'], 'cik': packet['cik'], 'target_fiscal_year': packet['fiscal_year'], 'form': packet['form'], 'source_part_spans': [[0, len(body)]], 'source_method': METHOD, 'source_scope': SCOPE, 'incorporated_references_included': False}
        relative_item = Path('outputs/item1') / f'{accession}.md'
        item = output / relative_item
        item.parent.mkdir(parents=True, exist_ok=True)
        item.write_text('---\nmetadata: ' + json.dumps(metadata, sort_keys=True) + '\n---\n\n# Item 1. Business\n\n' + body + '\n')
        rows.append({'industry':target['ICB_new'], 'sample_id': packet['company_id'], 'company_id': packet['company_id'], 'ticker': packet['canonical_ticker'], 'panel_year': str(packet['panel_year']), 'year': str(packet['fiscal_year']), 'target_fiscal_year': str(packet['fiscal_year']), 'company_name_wrds': packet['reporting_name'], 'historical_company_name': packet['reporting_name'], 'cik': packet['cik'], 'accession_number': accession, 'form': packet['form'], 'report_date': packet['reportDate'], 'actual_fiscal_period_end': packet['actual_period_end'], 'filing_date': packet['filingDate'], 'raw_path': str(published_source / 'raw' / packet['raw_filename']), 'raw_sha256': packet['raw_sha256'], 'item1_path': str(published / relative_item), 'item1_sha256': sha(item), 'item1_words': str(len(body.split())), 'parts_path': str(published / relative_parts), 'source_url': packet['source_url'], 'source_method': METHOD, 'source_scope': SCOPE, 'source_review_pass': 'true', 'clean_status': 'pass', 'review_basis': 'frozen accession mapping identity certificate + document CIK/name/period + primary Item1 boundary', 'fiscal_year_basis': packet['fiscal_year_basis']})
        evidence.append({k: v for k, v in packet.items() if k not in ('cross_reference_contexts', 'exhibit_references')})
        record.update(status='approved_for_extraction', accession=accession, fiscal_year=str(packet['fiscal_year']), release=args.release)
        coverage.append(record)
    assert len({(r['company_id'], r['accession_number']) for r in rows}) == len(rows)
    write_csv(base / 'approved_manifest.csv', rows)
    write_json(base / 'source_evidence.json', evidence)
    write_csv(output / 'outputs/coverage_inventory.csv', coverage)
    write_json(output / 'source_scope_policy.json', {'source_scope': SCOPE, 'include_primary_Item1': True, 'follow_incorporated_references': False, 'download_referenced_material': False, 'block_on_reference_completeness': False, 'preserve_reference_sentences_inside_Item1': True, 'reuse_prior_mixed_scope_phrase_outputs': False, 'required_checks': ['company_identity', 'fiscal_year', 'primary_document', 'Item1_boundary', 'source_hashes'], 'existing_run_policy': 'leave_running_jobs_and_outputs_unchanged'})
    files = [base / 'approved_manifest.csv', base / 'source_evidence.json', output / 'source_scope_policy.json', output / 'outputs/coverage_inventory.csv']
    for name in ['config.json', 'code/run_report.py', 'code/strict_stage_runtime.py', 'code/extract.sbatch', 'state/runtime_snapshot.json']:
        if (output / name).is_file():
            files.append(output / name)
    write_json(base / 'contract.json', {'release': args.release, 'reports': len(rows), 'source_scope': SCOPE, 'files': {str(p.relative_to(output)): sha(p) for p in files}, 'source_boundary_guard': True, 'model_calls': 0, 'prior_phrase_outputs_reused': 0})
    summary = {'source_scope': SCOPE, 'target_company_years': len(panel), 'approved_reports': len(rows), 'held_reports': sum(x['status']=='source_review_pending' for x in coverage), 'existing_run_claimed':sum(x['status']=='existing_run_claimed' for x in coverage), 'previously_extracted_preserved':sum(x['status']=='previously_extracted_preserved' for x in coverage), 'hold_reasons': dict(collections.Counter(x['reason'] for x in coverage if x['status'] != 'approved_for_extraction')), 'model_calls': 0, 'jobs_submitted': 0, 'prior_phrase_outputs_reused': 0}
    write_json(output / 'audits/item1_only_build.json', summary)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
