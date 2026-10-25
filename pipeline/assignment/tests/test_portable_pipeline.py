#!/usr/bin/env python3
"""Exercise the real recall/filter/cache/gate code with fake responses; all network is blocked."""
import importlib
import json
import os
import runpy
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

BUNDLE = Path(__file__).resolve().parents[1]

def main():
    with tempfile.TemporaryDirectory(prefix='portable_glm_test_') as tmp:
        run = Path(tmp) / 'demo'
        args = [sys.executable, str(BUNDLE / 'scripts/prepare_run.py'),
                '--source', str(BUNDLE / 'examples/phrases.csv'),
                '--taxonomy', str(BUNDLE / 'examples/taxonomy.csv'),
                '--industry', 'demo', '--identity-version', 'synthetic_test',
                '--output', str(run), '--workers', '2', '--budget-cap-usd', '1',
                '--confirm-verified-source']
        subprocess.run(args, check=True, capture_output=True, text=True)
        # Existing runs must never be overwritten by preparation.
        assert subprocess.run(args, capture_output=True).returncode != 0
        units = json.loads((run / 'work/pipeline/units.json').read_text())
        assert len(units) == 3
        assert sorted(line for unit in units for line in unit['source_rows']) == [2, 3, 4, 5]
        short = [u for u in units if u['kind'] == 'short']
        assert len(short) == 2 and all(len(u['phrase'].split()) == 4 for u in short)
        assert sorted(len(u['source_rows']) for u in short) == [1, 2]
        assert next(u for u in units if u['kind'] == 'long')['original_sentence'] == ''
        subprocess.run([sys.executable, str(run / 'work/run_glm_only.py'), '--offline'], check=True)

        secret = Path(tmp) / 'fake_credentials.json'
        secret.write_text(json.dumps({'OSS_API_KEY': 'TEST_NOT_A_REAL_KEY',
                                      'OPENAI_API_KEY': 'TEST_NOT_A_REAL_KEY'}))
        secret.chmod(0o600)
        os.environ.update(PIPELINE_SECRET_FILE=str(secret), GLM_FILTER_WORKERS='2',
                          GLM_FILTER_INITIAL_WORKERS='2', GLM_FILTER_RAMP_SECONDS='0',
                          GLM_TRACK_COSTS='0', GLM_DEFER_TRANSPORT='1')
        sys.path.insert(0, str(run / 'work'))
        common = importlib.import_module('pipeline_common')
        calls = {'embedding': 0, 'glm': 0}

        def blocked(*args, **kwargs):
            raise AssertionError('Unexpected real network access in offline integration test')

        socket.create_connection = blocked
        common.urllib.request.urlopen = blocked

        def fake_post(url, body, key, **kwargs):
            assert key == 'TEST_NOT_A_REAL_KEY'
            if url == 'https://api.openai.com/v1/embeddings':
                import base64
                calls['embedding'] += 1
                vector = np.ones(1536, dtype='<f4')
                encoded = base64.b64encode(vector.tobytes()).decode()
                return {'usage': {'total_tokens': 12}, 'data': [
                    {'index': i, 'embedding': encoded} for i in range(len(body['input']))]}, None
            assert url == 'https://litellm.parcc.upenn.edu/v1/chat/completions'
            calls['glm'] += 1
            assert body['model'] == 'parcc/GLM'
            assert body['chat_template_kwargs'] == {'enable_thinking': False}
            assert body['temperature'] == 0 and body['max_tokens'] == 2048
            payload = json.loads(body['messages'][1]['content'])
            assert set(payload['parent']) == {'parent_label', 'parent_scope', 'parent_exclusion'}
            decisions = [int(i % 2 == 0) for i in range(payload['N'])]
            return {'id': 'offline-test', 'usage': {'prompt_tokens': 100, 'completion_tokens': 10},
                    'choices': [{'finish_reason': 'stop', 'message': {
                        'content': json.dumps({'decisions': decisions})}}]}, None

        common.post = fake_post
        runner = importlib.import_module('run_glm_only')
        runner.run_with_recovery()
        assert calls['embedding'] > 0 and calls['glm'] == 3
        runpy.run_path(str(run / 'work/validate_glm_complete.py'), run_name='__main__')
        summary = json.loads((run / 'outputs/pipeline/filter_summary.json').read_text())
        assert summary['candidate_pairs'] == 9 and summary['pending'] == summary['deferred_pairs'] == 0
        before = dict(calls)
        runner.run_with_recovery()
        assert calls == before, 'A cached rerun must not make new API calls'
        runpy.run_path(str(run / 'work/validate_glm_complete.py'), run_name='__main__')

        # Optional downstream request preparation also works without old-machine files.
        reciprocal = importlib.import_module('run_reciprocal')
        reciprocal.prepare_data()
        batch = reciprocal.runner()
        batch.prepare('parent')
        assert calls == before
        # Feed synthetic batch responses through the real parser and reciprocal intersection.
        for stage in ['parent', 'phrase']:
            plan = batch.prepare(stage)
            for shard in plan['shards']:
                folder = batch.folder(stage)
                mappings = json.loads((folder / shard['mapping']).read_text())
                responses = []
                for mp in mappings:
                    chosen = [i for i in range(mp['candidate_count']) if i % 2 == 0]
                    obj = {'anchor_id': mp['key'], 'reviewed_count': mp['candidate_count'], 'matching_ids': chosen}
                    responses.append({'key': mp['key'], 'response': {
                        'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': json.dumps(obj)}]}}],
                        'usageMetadata': {'promptTokenCount': 0, 'candidatesTokenCount': 0}, 'modelVersion': 'offline-fake'}})
                rid = 'offline:' + stage + str(shard['shard'])
                batch.charge(rid, upper=0)
                (folder / f"intent_{shard['shard']:03d}.json").write_text(json.dumps({'reservation_id': rid}))
                (folder / f"responses_{shard['shard']:03d}.jsonl").write_text(''.join(json.dumps(r)+'\n' for r in responses))
            batch.parse(stage, complete=True)
        assert batch.final()['complete']
        reciprocal.export()
        runpy.run_path(str(run / 'work/export_assignments.py'), run_name='__main__')
        final = json.loads((run / 'outputs/pipeline/final/validation.json').read_text())
        assert final['complete'] and final['source_rows'] == 4 and final['unresolved'] == 0
        # Threshold semantics and deterministic top-k ties are distinct from the legacy mode.
        recall = importlib.import_module('pipeline_recall')
        scores = np.full(1002, .5); scores[-1] = .33
        assert recall.select_candidates(scores).tolist() == list(range(1000))
        recall.CONFIG['recall_policy'] = 'legacy'
        assert len(recall.select_candidates(scores)) == 1002
        recall.CONFIG['recall_policy'] = 'top1000'
        config_path = run / 'config.json'
        changed = json.loads(config_path.read_text()); changed['threshold'] = .1
        config_path.write_text(json.dumps(changed))
        assert subprocess.run([sys.executable, str(run/'work/run_glm_only.py'), '--offline'], capture_output=True).returncode != 0
        common.close_databases()
        print(json.dumps({'pass': True, 'real_api_calls': 0, 'source_rows': 4,
                          'units': 3, 'candidate_pairs': 9,
                          'checks': ['source conservation', 'four-word context boundary',
                                     'strict GLM payload', 'real recall and filtering code',
                                     'SQLite usage/cache persistence', 'zero-call resume',
                                     'completion gate', 'Gemini response parsing and reciprocal intersection',
                                     'final source-row export', 'top1000 threshold and ties', 'frozen config tamper rejected']}, indent=2))

if __name__ == '__main__': main()
