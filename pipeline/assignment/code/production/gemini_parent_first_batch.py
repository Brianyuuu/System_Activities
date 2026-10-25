#!/usr/bin/env python3
"""Resumable real Gemini Batch runner. No HTTP in prepare or import.

python batch_runner.py --dataset airline --run-dir airline_run prepare
python batch_runner.py --dataset airline --run-dir airline_run --budget-usd 100 run
All paths are relative to the current working directory. Both datasets can share
--ledger (default: beside this script). Never delete an uncertain POST intent.
-2 = outside stage; -1 = unresolved; 0 = rejected; 1 = accepted.
"""
import argparse
import ast
from collections import Counter, defaultdict
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
import numpy as np

API = 'https://generativelanguage.googleapis.com/v1beta'
MODEL = 'gemini-3.1-flash-lite'
TERMINAL = {prefix+state for prefix in ('BATCH_STATE_','JOB_STATE_') for state in ('SUCCEEDED','FAILED','CANCELLED','EXPIRED')}
INPUT_RATE, OUTPUT_RATE = .125, .75


CONTEXT_POLICY = 'both stages: original sentence only for strictly fewer than 4 words; parent labels only'
WORD_PATTERN = r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*"


def needs_sentence_context(unit):
    # Match the existing extraction tokenizer, but use the strict <4 boundary.
    return len(re.findall(WORD_PATTERN, unit['phrase'])) < 4


def compact(x):
    return json.dumps(x, ensure_ascii=False, separators=(',', ':'))


def emit(**x):
    print(compact(x), flush=True)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(p):
    return digest(Path(p).read_bytes())


def read(p):
    return json.loads(Path(p).read_text())


def atomic(p, data):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    with tmp.open('wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)
    fd = os.open(p.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def dump(p, x):
    atomic(p, (json.dumps(x, ensure_ascii=False, indent=2) + '\n').encode())


def save(p, a):
    b = io.BytesIO()
    np.save(b, a, allow_pickle=False)
    atomic(p, b.getvalue())


@contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def require(test, message):
    if not test:
        raise RuntimeError(message)


def constants(path):
    """Read historical constants without importing code or touching its ledger."""
    values = {}
    def value(n):
        if isinstance(n, ast.Name):
            return values[n.id]
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
            return value(n.left) + value(n.right)
        return ast.literal_eval(n)
    for n in ast.parse(path.read_text()).body:
        if isinstance(n, ast.Assign):
            try:
                v = value(n.value)
            except (ValueError, TypeError, KeyError):
                continue
            for t in n.targets:
                if isinstance(t, ast.Name):
                    values[t.id] = v
    return values


def historical_prompts():
    obj=read(Path(__file__).with_name('parent_first_prompts.json'))
    return obj['prompts'],obj['schema']


def get(url, raw=False):
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={'x-goog-api-key': os.environ['GEMINI_API_KEY']})
            with urllib.request.urlopen(req, timeout=180) as response:
                data = response.read()
            return data if raw else json.loads(data)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == 4:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 4:
                raise
        time.sleep(min(2 ** (attempt + 1), 30))


def post(url, body):
    # A paid POST is deliberately attempted exactly once. Intent survives failure.
    req = urllib.request.Request(url, data=compact(body).encode(), headers={'x-goog-api-key': os.environ['GEMINI_API_KEY'], 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.load(response)


def upload(path, display):
    # Files API upload is not a generation request; retrying upload cannot rebill inference.
    for attempt in range(4):
        try:
            req = urllib.request.Request('https://generativelanguage.googleapis.com/upload/v1beta/files', data=compact({'file': {'display_name': display}}).encode(), headers={'x-goog-api-key': os.environ['GEMINI_API_KEY'], 'Content-Type': 'application/json', 'X-Goog-Upload-Protocol': 'resumable', 'X-Goog-Upload-Command': 'start', 'X-Goog-Upload-Header-Content-Length': str(path.stat().st_size), 'X-Goog-Upload-Header-Content-Type': 'application/jsonl'})
            with urllib.request.urlopen(req, timeout=60) as response:
                url = response.headers['x-goog-upload-url']
            req = urllib.request.Request(url, data=path.read_bytes(), headers={'Content-Type': 'application/jsonl', 'X-Goog-Upload-Command': 'upload, finalize', 'X-Goog-Upload-Offset': '0'}, method='POST')
            with urllib.request.urlopen(req, timeout=180) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))


def strict_json(text):
    def pairs(items):
        d = {}
        for k, v in items:
            require(k not in d, 'duplicate JSON object key')
            d[k] = v
        return d
    return json.loads(text, object_pairs_hook=pairs)


def validate(rr, mp):
    require(isinstance(rr, dict) and not rr.get('error'), 'API error')
    candidates = rr.get('response', {}).get('candidates', [])
    require(len(candidates) == 1, 'candidate response count is not one')
    ca = candidates[0]
    require(ca.get('finishReason') == 'STOP', 'non-STOP response')
    raw = ''.join(p.get('text', '') for p in ca.get('content', {}).get('parts', []) if not p.get('thought', False))
    obj = strict_json(raw)
    require(type(obj) is dict and obj.get('anchor_id') == mp['key'], 'anchor mismatch')
    require(type(obj.get('reviewed_count')) is int and obj['reviewed_count'] == mp['candidate_count'], 'reviewed_count mismatch')
    ids = obj.get('matching_ids')
    require(type(ids) is list and all(type(i) is int for i in ids), 'noninteger local IDs')
    require(len(ids) == len(set(ids)) and all(0 <= i < mp['candidate_count'] for i in ids), 'duplicate or out-of-range local IDs')
    return set(ids)


class Runner:
    def __init__(self, args):
        self.a = args
        self.root = Path(args.run_dir).resolve()
        self.dataset = Path(args.dataset).resolve()
        self.ledger = Path(args.ledger).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.prompts, self.schema = historical_prompts()
        self.parents = read(self.dataset / 'parents.json')
        self.units = read(self.dataset / 'units.json')
        self.edges = np.load(self.dataset / 'edges.npy', allow_pickle=False)
        e = self.edges
        require(e.ndim == 2 and e.shape[1] == 2 and e.dtype.kind in 'iu', 'edges must be integer Nx2 [unit_index,parent_index]')
        require(not len(e) or (e.min() >= 0 and e[:, 0].max() < len(self.units) and e[:, 1].max() < len(self.parents)), 'edge index outside input catalogs')
        require(len(np.unique(e, axis=0)) == len(e), 'duplicate candidate edges')
        for records, idcol, columns in [(self.parents, 'parent_id', ['parent_label', 'parent_scope']), (self.units, 'unit_id', ['phrase', 'kind'])]:
            require(len({r[idcol] for r in records}) == len(records), 'duplicate input IDs')
            for r in records:
                require(all(isinstance(r.get(k), str) for k in columns), 'missing or nonstring input field')
        for u in self.units:
            require(u['kind'] in ('short', 'long'), 'unit kind must be short or long')
            if needs_sentence_context(u):
                require(isinstance(u.get('original_sentence'), str), 'short unit requires original_sentence')
        self.config = {'model': MODEL, 'input_hashes': {name: sha(self.dataset / name) for name in ['parents.json', 'units.json', 'edges.npy']}, 'prompt_hashes': {k: digest(v.encode()) for k, v in self.prompts.items()}, 'schema': self.schema, 'limit': 128, 'repair_limits': [24, 8, 8], 'temperature': 0, 'thinkingBudget': 0, 'maxOutputTokens': 2048, 'shard_request_limit': 1000, 'shard_byte_limit': 10000000, 'method': 'parent_all_then_parent_positive_phrase_intersection', 'context_policy': CONTEXT_POLICY, 'word_pattern': WORD_PATTERN, 'parent_evidence': 'label_only', 'estimated_rates_per_million': [INPUT_RATE, OUTPUT_RATE], 'implementation_version': 2}
        self.freeze = digest(compact(self.config).encode())
        p = self.root / 'manifest.json'
        if p.exists():
            require(read(p)['config'] == self.config, 'Frozen input/prompt/config changed; use a new run directory')
        else:
            require(not any(x.is_dir() for x in self.root.iterdir()), 'Run directory already contains stages without a manifest')
            dump(p, {'freeze': self.freeze, 'config': self.config, 'created_at': time.time(), 'units': len(self.units), 'parents': len(self.parents), 'pairs': len(e)})
            for k, v in self.prompts.items():
                atomic(self.root / (k + '_prompt.txt'), v.encode())
        self.runid = self.root.name[:35] + '-' + self.freeze[:12] + '-' + digest(str(self.root).encode())[:8]

    def charge(self, rid, upper=None, cost=None, usage=None):
        import pipeline_common as C
        if upper is not None:
            C.reserve('gemini_batch',rid,upper)
        if cost is not None:
            usage=usage or {}
            C.settle(rid,cost,usage.get('promptTokenCount',0),usage.get('candidatesTokenCount',0)+usage.get('thoughtsTokenCount',0),'Historical parent-first reciprocal Batch runner')
        return C.budget()

    def folder(self, stage):
        require(re.fullmatch(r'(parent|phrase)(_repair[123])?', stage) is not None, 'Invalid stage')
        p = self.root / stage
        p.mkdir(exist_ok=True)
        return p

    def selected(self, stage):
        base = stage.split('_repair')[0]
        if '_repair' in stage:
            self.materialize(base)
            return np.flatnonzero(np.load(self.root / base / 'decisions.npy') == -1)
        if base == 'parent':
            return np.arange(len(self.edges))
        self.materialize('parent')
        d = np.load(self.root / 'parent/decisions.npy')
        require(np.isin(d, [0, 1]).all(), 'Parent must finish all repairs before preparing phrase stage')
        return np.flatnonzero(d == 1)

    def prepare(self, stage):
        f = self.folder(stage)
        if (f / 'manifest.json').exists():
            m = read(f / 'manifest.json')
            require(m['freeze'] == self.freeze, 'Stage freeze mismatch')
            for s in m['shards']:
                require(sha(f / s['input']) == s['input_sha256'] and sha(f / s['mapping']) == s['mapping_sha256'], 'Stage input/mapping corruption')
            return m
        base = stage.split('_repair')[0]
        selected = self.selected(stage)
        limit = 24 if stage.endswith('repair1') else 8 if stage.endswith(('repair2', 'repair3')) else 128
        groups = defaultdict(list)
        for eid in selected:
            u, p = map(int, self.edges[eid])
            groups[(p, needs_sentence_context(self.units[u])) if base == 'parent' else (u,)].append(int(eid))
        shards, records, maps = [], [], []
        bound, byte_count, n = 0.0, 0, 0
        def flush():
            nonlocal records, maps, bound, byte_count
            if not records:
                return
            i = len(shards)
            inp, mp = f'input_{i:03d}.jsonl', f'mapping_{i:03d}.json'
            atomic(f / inp, ''.join(compact(r) + '\n' for r in records).encode())
            dump(f / mp, maps)
            shards.append({'shard': i, 'input': inp, 'mapping': mp, 'requests': len(records), 'pairs': sum(len(x['edge_ids']) for x in maps), 'upper_cost_usd': bound, 'input_bytes': byte_count, 'input_sha256': sha(f / inp), 'mapping_sha256': sha(f / mp)})
            records, maps, bound, byte_count = [], [], 0.0, 0
        for key, es in sorted(groups.items()):
            for off in range(0, len(es), limit):
                ids = es[off:off + limit]
                u, p = map(int, self.edges[ids[0]])
                unit, par = self.units[u], self.parents[p]
                anchor = f'{stage}_{n:07d}'
                n += 1
                if base == 'parent':
                    payload = {'anchor_id': anchor, 'parent_label': par['parent_label'], 'candidate_columns': ['local_id', 'target_phrase'], 'candidates': [[j, self.units[int(self.edges[x, 0])]['phrase']] for j, x in enumerate(ids)]}
                    if needs_sentence_context(unit):
                        payload['candidate_columns'].append('original_sentence_context')
                        for card, edge_id in zip(payload['candidates'], ids):
                            card.append(self.units[int(self.edges[edge_id, 0])]['original_sentence'])
                else:
                    payload = {'anchor_id': anchor, 'target_phrase': unit['phrase'], 'candidate_columns': ['local_id', 'parent_label'], 'candidates': [[j, self.parents[int(self.edges[x, 1])]['parent_label']] for j, x in enumerate(ids)]}
                    if needs_sentence_context(unit):
                        payload['original_sentence_context'] = unit['original_sentence']
                prompt = self.prompts[base]
                if '_repair' in stage:
                    payload['candidate_count'] = len(ids)
                    prompt += f'\nThere are exactly {len(ids)} candidates. Return reviewed_count={len(ids)} and only local IDs from 0 through {len(ids)-1}.'
                req = {'systemInstruction': {'parts': [{'text': prompt}]}, 'contents': [{'role': 'user', 'parts': [{'text': compact(payload)}]}], 'generationConfig': {'temperature': 0, 'maxOutputTokens': 2048, 'thinkingConfig': {'thinkingBudget': 0}, 'responseMimeType': 'application/json', 'responseSchema': self.schema}}
                rec = {'key': anchor, 'request': req}
                size = len((compact(rec) + '\n').encode())
                require(size <= 10000000, 'Single request exceeds local 10 MB guard')
                if len(records) >= 1000 or byte_count + size > 10000000:
                    flush()
                records.append(rec)
                maps.append({'key': anchor, 'edge_ids': ids, 'candidate_count': len(ids)})
                byte_count += size
                bound += (len(compact(req).encode()) + 2048) * INPUT_RATE / 1e6 + 2048 * OUTPUT_RATE / 1e6
        flush()
        save(f / 'selected.npy', selected.astype(np.int64))
        m = {'stage': stage, 'freeze': self.freeze, 'selected_sha256': sha(f / 'selected.npy'), 'requests': n, 'pairs': len(selected), 'limit': limit, 'shards': shards, 'estimated_upper_cost_usd': sum(s['upper_cost_usd'] for s in shards)}
        dump(f / 'manifest.json', m)
        emit(event='prepared', **{k: v for k, v in m.items() if k != 'shards'})
        return m

    def submit(self, stage):
        require(self.a.budget_usd is not None, 'submit/run requires explicit --budget-usd')
        m, f = self.prepare(stage), self.folder(stage)
        import pipeline_common as C
        remaining=sum(s['upper_cost_usd'] for s in m['shards'] if not (f / f'created_{s["shard"]:03d}.json').exists())
        require(C.budget()['committed_upper_usd']+remaining<=C.CAP, 'Full stage would exceed existing shared Gemini allowance')
        for s in m['shards']:
            i = s['shard']
            if (f / f'created_{i:03d}.json').exists():
                continue
            intent = f / f'intent_{i:03d}.json'
            require(not intent.exists(), f'Unreconciled submission intent: {intent}. Never blindly retry a paid POST.')
            up = f / f'uploaded_{i:03d}.json'
            display = f'{self.runid}-{stage}-{i:03d}'
            if not up.exists():
                dump(up, upload(f / s['input'], display + '-input'))
            uploaded = read(up)['file']
            require(uploaded.get('state', 'ACTIVE') == 'ACTIVE', 'Uploaded file not ACTIVE; inspect Files API state before submission')
            body = {'batch': {'display_name': display, 'input_config': {'file_name': uploaded['name']}}}
            rid = 'gemini_batch:' + display
            self.charge(rid, upper=s['upper_cost_usd'])
            dump(intent, {'display_name': display, 'reservation_id': rid, 'created_at': time.time(), 'input_sha256': s['input_sha256'], 'model': MODEL, 'body': body})
            try:
                obj = post(f'{API}/models/{MODEL}:batchGenerateContent', body)
                require(isinstance(obj.get('name'), str), 'POST response lacks batch name; reconcile intent')
            except Exception as exc:
                detail = {'exception_type': type(exc).__name__, 'time': time.time(), 'message': str(exc)}
                if isinstance(exc, urllib.error.HTTPError):
                    detail['http_status'] = exc.code
                    detail['body'] = exc.read().decode(errors='replace')[:2000]
                    if exc.code in (400, 401, 403, 404, 422):
                        self.charge(rid, cost=0, usage={'submission_rejected_http': exc.code})
                dump(f / f'submit_error_{i:03d}.json', detail)
                raise RuntimeError(f'POST failed or uncertain; retained intent {intent}; reconcile before any retry') from exc
            dump(f / f'created_{i:03d}.json', obj)
            emit(event='submitted', stage=stage, shard=i, requests=s['requests'], batch=obj['name'])

    def collect(self, stage):
        m, f = self.prepare(stage), self.folder(stage)
        states = []
        for s in m['shards']:
            i = s['shard']
            dest, created = f / f'responses_{i:03d}.jsonl', f / f'created_{i:03d}.json'
            if dest.exists():
                receipt = read(f / f'download_{i:03d}.json') if (f / f'download_{i:03d}.json').exists() else None
                if receipt:
                    require(sha(dest) == receipt['sha256'], 'Downloaded output changed')
                else:
                    dump(f / f'download_{i:03d}.json', {'sha256': sha(dest), 'recovered_after_download': True})
                states.append({'shard': i, 'state': 'DOWNLOADED'})
                continue
            if not created.exists():
                states.append({'shard': i, 'state': 'NOT_SUBMITTED'})
                continue
            obj = get(f'{API}/{read(created)["name"]}')
            dump(f / f'status_{i:03d}.json', obj)
            meta = obj.get('metadata', {})
            state = meta.get('state', obj.get('state'))
            output = meta.get('output') or obj.get('output') or obj.get('response', {}).get('output') or obj.get('response') or {}
            file = output.get('responsesFile') or output.get('responses_file')
            states.append({'shard': i, 'state': state, 'stats': meta.get('batchStats', {}), 'error': obj.get('error')})
            if state in TERMINAL and file:
                raw = get(f'https://generativelanguage.googleapis.com/download/v1beta/{file}:download?alt=media', raw=True)
                atomic(dest, raw)
                dump(f / f'download_{i:03d}.json', {'sha256': sha(dest), 'file': file, 'time': time.time()})
        complete = all(s['state'] == 'DOWNLOADED' or s['state'] in TERMINAL for s in states)
        dump(f / 'status.json', {'complete': complete, 'shards': states, 'time': time.time()})
        self.parse(stage, complete)
        emit(event='collected', stage=stage, complete=complete, states=dict(Counter(s['state'] for s in states)))
        return complete

    def parse(self, stage, complete):
        m, f = self.prepare(stage), self.folder(stage)
        require(sha(f / 'selected.npy') == m['selected_sha256'], 'Selected set changed')
        selected = np.load(f / 'selected.npy')
        d = np.full(len(self.edges), -2, dtype=np.int8)
        d[selected] = -1
        errors, usage_total, versions, seen_edges = [], Counter(), Counter(), []
        cost_total = 0.0
        billing_unresolved = []
        for s in m['shards']:
            i = s['shard']
            maps = read(f / s['mapping'])
            expected = {mp['key'] for mp in maps}
            require(len(expected) == len(maps), 'Duplicate mapping keys')
            for mp in maps:
                require(len(mp['edge_ids']) == mp['candidate_count'], 'Mapping count mismatch')
                seen_edges.extend(mp['edge_ids'])
            dest = f / f'responses_{i:03d}.jsonl'
            rs, duplicates, envelope_errors = {}, set(), []
            if dest.exists():
                for lineno, line in enumerate(dest.read_text().splitlines(), 1):
                    if not line.strip():
                        continue
                    try:
                        rr = strict_json(line)
                        key = rr.get('key') or rr.get('metadata', {}).get('key')
                        require(key in expected, 'unknown or absent response key')
                        if key in rs:
                            duplicates.add(key)
                        rs[key] = rr
                    except Exception as exc:
                        envelope_errors.append({'line': lineno, 'error': str(exc)})
            usage, billable = Counter(), bool(dest.exists()) and not duplicates and not envelope_errors and set(rs) == expected
            for mp in maps:
                rr = rs.get(mp['key'], {})
                env = rr.get('response', {})
                um = env.get('usageMetadata', {})
                versions[str(env.get('modelVersion', ''))] += 1
                usage_ok = all(type(um.get(k)) is int and um[k] >= 0 for k in ('promptTokenCount', 'candidatesTokenCount')) and type(um.get('thoughtsTokenCount', 0)) is int and um.get('thoughtsTokenCount', 0) >= 0
                billable = billable and usage_ok
                if usage_ok:
                    usage.update({k: um.get(k, 0) for k in ('promptTokenCount', 'candidatesTokenCount', 'thoughtsTokenCount')})
                try:
                    require(not envelope_errors, 'malformed/unknown output envelope in shard')
                    require(mp['key'] not in duplicates, 'duplicate response key')
                    require(mp['key'] in rs, 'missing response key or terminal batch without output')
                    ids = validate(rr, mp)
                    d[mp['edge_ids']] = [int(j in ids) for j in range(mp['candidate_count'])]
                except Exception as exc:
                    errors.append({'shard': i, 'key': mp['key'], 'edge_ids': mp['edge_ids'], 'error': str(exc)})
            if envelope_errors:
                dump(f / f'envelope_errors_{i:03d}.json', envelope_errors)
            cost = (usage['promptTokenCount'] * INPUT_RATE + (usage['candidatesTokenCount'] + usage['thoughtsTokenCount']) * OUTPUT_RATE) / 1e6
            cost_total += cost
            usage_total.update(usage)
            if billable:
                rid = read(f / f'intent_{i:03d}.json')['reservation_id']
                self.charge(rid, cost=cost, usage=dict(usage))
            else:
                billing_unresolved.append(i)
        require(len(seen_edges) == len(set(seen_edges)) and np.array_equal(np.sort(seen_edges), selected), 'Mapping must cover selected edge IDs exactly once')
        save(f / 'raw_decisions.npy', d)
        dump(f / 'errors.json', errors)
        summary = {'stage': stage, 'complete': complete, 'pairs': len(selected), 'requests': m['requests'], 'counts': dict(Counter(map(int, d))), 'validation_errors': len(errors), 'usage': dict(usage_total), 'known_usage_estimated_cost_usd': cost_total, 'billing_unresolved_shards': billing_unresolved, 'model_versions': dict(versions)}
        dump(f / 'raw_summary.json', summary)
        self.materialize(stage)

    def materialize(self, stage):
        f = self.folder(stage)
        require((f / 'raw_decisions.npy').exists() and (f / 'raw_summary.json').exists(), f'{stage} has no parsed results')
        d = np.load(f / 'raw_decisions.npy').copy()
        summary = read(f / 'raw_summary.json')
        applied = read(f / 'applied_repairs.json') if (f / 'applied_repairs.json').exists() else []
        for entry in applied:
            rf = self.folder(entry['stage'])
            require(sha(rf / 'raw_decisions.npy') == entry['sha256'], 'Applied repair results changed')
            fixed = np.load(rf / 'raw_decisions.npy')
            covered = fixed != -2
            require(np.array_equal(np.flatnonzero(covered), np.flatnonzero(d == -1)), 'Repair must cover exactly current unresolved pairs')
            d[covered] = fixed[covered]
        summary.update(counts=dict(Counter(map(int, d))), applied_repairs=applied, unresolved_pairs=int((d == -1).sum()))
        save(f / 'decisions.npy', d)
        dump(f / 'summary.json', summary)
        return d

    def apply_repair(self, base, repair):
        require(repair in (base + '_repair1', base + '_repair2', base + '_repair3'), 'Invalid repair lineage')
        f, rf = self.folder(base), self.folder(repair)
        require(read(rf / 'raw_summary.json')['complete'], 'Repair batch still pending')
        require(read(f / 'raw_summary.json')['complete'], 'Base batch still pending')
        marker = f / 'applied_repairs.json'
        applied = read(marker) if marker.exists() else []
        if not any(x['stage'] == repair for x in applied):
            d, fixed = self.materialize(base), np.load(rf / 'raw_decisions.npy')
            require(np.array_equal(np.flatnonzero(fixed != -2), np.flatnonzero(d == -1)), 'Repair selection does not equal unresolved set')
            # Commit only append-only references first; derived files can always be replayed.
            applied.append({'stage': repair, 'sha256': sha(rf / 'raw_decisions.npy')})
            dump(marker, applied)
        d = self.materialize(base)
        emit(event='repair_applied', base=base, repair=repair, unresolved=int((d == -1).sum()))

    def finish(self, stage):
        self.prepare(stage)
        f = self.folder(stage)
        if (f / 'raw_summary.json').exists() and read(f / 'raw_summary.json')['complete']:
            self.materialize(stage)
            return True
        self.submit(stage)
        start = time.monotonic()
        while not self.collect(stage):
            if self.a.max_wait_seconds and time.monotonic() - start >= self.a.max_wait_seconds:
                emit(event='poll_deadline', stage=stage, resumable=True)
                return False
            time.sleep(self.a.poll_seconds)
        return True

    def final(self):
        pd = self.materialize('parent')
        fp = self.root / 'phrase/raw_decisions.npy'
        fd = self.materialize('phrase') if fp.exists() else np.full(len(pd), -2, dtype=np.int8)
        if fp.exists():
            require(np.array_equal(np.flatnonzero(fd != -2), np.flatnonzero(pd == 1)), 'Phrase coverage must equal parent positives')
        out = np.full(len(pd), -1, dtype=np.int8)
        out[pd == 0] = 0
        out[(pd == 1) & (fd == 0)] = 0
        out[(pd == 1) & (fd == 1)] = 1
        save(self.root / 'final_model_decisions.npy', out)
        stage_summaries = {p.name: read(p / 'summary.json') for p in self.root.iterdir() if p.is_dir() and (p / 'summary.json').exists()}
        summary = {'freeze': self.freeze, 'model': MODEL, 'method': self.config['method'], 'complete': not bool((out == -1).any()), 'counts': dict(Counter(map(int, out))), 'candidate_pairs': len(out), 'parent_positive_pairs': int((pd == 1).sum()), 'phrase_skipped_parent_negative_pairs': int((pd == 0).sum()), 'stages': stage_summaries, 'known_usage_estimated_cost_usd': sum(s['known_usage_estimated_cost_usd'] for s in stage_summaries.values()), 'ledger_path': str(self.ledger), 'final_sha256': sha(self.root / 'final_model_decisions.npy')}
        dump(self.root / 'summary.json', summary)
        emit(event='final_checkpoint', complete=summary['complete'], counts=summary['counts'])
        return summary

    def run(self):
        for base in ('parent', 'phrase'):
            if not self.finish(base):
                self.final()
                return
            for i in (1, 2, 3):
                d = self.materialize(base)
                if not (d == -1).any():
                    break
                repair = base + f'_repair{i}'
                applied = read(self.folder(base) / 'applied_repairs.json') if (self.folder(base) / 'applied_repairs.json').exists() else []
                if any(x['stage'] == repair for x in applied):
                    continue
                if not self.finish(repair):
                    self.final()
                    return
                self.apply_repair(base, repair)
            if (self.materialize(base) == -1).any():
                emit(event='unresolved_after_batch_repairs', stage=base, action='stop; no synchronous fallback; no forced negative')
                self.final()
                return
        self.final()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--ledger', default=str(Path(__file__).with_name('cost_ledger.json')))
    p.add_argument('--budget-usd', type=float)
    p.add_argument('--poll-seconds', type=float, default=40)
    p.add_argument('--max-wait-seconds', type=float, default=0, help='0 waits until terminal; positive limit checkpoints for later run')
    p.add_argument('command', choices=['prepare', 'submit', 'collect', 'apply-repair', 'run'])
    p.add_argument('--stage', default='parent')
    p.add_argument('--repair')
    args = p.parse_args()
    require(args.poll_seconds > 0 and args.max_wait_seconds >= 0, 'Invalid polling bounds')
    with locked(Path(args.run_dir) / '.runner.lock'):
        r = Runner(args)
        if args.command in ('prepare', 'submit', 'collect'):
            getattr(r, args.command)(args.stage)
        elif args.command == 'apply-repair':
            require(args.repair is not None, '--repair required')
            r.apply_repair(args.stage, args.repair)
        else:
            r.run()


if __name__ == '__main__':
    main()
