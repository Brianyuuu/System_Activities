import csv,json,os,sys,time,hashlib,fcntl
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pipeline_common as C
import gemini_parent_first_batch as B
DATA=C.RUN/'gemini_parent_first';OUT=C.OUT/'gemini_parent_first';DATA.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
SOURCE=C.OUT/'gemini_candidates.csv'
# Existing immutable datasets retain their original source and paid judgments.
if (DATA/'dataset_manifest.json').exists():SOURCE=Path(B.read(DATA/'dataset_manifest.json')['source'])
def secret():
    p = Path(os.environ['PIPELINE_SECRET_FILE'])
    assert p.stat().st_mode & 63 == 0
    os.environ['GEMINI_API_KEY'] = json.loads(p.read_text())['GEMINI_API_KEY']
def prepare_data():
    from run_glm_only import validate
    validate()
    if SOURCE == C.OUT/'gemini_candidates.csv':
        from gemini_candidate_policy import build_candidates
        build_candidates()
    src = SOURCE
    marker = DATA / 'dataset_manifest.json'
    if marker.exists():
        m = B.read(marker)
        assert B.sha(src) == m['source_sha256']
        return m
    assert B.sha(C.ROOT / 'inputs/final_taxonomy_enrich.csv') == C.CONFIG['taxonomy_original_sha256']
    assert B.sha(C.ROOT / 'inputs/parents.csv') == C.CONFIG['taxonomy_sha256']
    screening = B.read(C.OUT / 'filter_summary.json')
    assert screening['pending'] == 0 and screening['deferred_pairs'] == 0, 'Unresolved GLM pairs; do not silently finalize'
    assert screening['completed_pairs'] == screening['candidate_pairs']
    units = B.read(C.RUN / 'units.json')
    with (C.OUT / 'parents_snapshot.csv').open(encoding='utf-8-sig', newline='') as f:
        parents = list(csv.DictReader(f))
    um = {u['unit_id']: i for i, u in enumerate(units)}
    pm = {p['parent_id']: i for i, p in enumerate(parents)}
    edges = []
    with src.open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            ui = um[r['unit_id']]
            pi = pm[r['parent_id']]
            u = units[ui]
            p = parents[pi]
            assert r['phrase'] == u['phrase'] and r['parent_label'] == p['parent_label'] and (r['parent_scope'] == p['parent_scope'])
            assert r['original_sentence'] == u.get('original_sentence', '')
            edges.append([ui, pi])
    assert len({tuple(x) for x in edges}) == len(edges)
    B.dump(DATA / 'parents.json', parents)
    B.dump(DATA / 'units.json', units)
    B.save(DATA / 'edges.npy', np.array(edges, dtype=np.int32))
    m = dict(source=str(src), source_sha256=B.sha(src), frozen_sha256=C.CONFIG['frozen_sha256'], candidate_pairs=len(edges), units=len(units), parents=len(parents), method='Parent first on candidate union (GLM kept plus mandatory embedding top30); phrase reciprocal only on parent-positive pairs; intersection required', context_policy=B.CONTEXT_POLICY, reference_status='bundled provenance/raise_from_phrase_work_status.md', reference_status_sha256=os.environ.get('REFERENCE_STATUS_SHA256', ''), created_at=time.time())
    B.dump(marker, m)
    return m
def runner():
    args = SimpleNamespace(dataset=str(DATA), run_dir=str(DATA), ledger=str(C.RUN / 'budget.sqlite'), budget_usd=C.CAP, poll_seconds=300, max_wait_seconds=0)
    return B.Runner(args)
def export():
    summary = B.read(DATA / 'summary.json')
    B.dump(OUT / 'summary.json', summary)
    if not summary['complete']:
        raise RuntimeError('Remaining unresolved judgments; do not finalize')
    pd = np.load(DATA / 'parent/decisions.npy')
    fd = np.load(DATA / 'phrase/decisions.npy')
    final = np.load(DATA / 'final_model_decisions.npy')
    assert np.isin(pd, [0, 1]).all() and np.array_equal(np.flatnonzero(fd != -2), np.flatnonzero(pd == 1))
    assert np.array_equal(final == 1, (pd == 1) & (fd == 1))
    n = 0
    with SOURCE.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames + ['gemini_parent_decision', 'gemini_phrase_decision', 'gemini_final_decision']
        with (OUT / 'all_pair_decisions.tmp').open('w', encoding='utf-8-sig', newline='') as g, (OUT / 'accepted_pairs.tmp').open('w', encoding='utf-8-sig', newline='') as h:
            allw = csv.DictWriter(g, fieldnames=fields)
            keepw = csv.DictWriter(h, fieldnames=fields)
            allw.writeheader()
            keepw.writeheader()
            for i, row in enumerate(reader):
                row.update(gemini_parent_decision=int(pd[i]), gemini_phrase_decision='' if fd[i] == -2 else int(fd[i]), gemini_final_decision=int(final[i]), status='gemini_accepted' if final[i] == 1 else 'gemini_rejected')
                allw.writerow(row)
                if final[i] == 1:
                    keepw.writerow(row)
                    n += 1
    for x in ['all_pair_decisions', 'accepted_pairs']:
        (OUT / (x + '.tmp')).replace(OUT / (x + '.csv'))
    assert n == int((final == 1).sum())
    B.dump(OUT / 'validation.json', dict(pass_validation=True, candidate_pairs=len(final), parent_positives=int((pd == 1).sum()), phrase_reviewed=int((fd != -2).sum()), phrase_skipped=int((fd == -2).sum()), accepted=n, unresolved=0, frozen_sha256=C.CONFIG['frozen_sha256'], time=time.time()))
    print('VALIDATED_FINAL', n, flush=True)
if __name__=='__main__':
 with (DATA/'.controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);secret();m=prepare_data();r=runner()
  model=B.get(B.API+'/models/'+B.MODEL);assert 'batchGenerateContent' in model['supportedGenerationMethods'];B.dump(OUT/'model_verified.json',{k:model.get(k) for k in ['name','version','supportedGenerationMethods']})
  stage=r.prepare('parent');B.dump(OUT/'prepare_summary.json',dict(dataset=m,parent_stage={k:v for k,v in stage.items() if k!='shards'},parent_shards=len(stage['shards'])));print('PARENT_PLAN',stage['pairs'],stage['requests'],stage['estimated_upper_cost_usd'],flush=True)
  r.run();export()
