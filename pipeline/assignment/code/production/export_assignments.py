import csv,json,hashlib
from collections import defaultdict,Counter
import numpy as np
import pipeline_common as C
from audit import read

def main():
    from run_glm_only import validate
    validate()
    pointer=C.OUT/'gemini_active_results.json'
    active=json.loads(pointer.read_text()) if pointer.exists() else {}
    gemini_out=__import__('pathlib').Path(active.get('output_dir',str(C.OUT/'gemini_parent_first')))
    candidate_source=__import__('pathlib').Path(active.get('candidate_source',str(C.OUT/'gemini_candidates.csv')))
    if not candidate_source.exists():candidate_source=C.OUT/'glm_kept_candidates.csv'
    verify=json.loads((gemini_out/'validation.json').read_text());assert verify['pass_validation'] and verify['unresolved']==0
    screening=json.loads((C.OUT/'filter_summary.json').read_text());assert screening['pending']==0 and screening['deferred_pairs']==0
    rows=read(C.ROOT/'inputs/phrases.csv'); originals=read(C.ROOT/'inputs/source_original.csv');assert len(rows)==len(originals)==C.CONFIG['source_rows']
    for row,original in zip(rows,originals):
        assert all(row[k]==v for k,v in original.items()),'Original source data changed'
    units=json.loads((C.RUN/'units.json').read_text());mapping={}
    for u in units:
        for i in u['source_rows']:
            assert i not in mapping
            mapping[i]=u['unit_id']
    assert set(mapping)==set(range(2,len(rows)+2))
    kept=Counter(); accepted=defaultdict(list)
    with (C.OUT/'glm_kept_candidates.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):kept[r['unit_id']]+=1
    reviewed=Counter()
    with candidate_source.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):reviewed[r['unit_id']]+=1
    assert sum(reviewed.values())==verify['candidate_pairs']
    with (gemini_out/'accepted_pairs.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):accepted[r['unit_id']].append((r['parent_id'],r['parent_label']))
    assert sum(map(len,accepted.values()))==verify['accepted']
    for pairs in accepted.values():assert len(pairs)==len(set(pairs))
    offsets=np.load(C.OUT/'embedding_candidates.npz')['offsets']; recalled={u['unit_id']:int(offsets[i+1]-offsets[i]) for i,u in enumerate(units)}
    output=C.OUT/'final';output.mkdir(exist_ok=True)
    statuses=Counter(); edges=0
    common=list(rows[0])+['assignment_unit_id']
    with (output/'source_parent_assignments.csv.tmp').open('w',encoding='utf-8-sig',newline='') as f,(output/'all_assigned.csv.tmp').open('w',encoding='utf-8-sig',newline='') as g:
        ew=csv.DictWriter(f,fieldnames=common+['parent_id','parent_label','assignment_status']);ew.writeheader()
        sw=csv.DictWriter(g,fieldnames=common+['parent_ids_json','parent_labels_json','parent_count','assignment_status','embedding_candidate_count','glm_kept_count']);sw.writeheader()
        for i,row in enumerate(rows,2):
            uid=mapping[i];pairs=accepted[uid]
            status='assigned' if pairs else 'rejected_by_gemini' if reviewed[uid] else 'no_embedding_candidate' if not recalled[uid] else 'rejected_by_glm'
            statuses[status]+=1;data=dict(row,assignment_unit_id=uid)
            for pid,label in pairs:ew.writerow(dict(data,parent_id=pid,parent_label=label,assignment_status='gemini_bidirectional_accepted'));edges+=1
            sw.writerow(dict(data,parent_ids_json=json.dumps([p[0] for p in pairs]),parent_labels_json=json.dumps([p[1] for p in pairs]),parent_count=len(pairs),assignment_status=status,embedding_candidate_count=recalled[uid],glm_kept_count=kept[uid]))
    for name in ['source_parent_assignments.csv','all_assigned.csv']:(output/(name+'.tmp')).replace(output/name)
    result=dict(complete=True,industry=C.CONFIG['industry'],source_rows=len(rows),assignment_units=len(units),parents=C.CONFIG['parents'],firms=len({r['ticker'] for r in rows}),firm_years=len({(r['ticker'],r['year']) for r in rows}),accepted_unit_parent_pairs=verify['accepted'],accepted_source_parent_pairs=edges,source_status_counts=dict(statuses),unresolved=0,taxonomy_sha256=C.CONFIG['taxonomy_original_sha256'],identity_version=C.CONFIG['identity_version'],source_original_sha256=C.CONFIG['source_original_sha256'],budget=C.budget(),output_hashes={name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in ['source_parent_assignments.csv','all_assigned.csv']})
    result.update(gemini_candidate_pairs=verify['candidate_pairs'],gemini_results_dir=str(gemini_out),candidate_policy='GLM kept UNION mandatory embedding top30' if (C.OUT/'gemini_candidates_manifest.json').exists() else 'GLM kept')
    assert result['firms']==C.CONFIG['firms'] and result['firm_years']==C.CONFIG['firm_years']
    C.dump(output/'validation.json',result);print('FINAL_EXPORT',json.dumps(result),flush=True)
if __name__=='__main__':main()
