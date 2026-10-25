#!/usr/bin/env python3
"""Build a traceable World membership CSV from complete, compatible assignment runs."""
import argparse
import csv
import json
from pathlib import Path
from matrix import read, sha, dump, COMPLETE

def build(runs, output, scope):
    if output.exists():raise FileExistsError('Choose a fresh World directory')
    if not scope.strip():raise ValueError('Declare the available corpus scope')
    memberships=[];taxonomy_hash=None;provenance=[]
    for run in runs:
        final=run/'outputs/pipeline/final';v=json.loads((final/'validation.json').read_text())
        taxonomy=run/'inputs/final_taxonomy_enrich.csv';source=run/'inputs/source_original.csv';assigned=final/'all_assigned.csv'
        if not v['complete'] or v['unresolved']!=0:raise ValueError('World run is incomplete')
        for p,digest in [(taxonomy,v['taxonomy_sha256']),(source,v['source_original_sha256']),(assigned,v['output_hashes']['all_assigned.csv'])]:
            if sha(p)!=digest:raise ValueError('World input hash mismatch: '+p.name)
        if taxonomy_hash and taxonomy_hash!=sha(taxonomy):raise ValueError('World runs must use the same frozen taxonomy')
        taxonomy_hash=sha(taxonomy);parents={r['parent_id']:r['parent_label'].strip() for r in read(taxonomy)}
        originals=read(source);rows=read(assigned);seen=set()
        if len(rows)!=len(originals) or len(rows)!=v['source_rows']:raise ValueError('Source conservation failed')
        for row in rows:
            i=int(row['source_original_row'])-2
            if i in seen or not 0<=i<len(originals) or any(row.get(k)!=value for k,value in originals[i].items()):raise ValueError('Source mapping mismatch')
            seen.add(i);ids=json.loads(row['parent_ids_json']);labels=json.loads(row['parent_labels_json'])
            if len(ids)!=len(labels) or len(ids)!=int(row['parent_count']) or len(ids)!=len(set(ids)):raise ValueError('Parent list mismatch')
            if row['assignment_status'] not in COMPLETE or bool(ids)!=(row['assignment_status']=='assigned'):raise ValueError('Unconfirmed membership')
            for pid,label in zip(ids,labels):
                if parents.get(pid)!=label:raise ValueError('World parent mismatch')
                memberships.append(dict(firm=row['ticker'],year=int(row['year']),parent_id=pid,parent_label=label,
                    source_id=f"{v['industry']}:{v['source_original_sha256']}:{i+2}"))
        provenance.append(dict(industry=v['industry'],assignments_sha256=sha(assigned),validation_sha256=sha(final/'validation.json'),source_sha256=sha(source)))
    if taxonomy_hash is None:raise ValueError('At least one completed run is required')
    output.mkdir(parents=True)
    path=output/'world.csv'
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['firm','year','parent_id','parent_label','source_id']);w.writeheader();w.writerows(memberships)
    dump(output/'world.json',dict(taxonomy_sha256=taxonomy_hash,world_sha256=sha(path),basis='confirmed_assignments',scope_description=scope,source_runs=provenance))
    return len(memberships)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--scope-description',required=True)
    a=p.parse_args();print(build(a.run,a.output,a.scope_description))
