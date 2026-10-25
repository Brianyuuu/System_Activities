"""Mandatory embedding top-30 union, without changing GLM judgments or Gemini prompts."""
import csv,json,hashlib,time
from pathlib import Path
import numpy as np
import pipeline_common as C

K=30
FIELDS=['unit_id','canonical_id','phrase','original_sentence','parent_id','parent_label','parent_scope','parent_exclusion','embedding_similarity','status']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def readcsv(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def rank(indices,scores,k):
 assert len(indices)==len(scores) and np.isfinite(scores).all()
 return np.lexsort((indices,-scores))[:k]
def loadtop(path):
 with np.load(path) as f:return {k:f[k] for k in f.files}
def cached_vectors(texts):
 wanted=set(texts);lookup={}
 paths=sorted((C.ROOT/'inputs/embedding_cache').glob('*.npz'))+sorted((C.RUN/'retrieval_embeddings').glob('*.npz'))
 for path in paths:
  with np.load(path) as saved:
   tx=saved['texts'];hits=[(i,str(t)) for i,t in enumerate(tx) if str(t) in wanted]
   if hits:
    vectors=saved['vectors']
    for i,t in hits:lookup[t]=vectors[i].copy()
 missing=wanted-lookup.keys()
 assert not missing,f'Missing {len(missing)} existing embedding vectors; no automatic new embedding API calls'
 return lookup
def select_top30():
 units=json.loads((C.RUN/'units.json').read_text());parents=readcsv(C.OUT/'parents_snapshot.csv')
 source=C.OUT/'embedding_candidates.npz';dest=C.OUT/'embedding_top30.npz';marker=C.OUT/'embedding_top30_manifest.json'
 hashes={str(p.name):sha(p) for p in [source,C.RUN/'units.json',C.OUT/'parents_snapshot.csv']}
 config=dict(k=K,input_hashes=hashes,tie_break='ascending frozen parent index',score_rule='existing max over alias/context versus parent label/scope; top30 over all parents regardless of threshold')
 if marker.exists():
  m=json.loads(marker.read_text());assert m['config']==config;assert sha(dest)==m['output_sha256'];return units,parents,loadtop(dest),m
 k=min(K,len(parents));indices=np.empty((len(units),k),np.int32);values=np.empty((len(units),k),np.float32);edgeids=np.full((len(units),k),-1,np.int64)
 with np.load(source) as ed:
  offsets=ed['offsets'];pi=ed['parent_indices'];scores=ed['scores'];short=np.flatnonzero(np.diff(offsets)<k)
  lookup=None
  if len(short):
   texts=[p[f].strip() for p in parents for f in ['parent_label','parent_scope']]
   texts.extend(v for uidx in short for v in units[int(uidx)]['aliases']+units[int(uidx)].get('context_texts',[]))
   lookup=cached_vectors(texts);label=np.vstack([lookup[p['parent_label'].strip()] for p in parents]);scope=np.vstack([lookup[p['parent_scope'].strip()] for p in parents])
  for i,u in enumerate(units):
   a,b=int(offsets[i]),int(offsets[i+1]);ip=pi[a:b];sc=scores[a:b]
   if b-a>=k:
    take=rank(ip,sc,k);indices[i]=ip[take];values[i]=sc[take];edgeids[i]=a+take
   else:
    vv=np.vstack([lookup[v] for v in u['aliases']+u.get('context_texts',[])]);allsc=np.maximum(vv@label.T,vv@scope.T).max(axis=0).astype(np.float32)
    # Retain recorded scores for recalled edges; recompute only the unrecorded tail.
    allsc[ip]=sc;allpi=np.arange(len(parents),dtype=np.int32);take=rank(allpi,allsc,k);indices[i]=take;values[i]=allsc[take];mapping={int(p):a+j for j,p in enumerate(ip)};edgeids[i]=[mapping.get(int(p),-1) for p in take]
  assert indices.shape==(len(units),k) and all(len(set(row))==k for row in indices)
  tmp=dest.with_suffix('.tmp')
  with tmp.open('wb') as f:np.savez_compressed(f,parent_indices=indices,scores=values,embedding_edge_indices=edgeids)
  tmp.replace(dest)
 m=dict(config=config,units=len(units),parents=len(parents),mandatory_pairs=int(indices.size),units_requiring_below_threshold_search=int(len(short)),pairs_outside_original_recall=int((edgeids<0).sum()),output_sha256=sha(dest),created_at=time.time())
 C.dump(marker,m);return units,parents,loadtop(dest),m
def candidate_row(u,p,score):
 return dict(unit_id=u['unit_id'],canonical_id=u['canonical_id'],phrase=u['phrase'],original_sentence=u['original_sentence'],**{k:p[k] for k in ['parent_id','parent_label','parent_scope','parent_exclusion']},embedding_similarity=float(score),status='embedding_top30_pending_final_judgment')
def build_candidates():
 screening=json.loads((C.OUT/'filter_summary.json').read_text());assert screening['pending']==screening['deferred_pairs']==0
 units,parents,top,topmeta=select_top30();src=C.OUT/'glm_kept_candidates.csv';dest=C.OUT/'gemini_candidates.csv';marker=C.OUT/'gemini_candidates_manifest.json'
 config=dict(policy='GLM kept UNION mandatory embedding top30',k=K,glm_source_sha256=sha(src),top30_sha256=topmeta['output_sha256'])
 if marker.exists():
  m=json.loads(marker.read_text());assert m['config']==config and sha(dest)==m['output_sha256'];return m
 um={u['unit_id']:i for i,u in enumerate(units)};pm={p['parent_id']:i for i,p in enumerate(parents)};seen=set();P=len(parents);kept=0;already=0;added=0
 tmp=dest.with_suffix('.tmp')
 with tmp.open('w',encoding='utf-8-sig',newline='') as g:
  w=csv.DictWriter(g,fieldnames=FIELDS);w.writeheader()
  with src.open(encoding='utf-8-sig',newline='') as f:
   for row in csv.DictReader(f):
    key=um[row['unit_id']]*P+pm[row['parent_id']];assert key not in seen;seen.add(key);w.writerow(row);kept+=1
  assert kept==screening['kept']
  for i,u in enumerate(units):
   for p,s in zip(top['parent_indices'][i],top['scores'][i]):
    key=i*P+int(p)
    if key in seen:already+=1;continue
    seen.add(key);w.writerow(candidate_row(u,parents[int(p)],s));added+=1
 tmp.replace(dest)
 assert kept+added==len(seen) and already+added==topmeta['mandatory_pairs']
 m=dict(config=config,glm_kept_pairs=kept,mandatory_top30_pairs=topmeta['mandatory_pairs'],top30_already_glm_kept=already,top30_added=added,candidate_pairs=len(seen),output_sha256=sha(dest),created_at=time.time(),below_threshold_pairs=topmeta['pairs_outside_original_recall'])
 C.dump(marker,m);print('TOP30_CANDIDATES',json.dumps(m),flush=True);return m
if __name__=='__main__':build_candidates()
