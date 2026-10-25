import csv,json,re,base64,time
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
from pipeline_common import *
from audit import read,write,SRC
PREV=ROOT/'inputs'
TAX=ROOT/'inputs/parents.csv'
MODEL='text-embedding-3-small';DIM=1536
def get_vectors(texts):
    lookup={};provenance=[]
    caches=sorted((ROOT/'inputs/embedding_cache').glob('*.npz'))
    target=set(texts)
    for p in caches:
        if not p.exists():continue
        saved=np.load(p);assert saved['vectors'].shape[1]==DIM
        for text,vec in zip(saved['texts'],saved['vectors']):
            if str(text) in target:lookup[str(text)]=vec.copy()
        provenance.append(str(p))
    unique=sorted(target);cache=RUN/'retrieval_embeddings';cache.mkdir(exist_ok=True)
    # Each cache shard stores exact text and requested model metadata.
    for p in cache.glob('*.npz'):
        saved=np.load(p);assert str(saved['model'])==MODEL
        for text,vec in zip(saved['texts'],saved['vectors']):lookup[str(text)]=vec
    missing=[p for p in unique if p not in lookup];print('embedding missing',len(missing),'cached',len(target)-len(missing),flush=True)
    def embed(batch):
        key=digest([MODEL,DIM,batch]);path=cache/(key+'.npz');body={'model':MODEL,'input':batch,'dimensions':DIM,'encoding_format':'base64'}
        upper=(len(json.dumps(body,ensure_ascii=False).encode())+2048)*.02/1e6
        for attempt in range(4):
            rid=f'embedding:{key}:{attempt}'
            with db() as c:old=c.execute('SELECT status FROM charges WHERE id=?',(rid,)).fetchone()
            if old:continue
            reserve('retrieval_embedding',rid,upper)
            try:
                env,_=post('https://api.openai.com/v1/embeddings',body,os.environ['OPENAI_API_KEY'])
                tok=env['usage']['total_tokens'];settle(rid,tok*.02/1e6,tok)
                data=sorted(env['data'],key=lambda r:r['index']);assert [r['index'] for r in data]==list(range(len(batch)))
                v=np.vstack([np.frombuffer(base64.b64decode(r['embedding']),dtype='<f4') for r in data]);assert v.shape==(len(batch),DIM) and np.isfinite(v).all()
                v/=np.linalg.norm(v,axis=1,keepdims=True);np.savez_compressed(path,texts=np.asarray(batch),vectors=v,model=MODEL)
                return batch,v
            except urllib.error.HTTPError as e:
                if e.code in (400,401,403,404):settle(rid,0,note='Rejected');raise
                print('embedding_retry',e.code,flush=True)
            except BudgetStop:raise
            time.sleep(2**attempt)
        raise RuntimeError('embedding failed '+key)
    batches=[missing[i:i+128] for i in range(0,len(missing),128)]
    with ThreadPoolExecutor(max_workers=6) as ex:
        for n,f in enumerate(as_completed([ex.submit(embed,b) for b in batches]),1):
            batch,v=f.result();lookup.update(zip(batch,v))
            if n%10==0:print('embedding_batches',n,'/',len(batches),flush=True)
    dump(RUN/'retrieval_embedding_provenance.json',dict(model=MODEL,dimensions=DIM,caches=provenance,new_texts=len(missing),total_texts=len(unique)))
    budget();return lookup
def select_candidates(scores):
    if CONFIG.get('recall_policy', 'legacy') == 'top1000':
        order=np.lexsort((np.arange(len(scores)), -scores))[:1000]
        return np.sort(order[scores[order] > ACTIVE_SIMILARITY_THRESHOLD])
    return np.flatnonzero(scores >= ACTIVE_SIMILARITY_THRESHOLD)

def main():
    units=json.loads((RUN/'units.json').read_text());parents=read(TAX)
    assert len(parents)==CONFIG["parents"] and len({p['parent_id'] for p in parents})==len(parents)
    write(OUT/'parents_snapshot.csv',parents)
    contexts=[]
    for u in units:
        u['context_texts']=[f'Phrase: {p}\nContext: {u["original_sentence"]}' for p in u['aliases']] if u['kind']=='short' else []
        contexts.extend(u['context_texts'])
    texts=[p[f].strip() for p in parents for f in ('parent_label','parent_scope')]+[p for u in units for p in u['aliases']]+contexts
    lookup=get_vectors(texts)
    label=np.vstack([lookup[p['parent_label'].strip()] for p in parents]);scope=np.vstack([lookup[p['parent_scope'].strip()] for p in parents])
    assert np.isfinite(label).all() and np.isfinite(scope).all()
    offsets=[0];pis=[];scores=[];counts=[]; thresholds={t:0 for t in [.29,.30,.31,.32,.33]}
    # Compute every alias/view. Candidate union cannot lose a member's recalled parent after merging.
    for start in range(0,len(units),64):
        chunk=units[start:start+64];views=[p for u in chunk for p in u['aliases']+u['context_texts']]
        vv=np.vstack([lookup[p] for p in views]);sim=np.maximum(vv@label.T,vv@scope.T);at=0
        for u in chunk:
            n=len(u['aliases'])+len(u['context_texts']);s=sim[at:at+n].max(axis=0);at+=n
            for t in thresholds:thresholds[t]+=int((s>=t).sum())
            ix=select_candidates(s);pis.append(ix.astype(np.int32));scores.append(s[ix].astype(np.float32));counts.append(len(ix));offsets.append(offsets[-1]+len(ix))
        if start%1024==0:print('recall_units',start,'/',len(units),'pairs',offsets[-1],flush=True)
    np.savez_compressed(OUT/'embedding_candidates.npz',offsets=np.array(offsets,dtype=np.int64),parent_indices=np.concatenate(pis),scores=np.concatenate(scores))
    # Context views are frozen during preparation; never mutate the unit snapshot on resume.
    summary=dict(model=MODEL,dimensions=DIM,threshold=ACTIVE_SIMILARITY_THRESHOLD,rule=('score > .33 AND global top1000; tie smaller parent index' if CONFIG.get('recall_policy') == 'top1000' else 'score >= .33; no top-k filtering'),parents=len(parents),units=len(units),candidate_pairs=offsets[-1],units_with_candidates=sum(c>0 for c in counts),threshold_counts=thresholds,max_candidates=max(counts),mean_candidates=sum(counts)/len(counts),p50_candidates=float(np.median(counts)),p95_candidates=float(np.quantile(counts,.95)),taxonomy_sha256=hashlib.sha256(TAX.read_bytes()).hexdigest(),source_sha256=hashlib.sha256((ROOT/'inputs/phrases.csv').read_bytes()).hexdigest())
    dump(OUT/'recall_summary.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
