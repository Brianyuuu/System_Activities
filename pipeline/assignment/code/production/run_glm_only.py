import os,json,sys,hashlib,fcntl,signal,csv,time
from pathlib import Path
import numpy as np
import pipeline_common as C
import pipeline_recall as R
import pipeline_filter as F
from audit import read,write

def validate():
 if not __debug__:raise RuntimeError('Python -O disables required validation')
 prepared=json.loads((C.ROOT/'preparation_manifest.json').read_text())
 assert C.CONFIG==prepared['config'],'Frozen run configuration changed; use a new run'
 for name,digest in prepared['hashes'].items():
  if name=='work/pipeline/glm_worker_target.json':continue # concurrency is operational, not semantic
  assert hashlib.sha256((C.ROOT/name).read_bytes()).hexdigest()==digest,'Frozen run file changed: '+name
 for name,key in [('phrases.csv','source_sha256'),('parents.csv','taxonomy_sha256'),('final_taxonomy_enrich.csv','frozen_sha256')]:
  assert hashlib.sha256((C.ROOT/'inputs'/name).read_bytes()).hexdigest()==C.CONFIG[key],name+' hash mismatch'
 rows=read(C.ROOT/'inputs/phrases.csv');parents=read(C.ROOT/'inputs/parents.csv');units=json.loads((C.RUN/'units.json').read_text())
 assert len(rows)==C.CONFIG['source_rows'] and all(r['status']=='fully_verified' for r in rows)
 assert len(parents)==C.CONFIG['parents'] and len({x['parent_id'] for x in parents})==len(parents)
 assert len(units)==C.CONFIG['assignment_units'] and sorted({i for u in units for i in u['source_rows']})==list(range(2,len(rows)+2))
 assert not C.CONFIG['gemini_enabled']
 assert len([i for u in units for i in u['source_rows']])==len(rows)
 assert hashlib.sha256((C.ROOT/'inputs/source_original.csv').read_bytes()).hexdigest()==C.CONFIG['source_original_sha256']
 print('VALIDATED',len(rows),'source rows',len(units),'units',len(parents),'parents',flush=True)

def checkpoint_signal(signum,frame):
 raise KeyboardInterrupt('Checkpoint requested by signal '+str(signum))
def main():
 for sig in [signal.SIGINT,signal.SIGTERM,signal.SIGUSR1]:signal.signal(sig,checkpoint_signal)
 with (C.RUN/'workflow.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);validate()
  if '--offline' in sys.argv:return
  secret=Path(os.environ['PIPELINE_SECRET_FILE']);assert secret.stat().st_mode&0o077==0
  data=json.loads(secret.read_text())
  for key in ['OSS_API_KEY','OPENAI_API_KEY']:
   assert data.get(key),'Missing '+key
   os.environ[key]=data[key]
  print('START EMBEDDING RECALL',flush=True)
  if not (C.OUT/'recall_summary.json').exists():R.main()
  print('START GLM SCREENING',flush=True)
  F.main()
  decisions=np.load(C.OUT/'glm_filter_decisions.npy');edges=np.load(C.OUT/'embedding_candidates.npz');units=json.loads((C.RUN/'units.json').read_text());parents=read(C.OUT/'parents_snapshot.csv');ui=np.load(C.RUN/'edge_unit_indices.npy')
  parent_indices=edges["parent_indices"];similarity_scores=edges["scores"]
  # Stream candidates to avoid materializing millions of dictionaries.
  fields=['unit_id','canonical_id','phrase','original_sentence','parent_id','parent_label','parent_scope','parent_exclusion','embedding_similarity','status']
  for code,name,status in [(1,'glm_kept_candidates.csv','glm_kept_pending_final_judgment'),(-2,'glm_deferred_candidates.csv','unresolved_glm_error')]:
   path=C.OUT/name;tmp=path.with_suffix('.tmp')
   with tmp.open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for e in np.flatnonzero(decisions==code):
     u=units[int(ui[e])];p=parents[int(parent_indices[e])]
     w.writerow(dict(unit_id=u['unit_id'],canonical_id=u['canonical_id'],phrase=u['phrase'],original_sentence=u['original_sentence'],**{k:p[k] for k in ['parent_id','parent_label','parent_scope','parent_exclusion']},embedding_similarity=float(similarity_scores[e]),status=status))
   tmp.replace(path)
  C.dump(C.OUT/'glm_only_status.json',dict(stage='glm_screening_finished',gemini_enabled=False,counts={str(x):int((decisions==x).sum()) for x in [-2,-1,0,1]},budget=C.budget(),updated_at=time.time()))
def run_with_recovery():
 # F.main drains all in-flight calls and saves their successes before propagating
 # a transport error. Re-enter only after that checkpoint and a bounded cooldown.
 state_path=C.OUT/'glm_recovery_status.json'
 history=[]
 max_recoveries=8
 for cycle in range(max_recoveries+1):
  try:
   if '--offline' not in sys.argv and C.GLM_DEFER_TRANSPORT:
    C.dump(state_path,dict(status='running',retry_policy='failed_requests_at_end_of_sweep',updated_at=time.time()))
   main()
   if '--offline' not in sys.argv:
    C.dump(state_path,dict(status='complete',recoveries=history,updated_at=time.time()))
   return
  except C.GLMTransportError as error:
   if cycle==max_recoveries:
    C.dump(state_path,dict(status='retry_limit_reached',recoveries=history,error=str(error),updated_at=time.time()))
    raise
   delay=min(60*2**cycle,900)
   entry=dict(cycle=cycle+1,delay_seconds=delay,error=str(error),checkpointed_at=time.time())
   history.append(entry)
   C.dump(state_path,dict(status='cooldown',recoveries=history,retry_at=time.time()+delay,updated_at=time.time()))
   print('GLM RECOVERY COOLDOWN',json.dumps(entry),flush=True)
   time.sleep(delay)
   C.GLM_NETWORK_DOWN.clear()
   C.dump(state_path,dict(status='resuming',recoveries=history,updated_at=time.time()))
  except Exception as error:
   C.dump(state_path,dict(status='failed_nonrecoverable',recoveries=history,error_type=type(error).__name__,updated_at=time.time()))
   raise

if __name__=='__main__':
 try:run_with_recovery()
 except KeyboardInterrupt:
  C.dump(C.OUT/'checkpointed_interruption.json',dict(status='checkpointed',time=time.time(),workers=F.WORKERS))
  print('CHECKPOINTED INTERRUPTION',flush=True)
  raise SystemExit(2)
