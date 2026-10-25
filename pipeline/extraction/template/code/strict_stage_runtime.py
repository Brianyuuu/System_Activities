"""Strict stage continuation: errors remain pending, never model decisions."""
import os,sys,json,time,hashlib,subprocess,re,datetime
from pathlib import Path
R=Path(__file__).resolve().parents[1]
BANNED={'--fallback-to-source-on-failure','--fallback-to-reject-on-failure','--fallback-to-parent-on-failure','--fallback-to-no-replacements-on-failure','--conservative-no-merge-on-failure'}
INPUTS={'--input','--input-csv','--manifest-csv','--prompt','--prompt-file','--repair-prompt-file','--system-prompt-file','--semantic-prompt'}
OUTPUTS={'--output','--output-dir','--map-output','--kept-output','--audit-output','--summary-output'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def output_files(cmd):
 out=[]
 for n,x in enumerate(cmd[:-1]):
  if x in OUTPUTS:
   p=Path(cmd[n+1])
   if p.is_file():out.append(p)
   elif p.is_dir():out.extend(f for f in p.iterdir() if f.is_file() and f.suffix in ('.csv','.json'))
 return sorted(set(out))
def normalize(cmd):
 cmd=[x for x in cmd if x not in BANNED and x!='--no-resume']
 if Path(cmd[1]).name=='05_decompose_mechanism_phrases.py':
  cmd[1]=str(R/'code/strict_runtime/05_decompose_mechanism_phrases.py')
 return cmd
def execute(command,env,state_dir,context):
 cmd=normalize(command);d=Path(state_dir);d.mkdir(parents=True,exist_ok=True)
 inputs={cmd[n+1]:sha(cmd[n+1]) for n,x in enumerate(cmd[:-1]) if x in INPUTS and Path(cmd[n+1]).is_file()}
 fingerprint=hashlib.sha256(json.dumps({'cmd':cmd,'script_sha':sha(cmd[1]),'inputs':inputs,'context':context},sort_keys=True).encode()).hexdigest()
 receipt=d/(fingerprint+'.json')
 prior=json.loads(receipt.read_text()) if receipt.exists() else {}
 if prior.get('status')=='complete' and prior.get('outputs') and all(Path(p).is_file() and sha(p)==h for p,h in prior['outputs'].items()):
  print('Strict checkpoint reused:',Path(cmd[1]).name,flush=True);return
 attempt=int(prior.get('attempts',0))
 while True:
  if attempt >= 3: raise RuntimeError("Stage exhausted 3 attempts; checkpoints preserved for review")
  attempt+=1;log=d/(fingerprint+f'.attempt{attempt}.log')
  rec={'status':'running','stage':Path(cmd[1]).name,'command':cmd,'context':context,'inputs':inputs,'attempts':attempt,'started_utc':now(),'job_id':os.getenv('SLURM_JOB_ID'),'log':str(log),'failure_fallback_enabled':False}
  save(receipt,rec)
  print('Strict stage',rec['stage'],'attempt',attempt,flush=True)
  child_env=dict(env);child_env['PHRASE_REPAIR_REUSE_OUTPUT']='1'
  with log.open('x') as f:rc=subprocess.run(cmd,env=child_env,stdout=f,stderr=subprocess.STDOUT).returncode
  if rc==0:
   rec.update(status='complete',ended_utc=now(),returncode=0,outputs={str(p):sha(p) for p in output_files(cmd)})
   save(receipt,rec);return
  body=log.read_text(errors='replace')[-50000:]
  auth=any(term in body.lower() for term in ('authenticationerror','permissiondeniederror','invalid_api_key','incorrect api key','http error 401','http error 403','error code: 401','error code: 403','status_code=401','status_code=403'))
  rec.update(returncode=rc,last_failure_utc=now(),status='authentication_requires_user' if auth else 'pending_retry')
  if auth:
   save(receipt,rec);raise RuntimeError('Authentication requires user; no fallback generated; see strict stage receipt')
  delay=min(300,30*(2**min(attempt-1,4)))
  rec['retry_after_seconds']=delay;save(receipt,rec)
  print('Stage incomplete; successful caches preserved; retry in',delay,'seconds. Log:',str(log),flush=True)
  time.sleep(delay)
def install(pipe,state_dir,context):
 def run(cmd,*,env,logger):execute(cmd,env,state_dir,context)
 pipe._run=run
