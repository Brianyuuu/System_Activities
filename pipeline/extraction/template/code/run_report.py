"""Execute one immutable release row with cached strict stages and rich provenance export."""
import csv,datetime,fcntl,hashlib,json,logging,os,re,sys,time,unicodedata
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'runtime/lib'),str(R/'code')]
import latest_phrase_pipeline as pipe
from strict_stage_runtime import execute
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def norm(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s)).strip()
def writecsv(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
 if not rows:tmp.write_text('');tmp.replace(p);return
 with tmp.open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 tmp.replace(p)
def main():
 start=time.time();release=os.environ['CASINO_RELEASE'];ix=int(os.environ['SLURM_ARRAY_TASK_ID']);base=R/'inputs/releases'/release
 contract=json.loads((base/'contract.json').read_text())
 assert contract['source_scope']=='primary_item1_only'
 for f,h in contract['files'].items():assert sha(R/f)==h,f
 for f,h in json.loads((R/'state/runtime_snapshot.json').read_text()).items():assert sha(R/f)==h,f
 row=list(csv.DictReader((base/'approved_manifest.csv').open(encoding='utf-8-sig')))[ix];key=row['company_id']+'_'+row['accession_number']
 assert row['source_scope']=='primary_item1_only'
 approved_parts=json.loads(Path(row['parts_path']).read_text())
 assert len(approved_parts)==1 and approved_parts[0]['kind']=='primary_Item1'
 lock=(R/'state'/f'{key}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 receipt=R/'state'/f'{key}.json'
 if receipt.exists():
  prior=json.loads(receipt.read_text())
  assert prior['item1_sha256']==row['item1_sha256']
  if prior['status']=='complete':assert sha(prior['enriched_output'])==prior['enriched_sha256'];print('VERIFIED_REUSED',key);return
  raise RuntimeError('Existing failed/running receipt needs explicit checkpoint review')
 for k in ['raw','item1']:assert sha(row[k+'_path'])==row[k+'_sha256']
 rec={'source_scope':'primary_item1_only','release':release,'index':ix,'company_id':row['company_id'],'panel_year':row['panel_year'],'fiscal_year':row['year'],'accession':row['accession_number'],'item1_sha256':row['item1_sha256'],'source_contract_sha256':sha(base/'contract.json'),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':os.environ['SLURM_JOB_ID'],'status':'running'};save(receipt,rec)
 settings=json.loads((R/'config.json').read_text())['latest_phrase_pipeline']
 os.environ.update(PHRASE_API_WORKERS='8',PHRASE_HTTP_EVENTS=str(R/'state'/f'{key}_http.jsonl'),GLM_USAGE_EVENTS=str(R/'state'/f'{key}_usage.jsonl'))
 env=pipe._api_environment(settings,R);env['PYTHONPATH']=str(R/'code/observer')+os.pathsep+env.get('PYTHONPATH','')
 output=Path(settings['output_root'])/row['sample_id']/row['year']/row['accession_number']
 context={k:row[k] for k in ['company_id','sample_id','accession_number','target_fiscal_year','company_name_wrds','raw_sha256','item1_sha256']};context['contract_sha256']=rec['source_contract_sha256']
 def run(cmd,*,env,logger):
  cmd=list(cmd)
  for j,k in enumerate(cmd[:-1]):
   if k in ['--api-workers','--workers','--submission-batch-size']:cmd[j+1]='8'
  execute(cmd,env,output/'_strict_stage_state',context)
 pipe._run=run
 try:
  summary=pipe._run_filing(row,settings,R,env,logging.getLogger('casino-full'))
  assert summary['status']=='pass' and summary['extraction_model']==summary['audit_model']=='parcc/GLM'
  phrases=list(csv.DictReader(Path(summary['final_output']).open(encoding='utf-8-sig')))
  rich=list(csv.DictReader((output/'12_specificity_audit/specificity_kept.csv').open(encoding='utf-8-sig'))) if phrases else []
  assert len(rich)==len(phrases)==summary['final_rows']
  parts=json.loads(Path(row['parts_path']).read_text());texts=[]
  for p in parts:
   assert sha(p['text_path'])==p['text_sha256'];texts.append(norm(Path(p['text_path']).read_text()))
  canonical=norm(Path(row['item1_path']).read_text().split('\n# Item 1. Business\n\n',1)[1]);enriched=[]
  for a,b in zip(phrases,rich):
   assert a['original_sentence']==b['source_sentence'];sentence=norm(a['original_sentence']);matches=[i for i,s in enumerate(texts) if sentence in s]
   assert sentence and sentence in canonical and matches,'source_sentence_part_provenance'
   flags=[];phrase=a['atom_mechanism_phrase']
   if re.fullmatch(r'(?i)(?:own|operate|offer|provide|open|manage|sell)\s+(?:the\s+)?\[[^]]+\](?:\s+in\s+\[[^]]+\])*(?:\s*,\s*\[[^]]+\])?',phrase.strip()):flags.append('generic_after_entity_masking')
   if re.search(r'(?i)\bexpired|terminated|formerly|previously\b',sentence):flags.append('historical_or_expired_source')
   if re.match(r'(?i)^(?:and/or|and |including |as well as|which |that )',sentence):flags.append('clause_requires_prior_context')
   if re.search(r'(?i)table of contents|EX-99',sentence):flags.append('source_header_noise')
   enriched.append({**a,'source_scope':'primary_item1_only','company_id':row['company_id'],'panel_year':row['panel_year'],'fiscal_year':row['year'],'reporting_name':row['historical_company_name'],'cik':row['cik'],'accession':row['accession_number'],'report_date':row['report_date'],'actual_fiscal_period_end':row['actual_fiscal_period_end'],'prior_context':b.get('prior_context',''),'context_plus_clause':(b.get('prior_context','')+' '+a['original_sentence']).strip(),'pre_entity_atom_phrase':b.get('atom_mechanism_phrase',''),'source_part_indices':'|'.join(map(str,matches)),'source_urls':' | '.join(parts[i]['source_url'] for i in matches),'semantic_review_flags':'|'.join(flags),'semantic_release_status':'not_individually_semantically_certified','source_sha256':row['item1_sha256']})
  states=[json.loads(p.read_text()) for p in (output/'_strict_stage_state').glob('*.json')]
  assert all(x['status']=='complete' and not x['failure_fallback_enabled'] for x in states)
  assert len(states)==13 or (not phrases and summary.get('reason') in ['no_focal_candidates','all_candidates_rejected','all_atomic_phrases_rejected'])
  diagnostics=json.loads((output/'01_extraction/chunk_diagnostics.json').read_text())
  extraction_manifest=json.loads((output/'01_extraction/extraction_manifest.json').read_text())
  assert extraction_manifest['unresolved_jobs']==0 and extraction_manifest['transport_fallback_rows']==0
  first=next(x for x in states if x['stage'].startswith('01b_'))
  assert str(output/'01_extraction/extraction_manifest.json') in first['outputs']
  for st in states:
   assert st['returncode']==0
   for file,expected in st['outputs'].items():assert sha(file)==expected,file
  rec['recovered_historical_chunk_errors']=sum(bool(x.get('error')) for x in diagnostics)
  dest=output/'final/enriched_phrases.csv';writecsv(dest,enriched)
  rec.update(status='complete',technical_qa_pass=True,semantic_release_pass=False,phrases=len(phrases),flagged_rows=sum(bool(x['semantic_review_flags']) for x in enriched),enriched_output=str(dest),enriched_sha256=sha(dest),stage_retries=sum(x['attempts']-1 for x in states),ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.time()-start)
  save(receipt,rec);print(json.dumps(rec),flush=True)
 except BaseException as e:
  rec.update(status='failed_preserve_checkpoints',error_type=type(e).__name__,error=str(e),elapsed_seconds=time.time()-start);save(receipt,rec);raise
if __name__=='__main__':main()
