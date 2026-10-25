"""Read-only validation and per-industry progress; never invokes models."""
from pathlib import Path
import csv,json,hashlib,collections,statistics,datetime
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False));t.replace(p)
def write(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp')
 with t.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for x in rows for k in x)));w.writeheader();w.writerows(rows)
 t.replace(p)
def main():
 coverage=list(csv.DictReader((R/'outputs/coverage_inventory.csv').open(encoding='utf-8-sig')))
 rows=list(csv.DictReader((R/'inputs/releases/banks_hardware_v1/approved_manifest.csv').open(encoding='utf-8-sig')));man={(x['company_id'],x['panel_year']):x for x in rows};metrics=[];phrases=[]
 for c in coverage:
  m=man.get((c['company_id'],c['panel_year']))
  if not m:continue
  c['status']='approved_pending_execution';c['phrase_rows']=0
  p=R/'state'/f'{m["company_id"]}_{m["accession_number"]}.json'
  if not p.exists():continue
  d=json.loads(p.read_text());c.update(status=d['status'],job_id=d.get('job_id',''))
  if d['status']!='complete':c['reason']=d.get('error','');continue
  assert d['item1_sha256']==m['item1_sha256'] and sha(d['enriched_output'])==d['enriched_sha256']
  pp=list(csv.DictReader(Path(d['enriched_output']).open(encoding='utf-8-sig')));assert len(pp)==d['phrases']
  for x in pp:assert x['source_scope']=='primary_item1_only';x['industry']=m['industry']
  phrases+=pp;c.update(phrase_rows=len(pp),technical_qa_pass=True,semantic_release_pass=False)
  metrics.append({**{k:d.get(k,'') for k in ['company_id','panel_year','accession','elapsed_seconds','started_utc','ended_utc','phrases','stage_retries']},'industry':m['industry'],'input_words':int(m['item1_words'])})
 counts=dict(collections.Counter(x['status'] for x in coverage));out=R/'outputs/progress'
 write(out/'report_coverage.csv',coverage)
 if metrics:write(out/'report_metrics.csv',metrics)
 if phrases:write(out/'phrases_completed.csv',phrases)
 summary={'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'target_company_years':len(coverage),'approved_new_reports':len(rows),'status_counts':counts,'new_completed_reports':len(metrics),'new_phrase_rows':len(phrases),'source_scope_new':'primary_item1_only','existing_phrase_outputs_preserved_not_recertified':True,'semantic_release_pass':False,'downstream_parent_GLM_screening_started':False,'per_industry':{i:{'target':sum(x['industry']==i for x in coverage),'statuses':dict(collections.Counter(x['status'] for x in coverage if x['industry']==i)),'new_phrases':sum(int(x.get('phrase_rows',0)) for x in coverage if x['industry']==i)} for i in sorted({x['industry'] for x in coverage})}}
 if metrics:summary['completed_report_timing']={'n':len(metrics),'mean_seconds':statistics.mean(x['elapsed_seconds'] for x in metrics),'median_seconds':statistics.median(x['elapsed_seconds'] for x in metrics),'completed_sample_only':True}
 save(out/'summary.json',summary);print(json.dumps(summary))
if __name__=='__main__':main()
