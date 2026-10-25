"""Deterministic source evidence packets. Ambiguities are held, never silently approved."""
import argparse,csv,datetime,hashlib,json,re,sys,time
from pathlib import Path
from bs4 import BeautifulSoup
PROJECT=Path(__file__).resolve().parents[1]
R=PROJECT/'source'
sys.path[:0]=[str(PROJECT/'runtime/lib'),str(PROJECT/'code')]
from clean_data import _visible_text,START_RE,START_ALT_RE,FORMAL_END_RE
from prepare_helpers import extract_candidate
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def norm(s):
 s=re.sub(r'\b(incorporated|inc|corporation|corp|company|co|the|limited|ltd|plc)\b','',s.lower())
 return re.sub(r'[^a-z0-9]','',s)
MONTHS={m.lower():i for i,m in enumerate(['January','February','March','April','May','June','July','August','September','October','November','December'],1)}
def parse_date_value(value):
 try:return datetime.date.fromisoformat(value).isoformat()
 except ValueError:
  evidence=cover_dates('year ended '+value)
  if evidence:return evidence[0]['date']
  for fmt in ['%m/%d/%Y','%m-%d-%Y']:
   try:return datetime.datetime.strptime(value.strip(),fmt).date().isoformat()
   except ValueError:pass
  raise ValueError('Unrecognized XBRL date display: '+value)
def cover_dates(text):
 results=[];flat=re.sub(r'\s+',' ',text)
 for m in re.finditer(r'(?i)(?:fiscal\s+year|year|transition\s+period)\s+(?:ended|ending)\s*[:\-]?\s*(.{0,100})',flat):
  z=re.search(r'(?i)('+'|'.join(MONTHS)+r')\s*(\d{1,2})\s*,?\s*((?:19|20)\d{2})',m[1])
  if z:
   try:results.append({'date':datetime.date(int(z[3]),MONTHS[z[1].lower()],int(z[2])).isoformat(),'excerpt':m[0]})
   except ValueError:pass
 return results
def prepare(f,panel,links,security,supp):
 a=f['accessionNumber'];co=f['company_id'];raw=R/'raw'/f['raw_filename'];assert sha(raw)==f['raw_sha256']
 blob=raw.read_bytes();txt=blob.decode('utf-8',errors='replace')
 # Legacy complete submissions contain multiple DOCUMENTs; isolate the original annual report.
 if re.search(r'(?i)<DOCUMENT>',txt):
  docs=re.findall(r'(?is)<DOCUMENT>(.*?)</DOCUMENT>',txt)
  forms=[f['form']]+(['10-K'] if f['form']=='10-K405' else [])
  annual=[x for x in docs if any(re.search(r'(?im)^\s*<TYPE>\s*'+re.escape(form)+r'\s*$',x) for form in forms)]
  assert len(annual)==1,'legacy_primary_document_not_unique';txt=annual[0]
 soup=BeautifulSoup(txt,'lxml');ix={}
 for el in soup.find_all(lambda el:bool(el.get('name')) and ':' in el.get('name','')):
  k=el.get('name').split(':')[-1]
  if k in ['DocumentFiscalYearFocus','DocumentPeriodEndDate','EntityCentralIndexKey','EntityRegistrantName','DocumentType']:ix.setdefault(k,[]).append(el.get_text(' ',strip=True))
 ix={k:sorted(set(v)) for k,v in ix.items()}
 for el in soup.find_all(lambda el:el.name and (el.name.lower() in ['ix:hidden','ix:header','script','style','noscript'] or 'display:none' in el.get('style','').replace(' ','').lower())):
  if el.parent is not None:el.decompose()
 visible=_visible_text(str(soup).encode());vp=R/'evidence'/f'{a}_primary.txt';vp.parent.mkdir(exist_ok=True);vp.write_text(visible)
 cover=visible[:14000];dts=cover_dates(cover[:9000]);secdate=datetime.date.fromisoformat(f['reportDate'])
 near=[x for x in dts if abs((datetime.date.fromisoformat(x['date'])-secdate).days)<=7]
 actual=parse_date_value(ix['DocumentPeriodEndDate'][0]) if len(ix.get('DocumentPeriodEndDate',[]))==1 else None
 issues=[]
 if not actual and near:actual=near[0]['date']
 if not actual:issues.append('cover_period_unverified');actual=f['reportDate']
 if actual and abs((datetime.date.fromisoformat(actual)-secdate).days)>7:issues.append('SEC_XBRL_period_conflict')
 if ix.get('EntityCentralIndexKey') and {int(x) for x in ix['EntityCentralIndexKey']}!={int(f['cik'])}:issues.append('XBRL_CIK_mismatch')
 if ix.get('DocumentType') and ix['DocumentType']!=[f['form']]:issues.append('XBRL_form_mismatch')
 py=int(f['reportDate'][:4]);pr=[x for x in panel if x['company_id']==co and int(x['year'])==py]
 assert len(pr)==1;pr=pr[0]
 dated=[x for x in links if x['company_id']==co and int(x['cik'])==int(f['cik']) and x['valid_start']<=actual<=x['valid_end']]
 supplement=[x for x in supp if x['company_id']==co and int(x['cik'])==int(f['cik']) and x['valid_start']<=actual<=x['valid_end']]
 secnames=[x for x in security if x['company_id']==co and x['namedt']<=actual<=x['nameenddt']]
 aliases=list(dict.fromkeys([x['issuer_name'] for x in dated]+[x['issuernm'] for x in secnames]+[x.strip() for x in pr['company_name'].split(' | ') if x.strip()]))
 hits=[x for x in aliases if len(norm(x))>=5 and norm(x) in norm(cover[:9000])]
 if not dated and not supplement:issues.append('no_exact_date_CIK_certificate')
 if not hits:issues.append('historical_cover_name_unverified')
 if pr['identity_review_required']=='1':issues.append('panel_identity_review_pending')
 reporting=next((x for x in ix.get('EntityRegistrantName',[]) if any(norm(x)==norm(h) for h in hits)),hits[0] if hits else '')
 fy=ix.get('DocumentFiscalYearFocus',[])
 if len(fy)==1 and re.fullmatch(r'(19|20)\d{2}',fy[0]):fiscal=int(fy[0]);fy_basis='XBRL_DocumentFiscalYearFocus'
 else:
  fiscal=int(actual[:4]);fy_basis='explicit_cover_period_end_year'
  if not near:issues.append('cover_fiscal_label_unverified')
  if int(actual[5:7])<=2:issues.append('early_calendar_year_fiscal_label_review')
 if fiscal!=py:issues.append('panel_year_vs_fiscal_year_review')
 if pr['has10k_dummy']!='1':issues.append('outside_positive_panel_scope')
 out={**f,'panel_year':py,'fiscal_year':fiscal,'fiscal_year_basis':fy_basis,'actual_period_end':actual,'xbrl':ix,'cover_period_evidence':near,'historical_name_matches':hits,'reporting_name':reporting,'cover_excerpt':cover[:9000],'identity_evidence':dated+supplement,'security_name_evidence':secnames,'source_issues':issues,'primary_text_path':str(vp),'primary_text_sha256':sha(vp)}
 bounds_path=R/'audits/manual_item1_bounds.json'
 bounds=json.loads(bounds_path.read_text()) if bounds_path.exists() else {}
 if a in bounds:
  b=bounds[a];assert b['raw_sha256']==f['raw_sha256']
  st=b['start'];en=b['end'];old_end=en
  assert visible[st:].startswith(b['start_heading']) and visible[en:].startswith(b['end_heading'])
  body=visible[st+len(b['start_heading']):en].strip();assert 500<=len(body.split())<=50000
  out['manual_boundary_evidence']=b
 else:body,st,en,old_end=extract_candidate(visible)
 bp=R/'evidence'/f'{a}_item1.txt';bp.write_text(body)
 refs=[];cross=[]  # Item 1-only: deliberately do not inspect or follow references.
 out.update(item1_body_path=str(bp),item1_body_sha256=sha(bp),item1_words=len(body.split()),boundary_start=st,boundary_end=en,boundary_corrected=en!=old_end,boundary_start_excerpt=visible[max(0,st-150):st+500],boundary_end_excerpt=visible[en-400:en+350],exhibit_references=refs,cross_reference_contexts=cross,source_status='evidence_ready' if not issues else 'held')
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--accessions',nargs='*');args=ap.parse_args()
 start=time.time();(R/'audits/source_packets').mkdir(parents=True,exist_ok=True);panel=list(csv.DictReader((R/'inputs/panel_scope.csv').open(encoding='utf-8-sig')));links=list(csv.DictReader((R/'inputs/dated_cik_cusip_evidence.csv').open(encoding='utf-8-sig')));security=list(csv.DictReader((R/'inputs/company_security_history.csv').open(encoding='utf-8-sig')));supp=json.loads((R/'inputs/cik_supplement_candidates.json').read_text());packets=[]
 candidates=json.loads((R/'inputs/all_filing_candidates.json').read_text())
 for i,f in enumerate(candidates):
  if args.accessions and f['accessionNumber'] not in args.accessions:continue
  if f.get('download_status')=='previously_extracted':
   save(R/'audits/source_packets'/f'{f["company_id"]}_{f["accessionNumber"]}.json',{**f,'source_status':'previously_extracted','source_issues':[]});continue
  if f.get('download_status')!='downloaded':continue
  dest=R/'audits/source_packets'/f'{f["company_id"]}_{f["accessionNumber"]}.json'
  try:p=prepare(f,panel,links,security,supp)
  except Exception as e:p={**f,'source_status':'held','source_issues':['preparation_error'],'error':str(e)}
  save(dest,p);packets.append(p)
  if i%25==0:print('PREPARED',i,p['source_status'],flush=True)
 save(R/'audits/source_packets.json',[json.loads(x.read_text()) for x in (R/'audits/source_packets').glob('*.json')]);save(R/'audits/preparation_summary.json',{'reports':len(packets),'seconds':time.time()-start,'model_calls':0,'reference_checks':0});print('SOURCE_PREP_COMPLETE',len(packets),round(time.time()-start,2),flush=True)
if __name__=='__main__':main()
