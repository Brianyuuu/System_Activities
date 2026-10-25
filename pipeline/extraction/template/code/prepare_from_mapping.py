"""Explicit accession mapping -> original document -> verified primary Item 1. No model calls."""
from pathlib import Path
import csv,json,hashlib,shutil,urllib.request,urllib.error,time,re,datetime,sys,collections,os
from bs4 import BeautifulSoup
R=Path(__file__).resolve().parents[1];S=R/'source';OLD=R/'unused_legacy_cache'
sys.path[:0]=[str(R/'runtime/lib'),str(R/'code')]
from clean_data import _visible_text
from prepare_helpers import extract_candidate
from prepare_sources import cover_dates,parse_date_value,norm
HEADERS={'User-Agent':os.environ.get('SEC_USER_AGENT',''),'Accept-Encoding':'identity'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False));t.replace(p)
def readcsv(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def acquire(m,cache):
 a=m['accession'];p=S/'raw'/f'{a}.html'
 if m.get('local_raw_path'):
  src=Path(m['local_raw_path']);assert m.get('raw_sha256') and sha(src)==m['raw_sha256'],'local raw hash mismatch'
  shutil.copyfile(src,p);return p,m.get('sec_primary_document_url',''),'verified_local_raw'
 if not HEADERS['User-Agent']:raise ValueError('SEC_USER_AGENT is required for downloads')
 if a in cache:
  f=cache[a];old=OLD/'source/raw'/f['raw_filename'];assert int(f['cik'])==int(m['cik']) and sha(old)==f['raw_sha256']
  shutil.copyfile(old,p);return p,f['source_url'],'verified_existing_raw_cache'
 errors=[]
 for url in dict.fromkeys([m['sec_primary_document_url'],m['sec_full_submission_url']]):
  if not url:continue
  assert url.startswith('https://www.sec.gov/Archives/edgar/data/') and (a in url or a.replace('-','') in url)
  for attempt in range(3):
   try:
    with urllib.request.urlopen(urllib.request.Request(url,headers=HEADERS),timeout=45) as h:b=h.read()
    assert b and b'Your Request Originates from an Undeclared Automated Tool' not in b
    p.write_bytes(b);time.sleep(.35);return p,url,'explicit_mapping_SEC_download'
   except Exception as e:
    errors.append(str(e))
    if isinstance(e,urllib.error.HTTPError) and e.code==404:break
    time.sleep(2**attempt)
 raise RuntimeError('Mapped URLs failed: '+' | '.join(errors))
def prepare(m,p,url,method):
 a=m['accession'];issues=[];raw=p.read_bytes();txt=raw.decode('utf-8',errors='replace');header=txt.split('<DOCUMENT>',1)[0] if '<DOCUMENT>' in txt else ''
 if re.search(r'(?i)<DOCUMENT>',txt):
  docs=re.findall(r'(?is)<DOCUMENT>(.*?)</DOCUMENT>',txt);forms=[m['form']]+(['10-K'] if m['form']=='10-K405' else [])
  annual=[x for x in docs if any(re.search(r'(?im)^\s*<TYPE>\s*'+re.escape(f)+r'\s*$',x) for f in forms)]
  assert len(annual)==1,'legacy_primary_document_not_unique';txt=annual[0]
 soup=BeautifulSoup(txt,'lxml');ix={}
 for el in soup.find_all(lambda e:bool(e.get('name')) and ':' in e.get('name','')):
  k=el.get('name').split(':')[-1]
  if k in ['DocumentFiscalYearFocus','DocumentPeriodEndDate','EntityCentralIndexKey','EntityRegistrantName','DocumentType']:ix.setdefault(k,[]).append(el.get_text(' ',strip=True))
 ix={k:sorted(set(v)) for k,v in ix.items()}
 for el in soup.find_all(lambda e:e.name and (e.name.lower() in ['ix:hidden','ix:header','script','style','noscript'] or 'display:none' in e.get('style','').replace(' ','').lower())):
  if el.parent is not None:el.decompose()
 visible=_visible_text(str(soup).encode());vp=S/'evidence'/f'{a}_primary.txt';vp.write_text(visible);cover=visible[:9000]
 expected=datetime.date.fromisoformat(m['report_period_end']);dates=cover_dates(cover);near=[x for x in dates if abs((datetime.date.fromisoformat(x['date'])-expected).days)<=7]
 if len(ix.get('DocumentPeriodEndDate',[]))==1:
  actual=parse_date_value(ix['DocumentPeriodEndDate'][0])
  if abs((datetime.date.fromisoformat(actual)-expected).days)>7:issues.append('mapping_XBRL_period_conflict')
 elif near:actual=near[0]['date']
 else:
  actual=m['report_period_end'];issues.append('cover_period_unverified')
 if ix.get('EntityCentralIndexKey') and {int(x) for x in ix['EntityCentralIndexKey']}!={int(m['cik'])}:issues.append('mapping_XBRL_CIK_conflict')
 header_ciks=re.findall(r'(?im)CENTRAL INDEX KEY:\s*(\d+)',header)
 if header_ciks and int(m['cik']) not in {int(x) for x in header_ciks}:issues.append('mapping_header_CIK_conflict')
 if ix.get('DocumentType') and ix['DocumentType'] not in [[m['form']],['10-K']]:issues.append('mapping_form_conflict')
 aliases=list(dict.fromkeys([m['company_name_at_year']]+[x.strip() for x in m['registrant_names'].split(' | ') if x.strip()]))
 hits=[x for x in aliases if len(norm(x))>=5 and norm(x) in norm(cover)]
 if not hits:issues.append('historical_cover_name_unverified')
 if not m['identity_basis'] or m['included_in_has10k']!='1' or m['panel_has10k']!='1':issues.append('mapping_identity_or_scope_unverified')
 fy=ix.get('DocumentFiscalYearFocus',[])
 if fy:
  if fy!=[m['fiscal_year']]:issues.append('mapping_XBRL_fiscal_year_conflict')
 elif m['fiscal_year_basis'] in ['annual_cover_period_year','SEC_annual_header_period_year']:
  if actual[:4]!=m['fiscal_year'] or not near:issues.append('mapping_cover_fiscal_year_unverified')
 else:
  # Mapping may certify an explicit prose label; retain a visible-label check.
  if not re.search(r'(?i)(?:fiscal\s+year|year)\s*(?:ended\s*)?'+re.escape(m['fiscal_year']),cover):issues.append('mapping_fiscal_label_requires_review')
 body,st,en,old_end=extract_candidate(visible);bp=S/'evidence'/f'{a}_item1.txt';bp.write_text(body)
 return {'company_id':m['company_id'],'canonical_ticker':m['ticker'],'accessionNumber':a,'cik':m['cik'],'form':m['form'],'reportDate':m['report_period_end'],'filingDate':m['filing_date'],'raw_filename':p.name,'raw_sha256':sha(p),'source_url':url,'source_method':method,'panel_year':int(m['fiscal_year']),'fiscal_year':int(m['fiscal_year']),'fiscal_year_basis':m['fiscal_year_basis'],'actual_period_end':actual,'xbrl':ix,'cover_period_evidence':near,'historical_name_matches':hits,'reporting_name':ix.get('EntityRegistrantName',[hits[0] if hits else m['registrant_names']])[0],'cover_excerpt':cover,'identity_evidence':[{'method':'user_provided_accession_mapping','identity_basis':m['identity_basis'],'company_id':m['company_id'],'cik':m['cik'],'accession':a,'verification_scope':m['verification_scope']}],'mapping_row':m,'primary_text_path':str(vp),'primary_text_sha256':sha(vp),'item1_body_path':str(bp),'item1_body_sha256':sha(bp),'item1_words':len(body.split()),'boundary_start':st,'boundary_end':en,'boundary_corrected':en!=old_end,'boundary_start_excerpt':visible[max(0,st-120):st+500],'boundary_end_excerpt':visible[en-350:en+350],'source_issues':issues,'source_status':'held' if issues else 'evidence_ready'}
def main():
 started=time.time();prov=json.loads((S/'inputs/panel_provenance.json').read_text());assert sha(S/'inputs/panel_scope.csv')==prov['scope_sha256'] and sha(S/'inputs/mapping_scope.csv')==prov['mapping_scope_sha256']
 panel={(x['company_id'],x['year']):x for x in readcsv(S/'inputs/panel_scope.csv') if x['has10k_dummy']=='1'}
 mappings=readcsv(S/'inputs/mapping_scope.csv');groups=collections.defaultdict(list)
 for m in mappings:
  assert m['included_in_has10k']=='1' and (m['company_id'],m['fiscal_year']) in panel;groups[(m['company_id'],m['fiscal_year'])].append(m)
 assert set(groups)==set(panel)
 claimed={};claims_by_key={}
 existing=json.loads((R/'inputs/extracted_accessions.json').read_text());cache={}
 packets=[];selection=[];errors=[];downloads=collections.Counter()
 for i,(key,ms) in enumerate(sorted(groups.items())):
  if len(ms)>1:
   packet={'company_id':key[0],'panel_year':int(key[1]),'source_status':'held','source_issues':['multiple_mapping_rows_require_review'],'mapping_candidates':ms}
   save(S/'audits/source_packets'/f'{key[0]}_{key[1]}_ambiguous.json',packet);packets.append(packet);continue
  m=ms[0]
  a=m['accession'];packet={'company_id':m['company_id'],'canonical_ticker':m['ticker'],'accessionNumber':a,'reportDate':m['report_period_end'],'panel_year':int(m['fiscal_year']),'mapping_row':m,'source_issues':[]}
  if a in claimed:
   c=claimed[a];assert (c['company_id'],c['panel_year'])==key;packet.update(source_status='existing_run_claimed',existing_run=str(OLD))
  elif a in existing:packet.update(source_status='previously_extracted',existing_extraction=existing[a])
  else:
   try:
    p,url,method=acquire(m,cache);downloads[method]+=1;packet=prepare(m,p,url,method)
   except Exception as e:
    packet.update(source_status='held',source_issues=['source_preparation_error'],error=str(e));errors.append({'company_id':key[0],'fiscal_year':key[1],'accession':a,'error':str(e)})
  save(S/'audits/source_packets'/f'{m["company_id"]}_{a}.json',packet);packets.append(packet)
  if i%20==0:print('MAPPED_PROGRESS',i,len(groups),dict(collections.Counter(x['source_status'] for x in packets)),flush=True)
 save(S/'audits/mapping_selection_decisions.json',selection);save(S/'audits/preparation_errors.json',errors)
 summary={'target_company_years':len(panel),'mapping_rows':len(mappings),'status_counts':dict(collections.Counter(x['source_status'] for x in packets)),'source_acquisition':dict(downloads),'errors':len(errors),'elapsed_seconds':time.time()-started,'model_calls':0,'identity_policy':'Explicit user mapping certificate plus document CIK/name/period and primary Item1 checks'}
 save(S/'audits/preparation_summary.json',summary);print(json.dumps(summary))
if __name__=='__main__':main()
