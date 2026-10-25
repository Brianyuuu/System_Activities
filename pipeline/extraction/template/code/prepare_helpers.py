"""Document evidence and Item 1 candidates; never turns exceptions into approvals."""
import csv, datetime, hashlib, json, re, sys, time
from pathlib import Path
from bs4 import BeautifulSoup
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'runtime/lib'))
from clean_data import _visible_text, _extract_item1, START_RE, START_ALT_RE, FORMAL_END_RE, _is_cross_reference

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def norm(s):return re.sub(r'[^a-z0-9]','',re.sub(r'\b(incorporated|inc|corp|corporation|company|co|the|limited|ltd)\b','',s.lower()))
def extract_candidate(visible):
    body,st,en=_extract_item1(visible,500);old_end=en
    starts=[m for pat in [START_RE,START_ALT_RE] for m in pat.finditer(visible) if m.start()==st]
    assert len(starts)>=1
    # A page number followed by a Properties subheading is not an Item 2 heading.
    explicit=[m for m in FORMAL_END_RE.finditer(visible) if m.start()>starts[0].end() and re.search(r'(?i)i\s*t\s*e\s*m',m.group()) and not _is_cross_reference(visible,m)]
    if explicit:
        en=explicit[0].start();body=visible[starts[0].end():en].strip()
    assert 500<=len(body.split())<=50000
    return body,st,en,old_end
def main():
    t=time.time();prov=json.loads((R/'inputs/panel_provenance.json').read_text())
    assert sha(R/'inputs/panel_scope.csv')==prov['scope_sha256']
    panel=list(csv.DictReader((R/'inputs/panel_scope.csv').open()));selected=json.loads((R/'inputs/pilot_selected.json').read_text())
    packets=[]
    for row in selected:
        a=row['accessionNumber'];raw=R/'raw'/f'{a}.html';assert sha(raw)==row['raw_sha256']
        soup=BeautifulSoup(raw.read_bytes(),'lxml');ix={}
        for el in soup.find_all(lambda el:bool(el.get('name')) and ':' in el.get('name','')):
            key=el.get('name','').split(':')[-1]
            if key in ['DocumentFiscalYearFocus','DocumentPeriodEndDate','EntityCentralIndexKey','EntityRegistrantName','DocumentType']:
                ix.setdefault(key,[]).append(el.get_text(' ',strip=True))
        ix={k:sorted(set(v)) for k,v in ix.items()}
        for el in soup.find_all(lambda el:el.name and (el.name.lower() in ['ix:hidden','ix:header','script','style','noscript'] or 'display:none' in el.get('style','').replace(' ','').lower())):
            if el.parent is not None:el.decompose()
        visible=_visible_text(str(soup).encode());vp=R/'evidence'/f'{a}_primary.txt';vp.parent.mkdir(exist_ok=True);vp.write_text(visible)
        cover=visible[:20000];matched=[x for x in panel if x['company_id']==row['company_id'] and x['year']==row['reportDate'][:4]]
        assert len(matched)==1
        pr=matched[0];assert row['cik'] in pr['cik_candidates_dated'].split(' | ') and pr['identity_review_required']=='0'
        aliases=[x.strip() for x in pr['company_name'].split(' | ') if x.strip()]
        aliases+=ix.get('EntityRegistrantName',[])+[x['name'] for x in row['former_names']]+[row['sec_current_name']]
        aliases=list(dict.fromkeys(aliases));hits=[x for x in aliases if len(norm(x))>=5 and norm(x) in norm(cover)]
        signals=[cover[max(0,m.start()-100):m.end()+220] for m in re.finditer(r'(?i)(?:fiscal\s+year|year|transition\s+period)\s+(?:ended|ending)',cover)]
        p={**row,'raw_path':str(raw),'visible_path':str(vp),'visible_sha256':sha(vp),'panel_row':pr,
           'xbrl':ix,'cover_name_matches':hits,'cover_period_signals':signals[:12],'cover_excerpt':cover[:6500],
           'source_approval':False,'preparation_status':'needs_review'}
        try:
            body,st,en,old_end=extract_candidate(visible);bp=R/'evidence'/f'{a}_item1.txt';bp.write_text(body)
            crossrefs=[body[max(0,m.start()-160):m.end()+220] for m in re.finditer(r'(?i)incorporat.{0,45}reference|business.{0,100}(?:pages?\s*\d|annual\s+report)|pages?\s*\d.{0,60}incorporat',body)]
            starts=[(m.start(),m.group()) for pattern in [START_RE,START_ALT_RE] for m in pattern.finditer(visible) if not _is_cross_reference(visible,m)]
            ends=[(m.start(),m.group()) for m in FORMAL_END_RE.finditer(visible) if not _is_cross_reference(visible,m)]
            p.update(item1_body_path=str(bp),item1_body_sha256=sha(bp),item1_words=len(body.split()),boundary_start=st,boundary_end=en,
                     old_extractor_boundary_end=old_end,boundary_corrected=old_end!=en,boundary_start_excerpt=visible[max(0,st-120):st+700],boundary_end_excerpt=visible[en-500:en+300],
                     cross_reference_signals=crossrefs,item1_heading_candidates=starts,following_headings=[x for x in ends if x[0]>=st][:8],preparation_status='candidate_extracted')
        except Exception as e:p.update(preparation_error=type(e).__name__+': '+str(e))
        packets.append(p);print('PREPARED',a,p['preparation_status'],p.get('item1_words'),flush=True)
    save(R/'audits/source_review_packets.json',packets)
    save(R/'audits/preparation_summary.json',{'reports':len(packets),'item1_candidates':sum(x['preparation_status']=='candidate_extracted' for x in packets),'seconds':time.time()-t,'model_calls':0,'all_source_approved':False})
if __name__=='__main__':main()
