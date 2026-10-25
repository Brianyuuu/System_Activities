"""Verify release scope, exact source-body fidelity and immutable contracts; no API."""
import argparse,csv,hashlib,json
from pathlib import Path

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--published-root',type=Path);ap.add_argument('--source-snapshot',type=Path);ap.add_argument('--published-source-root',type=Path);ap.add_argument('--release',default='item1_v1');a=ap.parse_args()
 root=a.root.resolve();pub=a.published_root or root
 def local(p):return root/Path(p).relative_to(pub)
 base=root/'inputs/releases'/a.release;contract=json.loads((base/'contract.json').read_text());assert contract['source_scope']=='primary_item1_only'
 for p,h in contract['files'].items():assert sha(root/p)==h,p
 for p,h in json.loads((root/'state/runtime_snapshot.json').read_text()).items():assert sha(root/p)==h,p
 rows=list(csv.DictReader((base/'approved_manifest.csv').open(encoding='utf-8-sig')))
 evidence={x['accessionNumber']:x for x in json.loads((base/'source_evidence.json').read_text())}
 assert rows, 'No eligible reports: source preparation has not passed'
 assert len(rows)==contract['reports'] and len({(x['company_id'],x['panel_year']) for x in rows})==len(rows)
 for row in rows:
  assert row['source_scope']=='primary_item1_only'
  item=local(row['item1_path']);assert sha(item)==row['item1_sha256']
  text=item.read_text();meta=json.loads(text.split('metadata: ',1)[1].split('\n',1)[0]);body=text.split('\n# Item 1. Business\n\n',1)[1].removesuffix('\n')
  assert meta['source_scope']=='primary_item1_only' and meta['incorporated_references_included'] is False
  assert meta['source_part_spans']==[[0,len(body)]]
  parts=json.loads(local(row['parts_path']).read_text());assert len(parts)==1 and parts[0]['kind']=='primary_Item1'
  part=local(parts[0]['text_path']);assert sha(part)==parts[0]['text_sha256'] and part.read_text()==body
  if a.source_snapshot:
   e=evidence[row['accession_number']];original=a.source_snapshot/Path(e['item1_body_path']).relative_to(a.published_source_root)
   assert sha(original)==e['item1_body_sha256'] and original.read_text().strip()==body
 assert not any((root/'outputs/phrases').glob('**/latest_pipeline_summary.json'))
 result={'pass':True,'reports':len(rows),'source_scope':'primary_item1_only','one_primary_part_per_report':True,'exact_reviewed_body_match':bool(a.source_snapshot),'all_input_and_runtime_hashes_match':True,'mixed_scope_phrase_outputs_reused':0,'model_calls':0,'jobs_submitted':0}
 (root/'audits/item1_only_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
