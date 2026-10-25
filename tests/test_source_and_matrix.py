"""No credentials or network: source gates, extraction handoff, and matrix arithmetic."""
import csv
import importlib.util
import json
import tempfile
import unittest
import subprocess
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
P, E, R, M = [module(n) for n in ['prepare_extraction', 'export_phrases', 'run_extraction', 'matrix']]
def csvwrite(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)

class SourceTests(unittest.TestCase):
    def test_source_release_export_and_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); raw=d/'source.html'
            raw.write_text('<html><body><p>Acme Widgets Corporation</p>'
                '<p>Annual report for the fiscal year ended December 31, 2025</p>'
                '<ix:nonNumeric name="dei:EntityCentralIndexKey">12345</ix:nonNumeric>'
                '<ix:nonNumeric name="dei:DocumentFiscalYearFocus">2025</ix:nonNumeric>'
                '<ix:nonNumeric name="dei:DocumentPeriodEndDate">2025-12-31</ix:nonNumeric>'
                '<p>Item 1. Business</p>'+''.join('<p>We manufacture industrial pumps and distribute these products to commercial customers worldwide.</p>' for _ in range(60))+
                '<p>Item 1A. Risk Factors</p><p>Risks outside Item 1.</p></body></html>')
            row=dict(company_id='ACME',ticker='ACME',fiscal_year='2025',cik='12345',
                accession='0000012345-26-000001',form='10-K',report_period_end='2025-12-31',
                filing_date='2026-02-01',company_name_at_year='Acme Widgets Corporation',
                registrant_names='Acme Widgets Corporation',identity_basis='SYNTHETIC TEST ONLY',
                fiscal_year_basis='annual_cover_period_year',verification_scope='synthetic',
                included_in_has10k='1',panel_has10k='1',industry='Demo',
                local_raw_path='source.html',raw_sha256=P.sha(raw))
            bad=dict(row,company_id='WRONG',cik='98765',accession='0000012345-26-000002')
            mapping=d/'mapping.csv';csvwrite(mapping,[row,bad])
            run=d/'run'; summary=P.prepare(mapping,run,True)
            self.assertEqual((summary['approved_reports'],summary['held_reports']),(1,1))
            approved=R.verify(run,0)
            self.assertNotIn('Risks outside',Path(approved['item1_path']).read_text())
            # Synthetic stage receipt tests only the handoff, not extraction model quality.
            phrases=run/'outputs/mock_phrases.csv'
            csvwrite(phrases,[dict(ticker='ACME',year='2025',atom_mechanism_phrase='manufacture industrial pumps',original_sentence='We manufacture industrial pumps.')])
            contract=run/'inputs/releases/v1/contract.json'
            receipt=dict(status='complete',technical_qa_pass=True,enriched_output=str(phrases),
                enriched_sha256=P.sha(phrases),phrases=1,source_contract_sha256=P.sha(contract),item1_sha256=approved['item1_sha256'])
            P.dump(run/'state/ACME_0000012345-26-000001.json',receipt)
            out=d/'export';E.export(run,out,'Demo')
            coverage=M.read(out/'firm_years.csv')
            self.assertEqual({r['firm']:r['complete'] for r in coverage},{'ACME':'1','WRONG':'0'})
            self.assertEqual(M.read(out/'phrases.csv')[0]['final_phrase'],'manufacture industrial pumps')
            phrases.write_text('corrupt')
            with self.assertRaises(ValueError):E.export(run,d/'badexport','Demo')
            (run/'runtime/lib/utils.py').write_text('changed')
            with self.assertRaises(ValueError):R.verify(run,0)
            with self.assertRaises(FileExistsError):P.prepare(mapping,run,True)
            with self.assertRaises(ValueError):P.prepare(mapping,d/'unreviewed',False)
            raw.write_text('tamper')
            with self.assertRaises(ValueError):P.prepare(mapping,d/'tampered',True)

class MatrixTests(unittest.TestCase):
    def fixture(self,d):
        run=d/'run';inputs=run/'inputs';final=run/'outputs/pipeline/final';final.mkdir(parents=True)
        taxonomy=[dict(parent_id='P'+str(i),parent_label=label) for i,label in enumerate(['alpha','beta','gamma'])]
        csvwrite(inputs/'final_taxonomy_enrich.csv',taxonomy)
        source=[dict(ticker=f,year=str(y),final_phrase=l,original_sentence=l) for f,y,l in
                [('A',2024,'alpha'),('A',2025,'beta'),('B',2025,'beta'),('C',2025,'gamma'),('D',2027,'alpha')]]
        csvwrite(inputs/'source_original.csv',source)
        assigned=[dict(r,source_original_row=str(i+2),parent_ids_json=json.dumps(['P'+str(['alpha','beta','gamma'].index(r['final_phrase']))]),
            parent_labels_json=json.dumps([r['final_phrase']]),parent_count='1',assignment_status='assigned') for i,r in enumerate(source)]
        csvwrite(final/'all_assigned.csv',assigned)
        M.dump(final/'validation.json',dict(complete=True,unresolved=0,industry='demo',source_rows=len(source),
            taxonomy_sha256=M.sha(inputs/'final_taxonomy_enrich.csv'),source_original_sha256=M.sha(inputs/'source_original.csv'),
            output_hashes={'all_assigned.csv':M.sha(final/'all_assigned.csv')}))
        coverage=[dict(firm=r['ticker'],year=r['year'],complete='1',expected_source_rows='1') for r in source]
        coverage.extend([dict(firm='A',year='2026',complete='0',expected_source_rows='0'),dict(firm='ZERO',year='2025',complete='1',expected_source_rows='0')])
        csvwrite(d/'coverage.csv',coverage)
        csvwrite(d/'world.csv',[dict(firm='EXTERNAL',year='2020',parent_id='P1',parent_label='beta',source_id='synthetic:1')])
        M.dump(d/'world.json',dict(taxonomy_sha256=M.sha(inputs/'final_taxonomy_enrich.csv'),world_sha256=M.sha(d/'world.csv'),basis='confirmed_assignments',scope_description='Synthetic test corpus'))
        return run
    def test_counts_distances_missingness_and_workbook(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);run=self.fixture(d);out=d/'matrix'
            payload=M.prepare(run,d/'coverage.csv',d/'world.csv',d/'world.json',out)
            vectors=np.zeros((3,3072));vectors[:,:3]=np.eye(3)
            panel=M.calculate(payload,vectors);by={(r[0],r[1]):r for r in panel}
            self.assertEqual(by['A',2025][2:6],[1,1,1,0])
            self.assertEqual(by['A',2024][5],1)
            self.assertEqual(by['A',2026][2:],[None]*7)
            self.assertEqual(by['ZERO',2025][2],0)
            self.assertEqual(by['D',2027][3:5],[None,None])
            self.assertEqual(by['A',2025][6],'B')
            self.assertAlmostEqual(by['A',2025][7],0)
            self.assertAlmostEqual(by['A',2025][8],.75)
            self.assertEqual(by['C',2025][6],'A') # deterministic alphabetic tie
            np.save(out/'vectors.npy',vectors)
            M.dump(out/'vectors.json',dict(contract=M.CONTRACT,labels=payload['normalized_labels'],payload_sha256=M.sha(out/'payload.json'),vectors_sha256=M.sha(out/'vectors.npy')))
            path=M.build(out)
            from openpyxl import load_workbook
            w=load_workbook(path)
            self.assertEqual(w['A'].freeze_panes,'B2')
            self.assertIsNotNone(w['A']['D1'].comment)
            self.assertEqual(w['A']['D2'].value,0)
            self.assertEqual(w['A']['B2'].fill.fgColor.rgb[-6:],'C6EFCE')
            self.assertFalse(json.loads((out/'validation.json').read_text())['visual_render_validation'])
            with self.assertRaises(FileExistsError):M.build(out)
            (out/'vectors.npy').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):M.build(out)
    def test_final_and_world_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);run=self.fixture(d)
            subprocess.run([sys.executable,str(ROOT/'scripts/build_world.py'),'--run',str(run),
                '--output',str(d/'generated_world'),'--scope-description','Synthetic completed runs'],check=True,capture_output=True)
            self.assertEqual(len(M.read(d/'generated_world/world.csv')),5)
            meta=json.loads((d/'generated_world/world.json').read_text())
            self.assertEqual(meta['world_sha256'],M.sha(d/'generated_world/world.csv'))
            (d/'world.csv').write_text('tampered')
            with self.assertRaises(ValueError):M.prepare(run,d/'coverage.csv',d/'world.csv',d/'world.json',d/'out')
            v=run/'outputs/pipeline/final/validation.json';value=json.loads(v.read_text());value['complete']=False;M.dump(v,value)
            with self.assertRaises(ValueError):M.prepare(run,d/'coverage.csv',d/'world.csv',d/'world.json',d/'out')
    def test_zero_center_and_uncertain_embedding(self):
        payload={'normalized_labels':['same'],'records':[dict(firm='A',year=2025,complete=True,labels=['same'],panel=['A',2025,1,None,None,1])]}
        self.assertEqual(M.calculate(payload,np.ones((1,3072)))[0][6:],[None,None,None])
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);M.dump(d/'payload.json',payload)
            with self.assertRaises(ValueError):M.embed(d,0)
            import os
            old=os.environ.get('GEMINI_API_KEY');os.environ['GEMINI_API_KEY']='TEST_NOT_A_REAL_KEY'
            M.dump(d/'embedding_cache/old.intent',{'labels':['same']})
            try:
                with self.assertRaises(RuntimeError):M.embed(d,1)
            finally:
                if old is None:os.environ.pop('GEMINI_API_KEY',None)
                else:os.environ['GEMINI_API_KEY']=old

if __name__=='__main__':unittest.main()
