import json
from pathlib import Path
import numpy as np
r=Path(__file__).resolve().parents[1];out=r/'outputs/pipeline'
s=json.loads((out/'filter_summary.json').read_text());d=np.load(out/'glm_filter_decisions.npy',mmap_mode='r')
assert s['status']=='complete' and s['pending']==0 and s['deferred_pairs']==0
assert s['completed_pairs']==s['candidate_pairs']==len(d)
assert np.isin(d,[0,1]).all()
assert (out/'glm_kept_candidates.csv').is_file()
print('GLM_COMPLETION_GATE_PASSED',len(d),flush=True)
