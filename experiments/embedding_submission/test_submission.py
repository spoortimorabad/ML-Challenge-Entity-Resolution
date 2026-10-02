import json,tempfile,shutil
from pathlib import Path
import numpy as np,pandas as pd
import embedding_core as c
import submit_pipeline as s

class FakeEncoder:
    def __init__(self,vec,fail_at=None):self.vec=vec;self.calls=0;self.fail_at=fail_at
    def encode(self,frame):
        self.calls+=1
        if self.calls==self.fail_at:raise RuntimeError('interrupted')
        return self.vec[[int(x.split('-')[1])%len(self.vec) for x in frame.entity_id]]
class MockMatcher:
    def __init__(self):self.pairs=0
    def predict_proba(self,x):
        self.pairs+=len(x)
        prob=np.where(x[:,0]>.5,.9,.2)
        return np.c_[1-prob,prob]

with tempfile.TemporaryDirectory() as td:
 root=Path(td);data=root/'dataset';(data/'test').mkdir(parents=True);work=root/'work';work.mkdir()
 rng=np.random.default_rng(8);v=rng.normal(size=(60,c.DIM)).astype('float32');v/=np.linalg.norm(v,axis=1,keepdims=True)
 for source in [2,3]:
  base=(source-2)*30
  rows=[(f'S{source}-{base+i}',f'Name {base+i}','10 Main Road',['US','India','France'][(base+i)//20]) for i in range(30)]
  pd.DataFrame(rows,columns=c.COLS).to_csv(data/f'test/test_source{source}.tsv',sep='\t',index=False)
 c.encode_pool(data,work,FakeEncoder(v),split='test');c.build_indexes(work)
 # Ensure the optimized batch rerank matches the original reference implementation.
 frame=pd.DataFrame([(f'S1-{i}',f'Name {i}','10 Main Rd',['US','India','France'][i//20]) for i in range(60)],columns=c.COLS)
 q=v.copy();search=s.TestSearcher(work)
 ids,score=search.search(frame,q)
 old_ids,old_score=c.retrieve(work,frame,q,save_k=20)
 assert np.array_equal(ids,old_ids)
 assert np.allclose(score,old_score,atol=2e-6)
 rec=search.fetch(ids);search.close()
 model=MockMatcher();bundle={'model':model,'threshold':.7}
 cand,accept=s.score_candidates(frame,ids,score,rec,bundle)
 assert model.pairs==sum(map(len,cand))==1200
 assert all(set(m)<=set(cs) for m,cs in zip(accept,cand))
 # Add an entity whose country has no available records: both outputs must be empty.
 frame.loc[len(frame)]=['S1-60','Unmatched','Elsewhere','ZZ']
 frame.to_csv(data/'test/test_source1.tsv',sep='\t',index=False)
 s.PREDICT_CHUNK=20
 try:s.predict(data,work,bundle,FakeEncoder(v,fail_at=2))
 except RuntimeError as e:assert str(e)=='interrupted'
 else:raise AssertionError('Interruption not exercised')
 assert len(list((work/'shards').glob('[0-9]*')))==1
 counts=s.predict(data,work,bundle,FakeEncoder(v))
 before=model.pairs
 again=s.predict(data,work,bundle,FakeEncoder(v,fail_at=1))
 assert again==counts and model.pairs==before
 out=s.merge_outputs(data,work)
 m=pd.read_csv(out/'matching_results.tsv',sep='\t',keep_default_na=False)
 cs=pd.read_csv(out/'candidate_pairs.tsv',sep='\t',keep_default_na=False)
 assert len(m)==len(cs)==61 and m.source1_entity_id.is_unique
 assert m.iloc[-1].matched_entity_ids==cs.iloc[-1].candidate_entity_ids==''
 assert m.source1_entity_id.tolist()==frame.entity_id.tolist()
 assert counts['candidate_pairs']==1200
 assert counts['empty_candidates']==1
 print('PASS: test-only encoding; three countries including France; batched reranking equivalent to evaluated retrieval; every scored pair exported; interrupted predictions resume without duplicates; cached rerun; 61 rows including empty business; TSV headers and ordering.')
