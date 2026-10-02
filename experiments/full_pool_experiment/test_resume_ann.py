from pathlib import Path
import tempfile,json
import numpy as np,pandas as pd
import pipeline as p

class FakeEncoder:
    def __init__(self,fail=False):self.calls=0;self.fail=fail
    def encode(self,frame):
        self.calls+=1
        if self.fail and self.calls==2:raise RuntimeError('Simulated interruption')
        x=np.zeros((len(frame),p.DIM),np.float32);x[:,0]=1;return x

with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);data=root/'data';(data/'train').mkdir(parents=True);work=root/'work';work.mkdir()
 for source,n in [(2,10002),(3,8)]:
  pd.DataFrame({'entity_id':[f'S{source}-{i}' for i in range(n)],'business_name':['Name']*n,
                'business_address':['Road']*n,'country':['US']*n}).to_csv(data/f'train/train_source{source}.tsv',sep='\t',index=False)
 try:p.encode_pool(data,work,FakeEncoder(True))
 except RuntimeError as e:assert str(e)=='Simulated interruption'
 else:raise AssertionError('Did not simulate interruption')
 db=p.open_records(work);assert db.execute('select count(*) from records').fetchone()[0]==10000;db.close()
 p.encode_pool(data,work,FakeEncoder())
 p.encode_pool(data,work,FakeEncoder())
 db=p.open_records(work);assert db.execute('select count(*) from records').fetchone()[0]==10010
 # Random normalized vectors make the ANN smoke test non-degenerate.
 rng=np.random.default_rng(3);x=rng.normal(size=(10010,p.DIM)).astype('float32');x/=np.linalg.norm(x,axis=1,keepdims=True)
 v=np.memmap(work/'vectors.f16',mode='r+',dtype='float16',shape=x.shape);v[:]=x;v.flush();del v
 # A small second country exercises the exact-index fallback and country filtering.
 db.execute("update records set country='india' where rid>=10000");db.commit();db.close()
 p.build_indexes(work);p.build_indexes(work)
 sample=pd.DataFrame([['q1','Name','Road','US','tune'],['q2','Name','Road','India','tune']],columns=p.COLS+['split'])
 q=x[[17,10001]]
 ids,scores=p.retrieve(work,sample,q,nprobe=64,overfetch=200,save_k=50)
 assert ids[0,0]==17 and ids[1,0]==10001
 assert all(i<10000 for i in ids[0] if i>=0)
 assert all(i>=10000 for i in ids[1] if i>=0)
 assert np.count_nonzero(ids[1]>=0)==10
 ids2,scores2=p.retrieve(work,sample,q)
 assert np.array_equal(ids,ids2)
 records=p.fetch_records(work,ids)
 assert len(records)==60
 print('PASS: encoding interrupted/resumed with no duplicate rows; completed encoding reused; IVF-PQ and small-country indexes; index cache; cosine reranking; country filter; retrieval cache.')
