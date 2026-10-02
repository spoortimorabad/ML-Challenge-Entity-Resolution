import tempfile,sys,json,pickle,shutil,types,hashlib,zipfile
from pathlib import Path
import numpy as np,pandas as pd,sklearn
root=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as tmp:
 tmp=Path(tmp);code=tmp/'code';code.mkdir()
 (code/'embedding_core.py').write_text((root/'test_country_core_source.txt').read_text())
 for name in ['submit_pipeline.py','country_submit.py']:shutil.copy2(root/name,code/name)
 sys.path.insert(0,str(code))
 import embedding_core as c,submit_pipeline as s,country_submit as country
 class FakeEncoder:
  def __init__(self,batch_size):pass
  def encode(self,frame):
   rows=[]
   for name in frame.business_name:
    seed=int(hashlib.sha256(name.encode()).hexdigest()[:8],16)
    v=np.random.default_rng(seed).normal(size=c.DIM).astype('float32');rows.append(v/np.linalg.norm(v))
   return np.asarray(rows,np.float32)
 class Model:
  n_features_in_=19;classes_=np.array([0,1])
  def predict_proba(self,x):return np.tile([.1,.9],(len(x),1))
 c.Encoder=FakeEncoder
 sys.modules['torch']=types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda:None))
 s.PREDICT_CHUNK=2
 data=tmp/'dataset';(data/'test').mkdir(parents=True);(tmp/'utils').mkdir()
 (tmp/'utils/validate_submission.py').write_text('# placeholder: real validator is run in Kaggle')
 rows=[['S1-0','Alpha','1 Main','US'],['S1-1','Beta','2 Main','India'],['S1-2','Gamma','3 Main','France'],['S1-3','Zero','4 Main','ZZ'],['S1-4','Other','5 Main','US']]
 pd.DataFrame(rows,columns=c.COLS).to_csv(data/'test/test_source1.tsv',sep='\t',index=False)
 for source in (2,3):
  target=[[f'S{source}-{i}',r[1],r[2],r[3]] for i,r in enumerate(rows) if r[3]!='ZZ']
  pd.DataFrame(target,columns=c.COLS).to_csv(data/f'test/test_source{source}.tsv',sep='\t',index=False)
 model=tmp/'training';model.mkdir();(model/'vectors.f16').write_bytes(b'KEEP TRAINING CACHE')
 bundle={'model':Model(),'threshold':.7,'features':c.MATCH_FEATURES}
 with (model/'matcher.pkl').open('wb') as f:pickle.dump(bundle,f)
 meta={'version':'full-pool-e5-v1','model_id':c.MODEL_ID,'revision':c.REVISION,'candidate_k':20,
 'ann_nprobe':64,'ann_overfetch':200,'threshold':.7,'features':c.MATCH_FEATURES,
 'versions':{'scikit-learn':sklearn.__version__}}
 (model/'experiment_results.json').write_text(json.dumps(meta))
 work=tmp/'submission';work.mkdir();shutil.copy2(code/'country_submit.py',work/'country_submit.py')
 checks=[]
 def validate(data,work):
  m=pd.read_csv(work/'output/matching_results.tsv',sep='\t',keep_default_na=False)
  ca=pd.read_csv(work/'output/candidate_pairs.tsv',sep='\t',keep_default_na=False)
  assert m.source1_entity_id.tolist()==[r[0] for r in rows]
  assert m.source1_entity_id.tolist()==ca.source1_entity_id.tolist()
  assert m.iloc[3].matched_entity_ids==ca.iloc[3].candidate_entity_ids==''
  assert m.source1_entity_id.is_unique
  checks.append('validated synthetic output')
 def package(work,meta):
  p=work/'backup.zip'
  with zipfile.ZipFile(p,'w') as z:z.writestr('test.txt','synthetic package hook')
  return p
 s.validate=validate;s.package=package
 country.run(data,work,model)
 assert len(checks)==1
 assert (model/'vectors.f16').read_bytes()==b'KEEP TRAINING CACHE'
 assert not list((work/'countries').glob('*/cache'))
 assert not list((work/'countries').glob('*/input'))
 # A rerun must not encode a completed country again.
 class FailEncoder:
  def __init__(self,*args):raise AssertionError('Should reuse committed country results')
 c.Encoder=FailEncoder
 country.run(data,work,model)
 assert len(checks)==2
 print('PASS: country partition, model reuse, end-to-end synthetic prediction, original S1 order, empty country handling, durable country resume, temporary-test cleanup, training-cache preservation. Official validator/GPU remain Kaggle checks.')
