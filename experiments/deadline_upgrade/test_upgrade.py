import tempfile,sys,json,pickle,hashlib,types,importlib.metadata
from pathlib import Path
import numpy as np,pandas as pd,sklearn
import embedding_core as c,submit_pipeline as s,upgrade_train as u
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

# Macro F0.5 and singleton accounting must match the old implementation exactly.
sample=pd.DataFrame({'entity_id':['a','b','c'],'split':['evaluation']*3})
truth={'a':{'x','y'},'b':set(),'c':{'z'}}
a={'owners':np.array([0,0,1,2]),'y':np.array([1,0,0,1])}
prob=np.array([.9,.8,.3,.4]);counts=np.array([2,0,1]);ids=['x','wrong','wrong2','z']
for t in [.2,.5,.85]:
 old=c.metrics(sample,truth,a['owners'],ids,prob,t,'evaluation')
 new=u.metrics(prob,a,np.arange(3),counts,t)
 assert old==new,(old,new)

class FakeEncoder:
 def __init__(self,*args):pass
 def encode(self,frame):
  out=[]
  for name in frame.business_name:
   seed=int(hashlib.sha256(name.encode()).hexdigest()[:8],16)
   x=np.random.default_rng(seed).normal(size=c.DIM).astype('float32');out.append(x/np.linalg.norm(x))
  return np.asarray(out,np.float32)
c.Encoder=FakeEncoder
sys.modules['torch']=types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda:None))
real_version=importlib.metadata.version
importlib.metadata.version=lambda p: 'synthetic-test' if p in ['torch','transformers'] else real_version(p)
u.TRAIN_N=30;u.TUNE_N=10;u.EVAL_N=10
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);data=root/'dataset';(data/'train').mkdir(parents=True);cache=root/'cache';cache.mkdir()
 rows=[(f'S1-{i}',f'Business {i}',f'{i} Main Road','US' if i%2 else 'India') for i in range(100)]
 frame=pd.DataFrame(rows,columns=c.COLS)
 frame.to_csv(data/'train/train_source1.tsv',sep='\t',index=False)
 for source in (2,3):
  target=[(f'S{source}-{i}',f'Business {i}' if source==2 else f'Different {i}',f'{i} Main Road','US' if i%2 else 'India') for i in range(100)]
  pd.DataFrame(target,columns=c.COLS).to_csv(data/f'train/train_source{source}.tsv',sep='\t',index=False)
 pd.DataFrame({'source1_entity_id':frame.entity_id,'matched_entity_ids':[f'S2-{i}' if i%3 else '' for i in range(100)]}).to_csv(data/'train/train_ground_truth.tsv',sep='\t',index=False)
 (data/'test').mkdir();frame.iloc[:20].to_csv(data/'test/test_source1.tsv',sep='\t',index=False)
 c.encode_pool(data,cache,FakeEncoder());c.build_indexes(cache)
 frame.iloc[:4].assign(split='train').to_csv(cache/'sample.tsv',sep='\t',index=False)
 c.atomic_json(cache/'configuration.json',{'revision':c.REVISION,'dim':c.DIM,'max_length':256,
   'files':[(str(p),p.stat().st_size) for p in sorted((data/'train').iterdir())]})
 with threadpool_limits(limits=4):
  old=HistGradientBoostingClassifier(max_iter=2,random_state=4).fit(np.random.default_rng(4).random((100,19)),np.tile([0,1],50))
 bundle={'model':old,'threshold':.7,'features':c.MATCH_FEATURES}
 with (cache/'matcher.pkl').open('wb') as f:pickle.dump(bundle,f)
 c.atomic_json(cache/'experiment_results.json',{'version':'full-pool-e5-v1','model_id':c.MODEL_ID,
  'revision':c.REVISION,'features':c.MATCH_FEATURES,'candidate_k':20,'ann_nprobe':64,'ann_overfetch':200,
  'threshold':.7,'versions':{'scikit-learn':sklearn.__version__}})
 assert u.resolve_cache(str(cache))==cache
 before={p.name:s.sha256(p) for p in cache.iterdir() if p.is_file()}
 work=root/'upgrade'
 report=u.run(data,cache,work,'2030-09-27T22:30:00+05:30')
 sampled=pd.read_csv(work/'sample.tsv',sep='\t')
 assert sampled.split.value_counts().to_dict()=={'train':30,'evaluation':10,'tune':10}
 assert not set(sampled.entity_id)&set(frame.iloc[:4].entity_id)
 assert {p.name:s.sha256(p) for p in cache.iterdir() if p.is_file()}==before,'Read-only caches changed'
 newbundle,meta=s.load_model(work/'model')
 assert meta['train_businesses']==30
 assert (meta['ann_nprobe'],meta['ann_overfetch']) in [(64,200),(128,400)]
 with (work/'pairs_strong/0000000.npz').open('rb') as f:
  z=np.load(f);assert z['x'].shape==(1000,19)
 # Cached feature generation is reused on rerun.
 report2=u.run(data,cache,work,'2030-09-27T22:30:00+05:30')
 assert report['new_model_fresh_evaluation']==report2['new_model_fresh_evaluation']
 print('PASS: scoring equivalence, fresh disjoint splits, both search profiles, new model fitting/tuning, metadata reload, benchmark/gate, cache immutability, checkpoint reuse. Real GPU and challenge data not run locally.')
