import json
from pathlib import Path

cells=[]
def md(s): cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
def code(s): cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=s.splitlines(True)))
md('''# Candidate retrieval and matcher gap audit
Run in a separate Kaggle notebook with the saved original full_pool_embedding_v1 output and the saved ml_deadline_upgrade_v2 output attached. No raw dataset, GPU, new embeddings, retraining, or test labels are required. Only load your own trusted pickle files.

This audits a seeded sample of **500 tuning businesses**, including singletons. Evaluation businesses remain unused. It compares search settings and top-K limits with the saved classifier and a fixed standard-profile threshold. Larger-K scores are diagnostic, not separately tuned estimates.

About 9 GiB is copied to a new working directory for representative local I/O. Source inputs are read-only; no existing submission is modified or deleted. Exact search examines up to 10 randomly selected businesses with baseline retrieval misses. Its conclusions apply only to that subset. Save a Kaggle version to retain results.
''')
code('''%pip install -q numpy==2.0.2 pandas==2.3.3 scikit-learn==1.6.1 faiss-cpu==1.11.0 threadpoolctl
''')
code('''from pathlib import Path
import os, sys, json, pickle, shutil, time, hashlib, gc, zipfile
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

# Set explicit paths if more than one matching saved output is attached.
UPGRADE_PATH = ""
CACHE_PATH = ""
N_TUNE = 500
EXACT_QUERY_LIMIT = 10
ROOT = Path('/kaggle/working/entity_gap_audit_v1')
OUT = ROOT / 'results'
OUT.mkdir(parents=True, exist_ok=True)

def discover(explicit, marker, required):
    paths = [Path(explicit)] if explicit else sorted({p.parent for p in Path('/kaggle/input').rglob(marker)})
    paths = [p for p in paths if all((p / n).exists() for n in required)]
    assert len(paths) == 1, f'Set explicit path; found {paths}'
    return paths[0]

UPGRADE = discover(UPGRADE_PATH, 'upgrade_report.json',
    ['sample.pkl', 'queries.npy', 'model/matcher.pkl', 'model/experiment_results.json'])
SOURCE = discover(CACHE_PATH, 'encoding_complete.json',
    ['vectors.f16', 'records.sqlite', 'index_catalog.json', 'sample.tsv', 'configuration.json'])
SAVED_CODE = UPGRADE.parent / 'code'
assert (SAVED_CODE/'submit_pipeline.py').exists()
sys.path.insert(0, str(SAVED_CODE))
import embedding_core as c
import submit_pipeline as s
bundle, model_meta = s.load_model(UPGRADE/'model')
report = json.loads((UPGRADE/'upgrade_report.json').read_text())
threshold = float(report['tuning_options']['standard']['threshold'])
with (UPGRADE/'sample.pkl').open('rb') as f:
    sample, truth = pickle.load(f)
queries = np.load(UPGRADE/'queries.npy', mmap_mode='r')
eligible = np.flatnonzero(sample['split'].to_numpy() == 'tune')
chosen = np.random.default_rng(20260928).choice(eligible, min(N_TUNE,len(eligible)), replace=False)
frame = sample.iloc[chosen].reset_index(drop=True)
q = np.ascontiguousarray(queries[chosen], dtype=np.float32)
assert np.allclose(np.linalg.norm(q,axis=1),1,atol=1e-4)
assert len(frame) and frame.entity_id.is_unique
frame.to_csv(OUT/'audit_businesses.tsv',sep='\\t',index=False)
print('Saved upgrade:',UPGRADE)
print('Training cache:',SOURCE)
print('Tuning businesses:',len(frame),'fixed cutoff:',threshold)
''')
code('''# Copy immutable cache files locally; never alter the attached source.
CACHE = ROOT/'local_cache'
CACHE.mkdir(exist_ok=True)
catalog = json.loads((SOURCE/'index_catalog.json').read_text())
names = ['vectors.f16','records.sqlite','encoding_complete.json','index_catalog.json'] + [x['file'] for x in catalog]
manifest = {'source':str(SOURCE), 'sizes':{n:(SOURCE/n).stat().st_size for n in names}}
identity = CACHE/'copy_identity.json'
if identity.exists():
    assert json.loads(identity.read_text()) == manifest, 'Different source cache; use a new ROOT.'
else:
    assert not any(CACHE.iterdir()), 'Unknown files in cache directory; use a new ROOT.'
    identity.write_text(json.dumps(manifest))
need = sum(manifest['sizes'][n] for n in names if not (CACHE/n).exists())
assert shutil.disk_usage(CACHE).free > need + 2*2**30, 'Need enough disk for cache plus 2 GiB.'
for name in names:
    dest=CACHE/name
    if dest.exists():
        assert dest.stat().st_size == manifest['sizes'][name]
        continue
    print('Copying',name,flush=True)
    temp=dest.with_name(dest.name+'.partial')
    shutil.copyfile(SOURCE/name,temp)
    assert temp.stat().st_size == manifest['sizes'][name]
    os.replace(temp,dest)
print('Local cache ready.')
''')
code('''# Truth is used only for scoring/diagnosis, never to choose search candidates.
searcher = s.TestSearcher(CACHE,64,200)
searcher.faiss.omp_set_num_threads(4)
wanted = set().union(*(truth[sid] for sid in frame.entity_id))
truth_records = {}
country_codes = {x['country']:i for i,x in enumerate(catalog)}
row_country = np.full(searcher.total,-1,dtype=np.int16)
started=time.perf_counter()
cursor=searcher.db.execute('SELECT rid,entity_id,business_name,business_address,country FROM records')
while True:
    batch=cursor.fetchmany(100000)
    if not batch: break
    for rid,eid,name,address,country in batch:
        row_country[rid]=country_codes[c.norm(country)]
        if eid in wanted: truth_records[eid]=(rid,(eid,name,address,country))
assert set(truth_records)==wanted, 'Missing truth IDs in the attached training cache.'
assert (row_country>=0).all()
print('Truth lookup complete:',len(wanted),'links;',round(time.perf_counter()-started,1),'seconds')
''')
code('''def retrieve(nprobe, overfetch):
    ids=np.full((len(frame),overfetch),-1,np.int64)
    sims=np.full((len(frame),overfetch),-np.inf,np.float32)
    timing={'ann_seconds':0.,'vector_gather_rerank_seconds':0.}
    countries=frame.country.map(c.norm).to_numpy()
    for country in np.unique(countries):
        index=searcher.indexes[country]
        index.nprobe=min(nprobe,index.nlist)
        owners=np.flatnonzero(countries==country)
        for start in range(0,len(owners),32):
            ix=owners[start:start+32]
            t=time.perf_counter()
            _,raw=index.search(np.ascontiguousarray(q[ix]),overfetch)
            timing['ann_seconds']+=time.perf_counter()-t
            t=time.perf_counter()
            valid=raw>=0
            vectors=np.asarray(searcher.vectors[np.maximum(raw,0)],dtype=np.float32)
            vectors/=np.maximum(np.linalg.norm(vectors,axis=2,keepdims=True),1e-12)
            cosine=np.einsum('bkd,bd->bk',vectors,q[ix],optimize=False)
            for j,i in enumerate(ix):
                positions=np.flatnonzero(valid[j])
                order=positions[np.lexsort((raw[j,positions],-cosine[j,positions]))]
                ids[i,:len(order)]=raw[j,order]
                sims[i,:len(order)]=cosine[j,order]
            timing['vector_gather_rerank_seconds']+=time.perf_counter()-t
    return ids,sims,timing

def evaluate(ids, probabilities, labels, k):
    valid=ids[:,:k]>=0
    accepted=valid & (probabilities[:,:k]>=threshold)
    hits=(valid & labels[:,:k]).sum(axis=1)
    tp=(accepted & labels[:,:k]).sum(axis=1)
    fp=(accepted & ~labels[:,:k]).sum(axis=1)
    counts=np.array([len(truth[x]) for x in frame.entity_id])
    fn=counts-tp
    denom=1.25*tp+fp+.25*fn
    score=np.divide(1.25*tp,denom,out=np.zeros(len(frame)),where=denom!=0)
    score[(counts==0)&(accepted.sum(axis=1)==0)]=1
    return dict(k=k,true_links=int(counts.sum()),retrieved_true=int(hits.sum()),
        recall=float(hits.sum()/max(counts.sum(),1)),macro_F05=float(score.mean()),
        true_accepted=int(tp.sum()),false_accepted=int(fp.sum()),
        missed_search=int((counts-hits).sum()),retrieved_rejected=int((hits-tp).sum()),
        mean_candidates=float(valid.sum(axis=1).mean()),
        singletons=int((counts==0).sum()),singletons_correct=int(((counts==0)&(accepted.sum(axis=1)==0)).sum()))

profiles=[('standard',64,200),('strong',128,400),('wider',256,800)]
summaries=[]; timings=[]; audits=[]; false_positives=[]; runs={}
with threadpool_limits(limits=4):
    for profile,nprobe,overfetch in profiles:
        print('Searching',profile,flush=True)
        c.prepared.cache_clear()
        ids,sims,timing=retrieve(nprobe,overfetch)
        t=time.perf_counter()
        records=searcher.fetch(ids[:,:80])
        timing['record_lookup_seconds']=time.perf_counter()-t
        t=time.perf_counter()
        probabilities=np.full((len(frame),80),np.nan)
        labels=np.zeros((len(frame),80),bool)
        xs=[]; slots=[]
        for i,left in enumerate(frame[c.COLS].itertuples(index=False,name=None)):
            for j,rid in enumerate(ids[i,:80]):
                if rid<0: continue
                right=records[int(rid)]
                xs.append(c.features(left,right)+[float(sims[i,j])]);slots.append((i,j))
                labels[i,j]=right[0] in truth[left[0]]
        if xs:
            pred=bundle['model'].predict_proba(np.asarray(xs,np.float32))[:,1]
            for (i,j),p in zip(slots,pred): probabilities[i,j]=p
        timing['features_model_seconds']=time.perf_counter()-t
        timing.update(profile=profile,businesses=len(frame),timing_scope='search plus scoring top 80; excludes initial DB scan and encoding')
        timings.append(timing)
        for k in (20,40,80): summaries.append(dict(profile=profile,**evaluate(ids,probabilities,labels,k)))
        for i,left in enumerate(frame[c.COLS].itertuples(index=False,name=None)):
            for eid in sorted(truth[left[0]]):
                rid,right=truth_records[eid]
                pos=np.flatnonzero(ids[i]==rid)
                rank=int(pos[0]+1) if len(pos) else None
                status=('absent_from_ann' if rank is None else 'outside_top20' if rank>20 else
                    'classifier_accepted' if probabilities[i,rank-1]>=threshold else 'classifier_rejected')
                v=np.asarray(searcher.vectors[rid],np.float32)
                cos=float(v.dot(q[i])/max(np.linalg.norm(v),1e-12))
                audits.append(dict(profile=profile,query_row=i,source1_id=left[0],source1_name=left[1],
                    source1_address=left[2],country=left[3],true_id=eid,true_name=right[1],true_address=right[2],
                    candidate_address_missing=not bool(str(right[2]).strip()),
                    cosine=cos,rank_within_retrieved=rank,status=status,
                    model_score=float(probabilities[i,rank-1]) if rank is not None and rank<=80 else None))
            for j in range(20):
                if ids[i,j]>=0 and probabilities[i,j]>=threshold and not labels[i,j]:
                    right=records[int(ids[i,j])]
                    false_positives.append(dict(profile=profile,source1_id=left[0],source1_name=left[1],
                        source1_address=left[2],country=left[3],candidate_id=right[0],candidate_name=right[1],
                        candidate_address=right[2],model_score=float(probabilities[i,j]),cosine=float(sims[i,j])))
        runs[profile]=(ids,sims)
        pd.DataFrame(summaries).to_csv(OUT/'retrieval_matcher_summary.csv',index=False)
        pd.DataFrame(timings).to_csv(OUT/'stage_timings.csv',index=False)
        pd.DataFrame(audits).to_csv(OUT/'true_link_audit.tsv',sep='\\t',index=False)
        pd.DataFrame(false_positives).to_csv(OUT/'false_accepts.tsv',sep='\\t',index=False)
        print(pd.DataFrame(summaries).tail(3).to_string(index=False),flush=True)
print('Timing profiles run sequentially: later runs may benefit from OS cache warming.')
''')
md('''## Exact-search spot check
For up to 10 random businesses with standard top-20 retrieval misses, compare against **every same-country stored vector**, in memory-bounded batches. This is diagnostic only, never the proposed production blocker. It reads the vector file once and may take several minutes. Counts include a numerical-tolerance interval for nearly tied cosine scores. Truth IDs are used only to measure ranks, not to generate candidates.
''')
code('''base=pd.DataFrame(audits)
miss=base[(base.profile=='standard') & base.status.isin(['absent_from_ann','outside_top20'])]
available=np.sort(miss.query_row.unique())
selected=np.random.default_rng(19).choice(available,min(EXACT_QUERY_LIMIT,len(available)),replace=False)
exact=[]
if len(selected):
    items=miss[miss.query_row.isin(selected)].copy().reset_index(drop=True)
    mapping={int(owner):j for j,owner in enumerate(selected)}
    owner_columns=np.array([mapping[int(x)] for x in items.query_row])
    target_cos=items.cosine.to_numpy()
    owners_country=np.array([country_codes[c.norm(frame.iloc[int(i)].country)] for i in selected])
    greater=np.zeros(len(items),np.int64);near=np.zeros(len(items),np.int64)
    t=time.perf_counter()
    with threadpool_limits(limits=4):
        for start in range(0,searcher.total,50000):
            end=min(start+50000,searcher.total)
            v=np.array(searcher.vectors[start:end],dtype=np.float32)
            v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)
            sims=v @ q[selected].T
            for j in range(len(items)):
                column=owner_columns[j]
                values=sims[row_country[start:end]==owners_country[column],column]
                greater[j]+=np.count_nonzero(values>target_cos[j]+1e-6)
                near[j]+=np.count_nonzero(np.abs(values-target_cos[j])<=1e-6)
            if start%1000000==0: print(f'Exact scan {end:,}/{searcher.total:,}',flush=True)
    for j,r in items.iterrows():
        lo=int(greater[j]+1);hi=int(greater[j]+max(near[j],1))
        reason=('exact_top20_but_pipeline_missed' if hi<=20 else
                'embedding_rank_below_top20' if lo>20 else 'near_tie_at_top20_boundary')
        exact.append(dict(source1_id=r.source1_id,true_id=r.true_id,baseline_status=r.status,
                          exact_rank_min=lo,exact_rank_max=hi,diagnosis=reason))
    print('Exact scan seconds:',round(time.perf_counter()-t,1))
exact_df=pd.DataFrame(exact,columns=['source1_id','true_id','baseline_status','exact_rank_min','exact_rank_max','diagnosis'])
exact_df.to_csv(OUT/'exact_rank_spotcheck.csv',index=False)
print(exact_df.to_string(index=False))
''')
code('''audit=pd.DataFrame(audits)
breakdown=audit.groupby(['profile','country','candidate_address_missing','status'],dropna=False).size().rename('links').reset_index()
breakdown.to_csv(OUT/'error_breakdown.csv',index=False)
config={'tuning_only':True,'sample_seed':20260928,'selected_sample_indices':chosen.tolist(),
        'threshold':threshold,'profiles':profiles,'k_values':[20,40,80],
        'exact_query_limit':EXACT_QUERY_LIMIT,'source_cache':str(SOURCE),'saved_upgrade':str(UPGRADE),
        'limitations':['No new model training or threshold tuning.',
        'Exact ranks cover only sampled baseline misses.',
        'ANN profiles may have different candidate sets; their results need not be nested.',
        'Timing excludes encoding and includes top-80 feature computation; not a full submission ETA.',
        'Tuning data covers US and India, not France.']}
(OUT/'audit_config.json').write_text(json.dumps(config,indent=2))
searcher.close()
archive=ROOT/'entity_gap_audit_results.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for path in sorted(OUT.iterdir()): z.write(path,path.name)
print('SUMMARY')
display(pd.DataFrame(summaries))
print('BASELINE ERROR BREAKDOWN')
display(breakdown[breakdown.profile=='standard'])
print('Saved report:',archive)
print('Retain this notebook version and download the ZIP from Kaggle Output.')
''')
nb=dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}},nbformat=4,nbformat_minor=5)
path=Path(__file__).parent/'amazon_ml_gap_audit.ipynb'
path.write_text(json.dumps(nb,indent=2))
for cell in cells:
    if cell['cell_type']=='code':
        source=''.join(cell['source'])
        if not source.startswith('%'): compile(source,'cell','exec')
print(path.resolve())
