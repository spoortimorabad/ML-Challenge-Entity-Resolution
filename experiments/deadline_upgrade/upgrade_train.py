"""Reuse read-only training embeddings; train 30k, tune 2k, evaluate 2k fresh S1 entities."""
import os,sys,json,pickle,hashlib,time,gc,argparse
from pathlib import Path
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
import embedding_core as c
import submit_pipeline as s

SEED=20260927
TRAIN_N=30000
TUNE_N=2000
EVAL_N=2000
PROFILES={'standard':(64,200),'strong':(128,400)}

def resolve_cache(explicit=None):
    if explicit:candidates=[Path(explicit)]
    else:
        candidates=[x.parent for x in Path('/kaggle/input').rglob('encoding_complete.json')]
        local=Path('/kaggle/working/full_pool_embedding_v1')
        if local.exists():candidates.append(local)
    valid=[]
    for p in sorted(set(candidates)):
        required=['vectors.f16','records.sqlite','index_catalog.json','encoding_complete.json',
                  'sample.tsv','matcher.pkl','experiment_results.json','configuration.json']
        if not all((p/n).exists() for n in required):continue
        meta=json.loads((p/'experiment_results.json').read_text())
        if meta.get('version')!='full-pool-e5-v1':continue
        assert meta['model_id']==c.MODEL_ID and meta['revision']==c.REVISION
        count=json.loads((p/'encoding_complete.json').read_text())['total']
        assert (p/'vectors.f16').stat().st_size==count*c.DIM*2
        catalog=json.loads((p/'index_catalog.json').read_text())
        assert sum(x['count'] for x in catalog)==count
        assert all((p/x['file']).exists() for x in catalog)
        valid.append(p)
    if len(valid)!=1:
        raise RuntimeError('Attach the FULL saved v1 notebook output as input, not only the small results ZIP. '
          'Set CACHE_DIR to its full_pool_embedding_v1 folder if more than one is found. Found: '+str(valid))
    return valid[0]

def setup(data,cache,work):
    work.mkdir(parents=True,exist_ok=True)
    old=json.loads((cache/'configuration.json').read_text())
    assert old['revision']==c.REVISION and old['dim']==c.DIM and old['max_length']==256
    for path,size in old['files']:
        actual=data/'train'/Path(path).name
        assert actual.exists() and actual.stat().st_size==size,'Training dataset does not match cache.'
    config={'version':'deadline-upgrade-v2','seed':SEED,'train_n':TRAIN_N,'tune_n':TUNE_N,'evaluation_n':EVAL_N,
      'old_matcher_sha256':s.sha256(cache/'matcher.pkl'),'old_sample_sha256':s.sha256(cache/'sample.tsv'),
      'cache_manifest_sha256':s.sha256(cache/'encoding_complete.json'),
      'profiles':PROFILES,'features':c.MATCH_FEATURES,
      'core_sha256':s.sha256(Path(c.__file__)),'runner_sha256':s.sha256(Path(s.__file__)),
      'upgrade_sha256':s.sha256(Path(__file__))}
    config=json.loads(json.dumps(config))
    path=work/'upgrade_config.json'
    if path.exists():assert json.loads(path.read_text())==config,'Configuration changed. Use a new UPGRADE_WORK.'
    else:c.atomic_json(path,config)


def sample_data(data,cache,work):
    path=work/'sample.pkl'
    if path.exists():
        with path.open('rb') as f:return pickle.load(f)
    exclude=set(pd.read_csv(cache/'sample.tsv',sep='\t',dtype=str,keep_default_na=False).entity_id)
    rng=np.random.default_rng(SEED);sample=pd.DataFrame();n=TRAIN_N+TUNE_N+EVAL_N
    for chunk in c.read_chunks(data/'train/train_source1.tsv'):
        chunk=chunk.loc[~chunk.entity_id.isin(exclude),c.COLS].copy()
        chunk['_priority']=rng.random(len(chunk))
        sample=pd.concat([sample,chunk],ignore_index=True).nsmallest(n,'_priority')
    sample=sample.drop(columns='_priority').reset_index(drop=True)
    assert len(sample)==n and sample.entity_id.is_unique
    assert not set(sample.entity_id)&exclude
    ids=set(sample.entity_id);truth={}
    for chunk in c.read_chunks(data/'train/train_ground_truth.tsv',['source1_entity_id','matched_entity_ids']):
        for r in chunk[chunk.source1_entity_id.isin(ids)].itertuples(index=False):
            truth[r.source1_entity_id]=set(x for x in r.matched_entity_ids.split(',') if x)
    assert set(truth)==ids
    strata=sample.country+'|'+sample.entity_id.map(lambda x:str(not truth[x]))
    _,held=train_test_split(np.arange(n),test_size=TUNE_N+EVAL_N,random_state=SEED,stratify=strata)
    tune,evaluation=train_test_split(held,test_size=EVAL_N,random_state=SEED+1,stratify=strata.iloc[held])
    sample['split']='train';sample.loc[tune,'split']='tune';sample.loc[evaluation,'split']='evaluation'
    sample.to_csv(work/'sample.tsv',sep='\t',index=False)
    with Path(str(path)+'.tmp').open('wb') as f:pickle.dump((sample,truth),f)
    os.replace(str(path)+'.tmp',path)
    return sample,truth


def queries(sample,work,encoder):
    path=work/'queries.npy'
    if path.exists():return np.load(path)
    out=[]
    for start in range(0,len(sample),1000):
        out.append(encoder.encode(sample.iloc[start:start+1000]))
        c.log(f'Query embeddings: {min(start+1000,len(sample)):,}/{len(sample):,}')
    q=np.concatenate(out)
    with Path(str(path)+'.tmp').open('wb') as f:np.save(f,q)
    os.replace(str(path)+'.tmp',path)
    return q


def pair_arrays(frame,truth,owners_global,ids,scores,records):
    xs=[];ys=[];owners=[]
    for local,left in enumerate(frame[c.COLS].itertuples(index=False,name=None)):
        for rank,rid in enumerate(ids[local]):
            if rid<0:continue
            right=records[int(rid)]
            assert c.norm(left[3])==c.norm(right[3])
            xs.append(c.features(left,right)+[float(scores[local,rank])])
            ys.append(int(right[0] in truth[left[0]]));owners.append(int(owners_global[local]))
    return dict(x=np.asarray(xs,np.float32).reshape(-1,len(c.MATCH_FEATURES)),
                y=np.asarray(ys,np.uint8),owners=np.asarray(owners,np.int32))


def prepare_pairs(cache,work,sample,truth,q,profile,indices):
    npb,retr=PROFILES[profile]
    folder=work/f'pairs_{profile}';folder.mkdir(exist_ok=True)
    search=s.TestSearcher(cache,nprobe=npb,overfetch=retr);search.faiss.omp_set_num_threads(4)
    pieces=[];started=time.time()
    try:
        with threadpool_limits(limits=4):
            for start in range(0,len(indices),500):
                selected=indices[start:start+500];path=folder/f'{start:07d}.npz'
                if path.exists():
                    with np.load(path) as z:
                        assert np.array_equal(z['query_indices'],selected)
                        piece={k:z[k].copy() for k in ['x','y','owners']}
                else:
                    frame=sample.iloc[selected].reset_index(drop=True)
                    ids,scores=search.search(frame,q[selected]);records=search.fetch(ids)
                    piece=pair_arrays(frame,truth,selected,ids,scores,records)
                    with Path(str(path)+'.tmp').open('wb') as f:np.savez(f,query_indices=selected,**piece)
                    os.replace(str(path)+'.tmp',path)
                    del records,ids,scores
                pieces.append(piece)
                if start==0 or (start+len(selected))%2500==0:
                    c.log(f'{profile}: {start+len(selected):,}/{len(indices):,} businesses; {(time.time()-started)/60:.1f} min')
    finally:search.close()
    return {k:np.concatenate([piece[k] for piece in pieces]) for k in ['x','y','owners']}


def metrics(prob,pairs,selected,truth_counts,threshold):
    n=len(truth_counts);accepted=prob>=threshold
    predicted=np.bincount(pairs['owners'][accepted],minlength=n)
    tp=np.bincount(pairs['owners'][accepted & (pairs['y']==1)],minlength=n)
    fp=predicted-tp;fn=truth_counts-tp
    denom=1.25*tp+fp+.25*fn
    score=np.divide(1.25*tp,denom,out=np.zeros(n,dtype=float),where=denom!=0)
    score[(truth_counts==0)&(predicted==0)]=1
    singleton=truth_counts[selected]==0
    return dict(macro_F05=float(score[selected].mean()),correct_accepted=int(tp[selected].sum()),
       incorrect_accepted=int(fp[selected].sum()),missed_true_links=int(fn[selected].sum()),
       singletons_correct=int(((predicted[selected]==0)&singleton).sum()),singletons=int(singleton.sum()))


def tune(prob,pairs,selected,counts):
    trials=[dict(threshold=float(t),**metrics(prob,pairs,selected,counts,t)) for t in np.linspace(.05,.95,91)]
    return max(trials,key=lambda x:(x['macro_F05'],x['threshold'])),trials


def retrieval(pairs,selected,counts):
    hits=np.bincount(pairs['owners'],weights=pairs['y'],minlength=len(counts))
    sizes=np.bincount(pairs['owners'],minlength=len(counts))
    found=int(hits[selected].sum());total=int(counts[selected].sum())
    return dict(found=found,true_links=total,recall=found/total if total else 1,
                mean_candidates=float(sizes[selected].mean()),max_candidates=int(sizes[selected].max()))


def benchmark(cache,sample,encoder,bundle,profile):
    # Raw text -> embedding -> search -> lexical features -> model score, not search alone.
    npb,over=PROFILES[profile];search=s.TestSearcher(cache,npb,over)
    frame=sample.sample(n=min(1000,len(sample)),random_state=42).reset_index(drop=True)
    elapsed=[]
    try:
        with threadpool_limits(limits=4):
            search.faiss.omp_set_num_threads(4)
            for start in range(0,len(frame),250):
                part=frame.iloc[start:start+250].reset_index(drop=True)
                t=time.time();q=encoder.encode(part);ids,score=search.search(part,q)
                records=search.fetch(ids);s.score_candidates(part,ids,score,records,bundle)
                elapsed.append(time.time()-t)
    finally:search.close()
    return len(frame)/sum(elapsed)


def run(data,cache,work,target_finish,batch_size=64):
    data=Path(data);cache=Path(cache);work=Path(work)
    setup(data,cache,work)
    old_bundle,old_meta=s.load_model(cache)
    sample,truth=sample_data(data,cache,work)
    c.log('Fresh sample: '+str(sample.split.value_counts().to_dict()))
    encoder=c.Encoder(batch_size)
    q=queries(sample,work,encoder)
    strong=prepare_pairs(cache,work,sample,truth,q,'strong',np.arange(len(sample)))
    held=np.flatnonzero(sample.split.to_numpy()!='train')
    standard=prepare_pairs(cache,work,sample,truth,q,'standard',held)
    train_mask=sample.split.to_numpy()[strong['owners']]=='train'
    saved=work/'new_classifier.pkl'
    if saved.exists():
        with saved.open('rb') as f:model=pickle.load(f)
    else:
        c.log(f'Training 30k-business matcher: {train_mask.sum():,} pairs; {strong["y"][train_mask].sum():,} positives')
        model=HistGradientBoostingClassifier(learning_rate=.08,max_iter=250,max_leaf_nodes=31,
                    l2_regularization=1.,early_stopping=False,random_state=2026)
        with threadpool_limits(limits=4):model.fit(strong['x'][train_mask],strong['y'][train_mask])
        with Path(str(saved)+'.tmp').open('wb') as f:pickle.dump(model,f)
        os.replace(str(saved)+'.tmp',saved)
    strong={k:a[~train_mask] for k,a in strong.items()}
    pairs={'standard':standard,'strong':strong}
    counts=np.array([len(truth[x]) for x in sample.entity_id])
    ti=np.flatnonzero(sample.split.to_numpy()=='tune');ei=np.flatnonzero(sample.split.to_numpy()=='evaluation')
    scores={};options={};benchmark_rates={};all_trials=[]
    total_test=sum(len(chunk) for chunk in c.read_chunks(data/'test/test_source1.tsv',['entity_id']))
    for name,arr in pairs.items():
        with threadpool_limits(limits=4):scores[name]=model.predict_proba(arr['x'])[:,1]
        best,trials=tune(scores[name],arr,ti,counts);options[name]=best
        all_trials.extend(dict(profile=name,**r) for r in trials)
        bundle={'model':model,'threshold':best['threshold']}
        benchmark_rates[name]=benchmark(cache,sample,encoder,bundle,name)
        c.log(f'{name}: tuning macro F0.5={best["macro_F05"]:.5f}; prediction benchmark={benchmark_rates[name]:.1f} businesses/sec')
    # Profile and threshold chosen using tuning scores and time budget, before evaluation.
    finish=datetime.fromisoformat(target_finish)
    assert finish.tzinfo is not None,'Use an ISO timestamp with timezone, e.g. +05:30.'
    remaining_hours=(finish-datetime.now(timezone.utc)).total_seconds()/3600
    # 35% slower than warm training benchmark, plus 2h test encoding/index/export allowance.
    estimates={name:total_test/rate/3600*1.35+2.0 for name,rate in benchmark_rates.items()}
    feasible=[name for name in pairs if estimates[name]<=remaining_hours]
    chosen=max(feasible or list(pairs),key=lambda name:(options[name]['macro_F05'],name=='standard'))
    cutoff=options[chosen]['threshold']
    with threadpool_limits(limits=4):old_prob=old_bundle['model'].predict_proba(standard['x'])[:,1]
    baseline=metrics(old_prob,standard,ei,counts,old_bundle['threshold'])
    evaluation=metrics(scores[chosen],pairs[chosen],ei,counts,cutoff)
    search_eval=retrieval(pairs[chosen],ei,counts)
    evaluation['missed_by_search']=search_eval['true_links']-search_eval['found']
    evaluation['retrieved_but_rejected']=evaluation['missed_true_links']-evaluation['missed_by_search']
    # Only one fresh evaluation comparison; retain the already-submitted fallback on regression.
    promote=bool(feasible and evaluation['macro_F05']>baseline['macro_F05'])
    report={'old_model_fresh_evaluation':baseline,'new_model_fresh_evaluation':evaluation,
       'tuning_options':options,'chosen_profile':chosen,'threshold':cutoff,
       'search_recall_evaluation':search_eval,'prediction_benchmark_per_second':benchmark_rates,
       'estimated_test_hours_with_margin':estimates,'hours_until_target_finish':remaining_hours,
       'target_finish':target_finish,'promote_to_test':promote,
       'note':'Timing is an estimate, not a guarantee. Evaluation excludes all previous 4000 sampled S1 records.'}
    c.atomic_json(work/'upgrade_report.json',report)
    pd.DataFrame(all_trials).to_csv(work/'threshold_tuning.csv',index=False)
    by_country=[]
    for country in sorted(sample.country.unique()):
        selected=ei[sample.country.to_numpy()[ei]==country]
        by_country.append({'country':country,'businesses':len(selected),
           'old':metrics(old_prob,standard,selected,counts,old_bundle['threshold']),
           'new':metrics(scores[chosen],pairs[chosen],selected,counts,cutoff)})
    c.atomic_json(work/'country_evaluation.json',by_country)
    model_dir=work/'model';model_dir.mkdir(exist_ok=True)
    bundle={'model':model,'threshold':cutoff,'features':c.MATCH_FEATURES,
            'ann_nprobe':PROFILES[chosen][0],'ann_overfetch':PROFILES[chosen][1]}
    with (model_dir/'matcher.pkl').open('wb') as f:pickle.dump(bundle,f)
    from importlib.metadata import version
    meta={'version':'deadline-upgrade-v2','model_id':c.MODEL_ID,'revision':c.REVISION,'candidate_k':20,
       'ann_nprobe':PROFILES[chosen][0],'ann_overfetch':PROFILES[chosen][1],'threshold':cutoff,
       'evaluation':evaluation,'tuning':options[chosen],'features':c.MATCH_FEATURES,
       'train_businesses':TRAIN_N,'tune_businesses':TUNE_N,'evaluation_businesses':EVAL_N,
       'versions':{p:version(p) for p in ['numpy','pandas','scikit-learn','faiss-cpu','torch','transformers']}}
    c.atomic_json(model_dir/'experiment_results.json',meta)
    print(json.dumps(report,indent=2),flush=True)
    print('AUTO TEST RUN APPROVED' if promote else 'KEEP 0.84 FALLBACK: validation did not improve or estimated runtime exceeds target.',flush=True)
    del encoder;gc.collect()
    import torch;torch.cuda.empty_cache()
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--work',type=Path,required=True)
    ap.add_argument('--target-finish',required=True);ap.add_argument('--batch-size',type=int,default=64)
    args=ap.parse_args();run(args.data,args.cache,args.work,args.target_finish,args.batch_size)
