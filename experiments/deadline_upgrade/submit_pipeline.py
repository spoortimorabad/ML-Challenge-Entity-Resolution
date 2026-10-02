"""Run the evaluated E5 + matcher pipeline on the complete challenge test set.

No fitting or threshold selection occurs here. Candidate exports contain exactly
all final candidate pairs passed to the saved matcher.
"""
import argparse, csv, gc, hashlib, json, os, pickle, shutil, subprocess, sys, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import embedding_core as core

VERSION='embedding-deadline-v2'
PREDICT_CHUNK=2000

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def locate_model(explicit=None):
    if explicit:
        candidates=[Path(explicit)]
    else:
        candidates=[]
        local=Path('/kaggle/working/full_pool_embedding_v1')
        if (local/'matcher.pkl').exists():candidates.append(local)
        for path in Path('/kaggle/input').rglob('experiment_results.json'):
            parent=path.parent
            if (parent/'matcher.pkl').exists():candidates.append(parent)
    valid=[]
    for path in sorted(set(candidates)):
        result=path/'experiment_results.json'
        if result.exists() and (path/'matcher.pkl').exists():
            try:meta=json.loads(result.read_text())
            except (ValueError,OSError):continue
            if meta.get('version')=='full-pool-e5-v1' and meta.get('model_id')==core.MODEL_ID:valid.append(path)
    if len(valid)!=1:
        raise RuntimeError('Attach the completed full-pool experiment notebook OUTPUT as an input, '
          'then set MODEL_DIR to the folder containing matcher.pkl and experiment_results.json. '
          f'Found {len(valid)} matching folders: {[str(p) for p in valid]}')
    return valid[0]

def load_model(model_dir):
    import sklearn
    meta=json.loads((model_dir/'experiment_results.json').read_text())
    assert meta['revision']==core.REVISION and meta['features']==core.MATCH_FEATURES
    assert meta['candidate_k']==20 and (meta['ann_nprobe'],meta['ann_overfetch']) in [(64,200),(128,400)]
    assert sklearn.__version__==meta['versions']['scikit-learn'], (
        'Install scikit-learn=='+meta['versions']['scikit-learn']+' before loading this model.')
    # This must be YOUR trusted matcher produced by the previous notebook.
    with (model_dir/'matcher.pkl').open('rb') as f:bundle=pickle.load(f)
    assert bundle['features']==core.MATCH_FEATURES
    assert abs(float(bundle['threshold'])-float(meta['threshold']))<1e-10
    assert bundle['model'].n_features_in_==len(core.MATCH_FEATURES)
    assert list(bundle['model'].classes_)==[0,1]
    bundle['ann_nprobe']=meta['ann_nprobe'];bundle['ann_overfetch']=meta['ann_overfetch']
    return bundle,meta

def initialize(data,work,model_dir):
    data=Path(data);work=Path(work);model_dir=Path(model_dir)
    work.mkdir(parents=True,exist_ok=True)
    files=[data/f'test/test_source{i}.tsv' for i in (1,2,3)]
    validator=data.parent/'utils/validate_submission.py'
    for path in files+[validator]:
        if not path.exists():raise FileNotFoundError(path)
    bundle,meta=load_model(model_dir)
    core.log('Checking input fingerprints and saved model...')
    config={'version':VERSION,'inputs':{p.name:sha256(p) for p in files},
      'matcher_sha256':sha256(model_dir/'matcher.pkl'),'model_id':core.MODEL_ID,
      'revision':core.REVISION,'max_length':256,'dim':core.DIM,
      'candidate_k':20,'nprobe':meta['ann_nprobe'],'overfetch':meta['ann_overfetch'],'threshold':float(bundle['threshold']),
      'prediction_chunk':PREDICT_CHUNK,
      'core_sha256':sha256(Path(core.__file__)),'runner_sha256':sha256(Path(__file__))}
    cp=work/'submission_config.json'
    if cp.exists():assert json.loads(cp.read_text())==config,'Inputs, model or code changed. Use a NEW WORK directory.'
    else:core.atomic_json(cp,config)
    artifact=work/'model';artifact.mkdir(exist_ok=True)
    for name in ['matcher.pkl','experiment_results.json']:
        dest=artifact/name
        if dest.resolve()!=(model_dir/name).resolve():shutil.copy2(model_dir/name,dest)
    core.log(f"Saved matcher cutoff: {bundle['threshold']:.3f}; 20 candidates; {meta['ann_nprobe']} cells / {meta['ann_overfetch']} intermediate neighbors")
    return bundle,meta,config

class TestSearcher:
    def __init__(self,work,nprobe=64,overfetch=200):
        import faiss
        self.work=work;self.faiss=faiss;self.overfetch=overfetch
        self.total=json.loads((work/'encoding_complete.json').read_text())['total']
        self.vectors=np.memmap(work/'vectors.f16',mode='r',dtype=np.float16,shape=(self.total,core.DIM))
        self.db=sqlite3.connect(f'file:{(work/"records.sqlite").resolve()}?mode=ro',uri=True)
        self.db.execute('PRAGMA cache_size=-65536')
        self.indexes={}
        for item in json.loads((work/'index_catalog.json').read_text()):
            index=faiss.read_index(str(work/item['file']))
            assert index.ntotal==item['count']
            if hasattr(index,'nprobe'):index.nprobe=min(nprobe,index.nlist)
            self.indexes[item['country']]=index
    def search(self,frame,q):
        """Same approximate search and float16-vector cosine reranking as evaluation."""
        ids=np.full((len(frame),20),-1,np.int64)
        scores=np.full((len(frame),20),-1,np.float32)
        countries=frame.country.map(core.norm).to_numpy()
        for country in np.unique(countries):
            qi=np.flatnonzero(countries==country)
            index=self.indexes.get(country)
            if index is None:continue
            for start in range(0,len(qi),64):
                indices=qi[start:start+64]
                _,approx=index.search(np.ascontiguousarray(q[indices],dtype=np.float32),self.overfetch)
                # Batch gathers reduce Python/disk overhead; re-ranking remains cosine.
                valid=approx>=0
                safe=np.maximum(approx,0)
                x=np.asarray(self.vectors[safe],dtype=np.float32)
                x/=np.maximum(np.linalg.norm(x,axis=2,keepdims=True),1e-12)
                cos=np.einsum('bkd,bd->bk',x,q[indices],optimize=False)
                cos[~valid]=-np.inf
                for local,i in enumerate(indices):
                    positions=np.flatnonzero(valid[local])
                    if not len(positions):continue
                    order=np.lexsort((approx[local,positions],-cos[local,positions]))
                    chosen=positions[order[:20]]
                    ids[i,:len(chosen)]=approx[local,chosen]
                    scores[i,:len(chosen)]=cos[local,chosen]
                del x
        return ids,scores
    def fetch(self,ids):
        valid=np.unique(ids[ids>=0]);rows={}
        for start in range(0,len(valid),800):
            batch=[int(x) for x in valid[start:start+800]]
            sql='SELECT * FROM records WHERE rid IN ('+','.join('?'*len(batch))+')'
            for row in self.db.execute(sql,batch):rows[row[0]]=tuple(row[1:])
        assert len(rows)==len(valid),'Candidate IDs missing from test record store.'
        return rows
    def close(self):
        self.db.close();self.indexes.clear();del self.vectors

# sqlite is separate from core to keep this module directly runnable.
import sqlite3

def score_candidates(frame,ids,scores,records,bundle):
    candidates=[[] for _ in range(len(frame))];accepted=[[] for _ in range(len(frame))]
    features=[];owners=[];pair_ids=[]
    for i,left in enumerate(frame[core.COLS].itertuples(index=False,name=None)):
        seen=set()
        for rank,rid in enumerate(ids[i]):
            if rid<0:continue
            right=records[int(rid)]
            assert core.norm(left[3])==core.norm(right[3]),'Country mismatch in retrieval.'
            assert right[0].startswith(('S2-','S3-'))
            assert right[0] not in seen,'Duplicate candidate entity ID in test input/retrieval.'
            seen.add(right[0]);candidates[i].append(right[0])
            features.append(core.features(left,right)+[float(scores[i,rank])])
            owners.append(i);pair_ids.append(right[0])
    if features:
        x=np.asarray(features,np.float32)
        assert np.isfinite(x).all()
        prob=bundle['model'].predict_proba(x)[:,1]
        assert np.isfinite(prob).all()
        for owner,cid,score in zip(owners,pair_ids,prob):
            if score>=bundle['threshold']:accepted[owner].append(cid)
    assert sum(map(len,candidates))==len(features)
    for cand,match in zip(candidates,accepted):
        assert len(cand)<=20 and set(match)<=set(cand)
    return candidates,accepted

def write_shard(path,frame,candidates,accepted,start):
    staging=path.with_name(path.name+'.tmp');staging.mkdir(parents=True,exist_ok=True)
    for name,values,col in [('candidate_pairs.tsv',candidates,'candidate_entity_ids'),
                            ('matching_results.tsv',accepted,'matched_entity_ids')]:
        with (staging/name).open('w',encoding='utf-8',newline='') as f:
            writer=csv.writer(f,delimiter='\t',lineterminator='\n')
            writer.writerow(['source1_entity_id',col])
            for sid,links in zip(frame.entity_id,values):writer.writerow([sid,','.join(links)])
            f.flush();os.fsync(f.fileno())
    stats={'start':start,'rows':len(frame),'first':str(frame.entity_id.iloc[0]),'last':str(frame.entity_id.iloc[-1]),
       'candidate_pairs':sum(map(len,candidates)),'predicted_links':sum(map(len,accepted)),
       'empty_candidates':sum(not x for x in candidates),'empty_matches':sum(not x for x in accepted),
       'candidate_histogram':np.bincount([len(x) for x in candidates],minlength=21).tolist()}
    core.atomic_json(staging/'stats.json',stats)
    os.replace(staging,path)
    return stats

def predict(data,work,bundle,encoder):
    shards=work/'shards';shards.mkdir(exist_ok=True)
    searcher=TestSearcher(work,bundle.get("ann_nprobe",64),bundle.get("ann_overfetch",200))
    counts={};start=0;started=time.time();new=0
    total_s1=sum(len(c) for c in core.read_chunks(data/'test/test_source1.tsv',['entity_id']))
    core.log(f'Predicting all {total_s1:,} test Source 1 businesses; output checkpoint every {PREDICT_CHUNK:,}.')
    try:
        with threadpool_limits(limits=4):
            searcher.faiss.omp_set_num_threads(min(4,os.cpu_count() or 2))
            for frame in pd.read_csv(data/'test/test_source1.tsv',sep='\t',dtype=str,
                                      keep_default_na=False,usecols=core.COLS,chunksize=PREDICT_CHUNK):
                frame=frame[core.COLS].reset_index(drop=True)
                path=shards/f'{start:010d}'
                if path.exists():
                    stats=json.loads((path/'stats.json').read_text())
                    assert stats['start']==start and stats['rows']==len(frame)
                    assert stats['first']==frame.entity_id.iloc[0] and stats['last']==frame.entity_id.iloc[-1]
                    assert all((path/n).exists() for n in ['candidate_pairs.tsv','matching_results.tsv'])
                else:
                    if shutil.disk_usage(work).free<1.5*2**30:raise RuntimeError('Low disk. Completed shards saved; free space before rerunning.')
                    q=encoder.encode(frame)
                    ids,scores=searcher.search(frame,q)
                    records=searcher.fetch(ids)
                    cand,accepted=score_candidates(frame,ids,scores,records,bundle)
                    stats=write_shard(path,frame,cand,accepted,start)
                    new+=len(frame)
                    del records,q,ids,scores,cand,accepted
                    if new==len(frame) or (start+len(frame))%10000==0:
                        rate=new/max(time.time()-started,1)
                        eta=(total_s1-start-len(frame))/max(rate,1)/60
                        core.log(f'{start+len(frame):,}/{total_s1:,} businesses; {rate:.1f}/sec; remaining prediction ~{eta:.1f} min')
                for k in ['rows','candidate_pairs','predicted_links','empty_candidates','empty_matches']:
                    counts[k]=counts.get(k,0)+stats[k]
                start+=len(frame)
    finally:searcher.close();gc.collect()
    assert start==total_s1
    core.atomic_json(work/'prediction_complete.json',counts)
    core.log('All test businesses predicted: '+json.dumps(counts))
    return counts

def merge_outputs(data,work):
    counts=json.loads((work/'prediction_complete.json').read_text())
    out=work/'output';out.mkdir(exist_ok=True)
    paths=sorted(p for p in (work/'shards').iterdir() if p.is_dir() and p.name.isdigit())
    expected=0;hist=np.zeros(21,np.int64)
    for p in paths:
        s=json.loads((p/'stats.json').read_text())
        assert s['start']==expected
        expected+=s['rows'];hist+=s['candidate_histogram']
    assert expected==counts['rows']
    for filename,second in [('matching_results.tsv','matched_entity_ids'),('candidate_pairs.tsv','candidate_entity_ids')]:
        path=out/filename;tmp=Path(str(path)+'.tmp');rows=0
        with tmp.open('wb') as f:
            header=f'source1_entity_id\t{second}\n'.encode()
            f.write(header)
            for p in paths:
                with (p/filename).open('rb') as src:
                    assert src.readline()==header
                    for line in src:
                        f.write(line);rows+=1
        assert rows==expected
        os.replace(tmp,path)
    counts['mean_candidates']=counts['candidate_pairs']/counts['rows']
    counts['candidate_histogram']=hist.tolist()
    core.atomic_json(out/'submission_summary.json',counts)
    core.log('Merged submission files: '+json.dumps(counts))
    return out

def validate(data,work):
    validator=data.parent/'utils/validate_submission.py'
    out=work/'output'
    command=[sys.executable,str(validator),'--matching',str(out/'matching_results.tsv'),
             '--candidate',str(out/'candidate_pairs.tsv'),'--test-dir',str(data/'test'),'--check-ids']
    core.log('Running official validator, including test ID existence checks...')
    with (out/'validation.log').open('w') as log:
        proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in proc.stdout:
            print(line,end='',flush=True);log.write(line);log.flush()
        code=proc.wait()
    if code:raise RuntimeError(f'Official validator failed (exit {code}). Do not submit. See validation.log.')
    core.atomic_json(out/'validated.json',{'matching_sha256':sha256(out/'matching_results.tsv'),
               'candidate_sha256':sha256(out/'candidate_pairs.tsv'),'check_ids':True})
    core.log('VALIDATOR PASSED. Portal upload: '+str(out/'matching_results.tsv'))

def package(work,meta):
    from importlib.metadata import version
    out=work/'output';assert (out/'validated.json').exists()
    validation=json.loads((out/'validated.json').read_text())
    assert validation['matching_sha256']==sha256(out/'matching_results.tsv')
    assert validation['candidate_sha256']==sha256(out/'candidate_pairs.tsv')
    (work/'requirements.txt').write_text('\n'.join(f'{p}=={version(p)}' for p in
             ['torch','transformers','numpy','pandas','scikit-learn','faiss-cpu','safetensors','sentencepiece'])+'\n')
    (work/'README.md').write_text('''# Embedding submission v2
Upload `output/matching_results.tsv` to the leaderboard portal.
Retain `output/candidate_pairs.tsv`, the model, code, and methodology for final review.
The candidate file contains exactly every final pair evaluated by the classifier.

## Reproduce predictions
Install requirements in a compatible GPU environment. Public E5 weights are pinned by revision.
Run: `python submit_pipeline.py --data /path/to/student_resource/dataset --model-dir ./model --work ./rerun --stage all`
The input dataset and official validator must be provided separately. No business lookups occur.
All test S1 entities receive one row, including empty predictions. Country matching is enforced.

## Reproduce training
Run `training_experiment.ipynb` on the supplied training dataset to recreate the sampled
training experiment, matcher and cutoff. Its dataset and working paths are configurable.
The trained matcher in `model/matcher.pkl` comes from that experiment, not test labels.
The upgrade training notebook requires the v1 training caches as a read-only input.
Only load your own trusted pickle models.

## Checkpoints
This archive excludes large vector and index caches. The complete saved Kaggle output
contains those, plus completed prediction shards. Rerunning against an unchanged working
folder resumes completed work. A discarded unsaved session cannot be resumed.

## Final package
This archive preserves submission evidence and runnable code. Before final challenge
closure, transfer methodology into the organizer's Documentation_template.md and check
any organizer-specific archive naming or additional requirements.
''')
    (work/'methodology.md').write_text(f'''# Methodology — embedding submission v2
Names and addresses are NFKC/whitespace normalized, preserving Unicode. E5 encodes
`query: business name: NAME; address: ADDRESS` with attention-masked mean pooling and
L2 normalization, maximum 256 tokens. Model: {core.MODEL_ID}; revision: {core.REVISION}.
Pretrained weights are fixed, not fine-tuned. Business data comes only from the challenge.

Test S2/S3 vectors form independent country-specific IVF-PQ indexes (up to 2048 cells,
48 subquantizers, 8 bits each). Quantizers use up to 110000 unlabeled target vectors
per country, seed 2026. Search probes {meta['ann_nprobe']} cells, retrieves up to {meta['ann_overfetch']} approximate candidates,
then reranks by cosine against stored float16 vectors and keeps at most 20.
Only these final 20 are passed to the matcher and written to candidate_pairs.tsv.
Intermediate retrieval counts and code are disclosed here for auditability.

The HistGradientBoostingClassifier uses the 18 lexical features defined in embedding_core.py
and embedding cosine similarity. It was trained on 30000 sampled S1 businesses (up to 600000 pairs),
with 2000 tuning and 2000 fresh evaluation businesses. Threshold: {meta['threshold']}.
Training targets include the entire provided train S2/S3 pool. No truth matches are injected.
Local evaluation macro F0.5: {meta['evaluation']['macro_F05']:.6f}; this is not a test leaderboard score.
Test predictions use the saved model and cutoff without retraining or threshold changes.

Source 1 is processed in batches and files are assembled in input order. The official
validator runs with --check-ids. Empty match lists are retained. France is present in test
but absent from labeled training; country-specific French accuracy has not been validated.
The country index code derives countries dynamically, including France.

See output/submission_summary.json for actual candidate counts, and requirements.txt for versions.
''')
    # A small ZIP is easy to download even if the browser previews TSV files.
    with zipfile.ZipFile(out/'matching_results_download.zip','w',zipfile.ZIP_DEFLATED) as z:
        z.write(out/'matching_results.tsv','matching_results.tsv')
    archive=work.parent/'embedding_submission_v2_backup.zip'
    members=['README.md','methodology.md','requirements.txt','submission_config.json',
      'embedding_core.py','submit_pipeline.py','training_experiment.ipynb',
      'model/matcher.pkl','model/experiment_results.json','output/matching_results.tsv',
      'output/candidate_pairs.tsv','output/submission_summary.json','output/validation.log','output/validated.json']
    for name in members:
        assert (work/name).exists(),f'Missing backup member: {name}'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in members:z.write(work/name,name)
    core.log('Downloadable matches ZIP: '+str(out/'matching_results_download.zip'))
    core.log('Retain backup: '+str(archive))
    return archive

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--stage',choices=['all','encode','index','predict','finalize'],default='all')
    parser.add_argument('--batch-size',type=int,default=64)
    args=parser.parse_args();data,work=args.data,args.work
    bundle,meta,config=initialize(data,work,args.model_dir)
    if args.stage in ('all','encode'):
        encoder=core.Encoder(args.batch_size)
        core.encode_pool(data,work,encoder,split='test')
        del encoder;gc.collect()
        import torch;torch.cuda.empty_cache()
    if args.stage in ('all','index'):core.build_indexes(work)
    if args.stage in ('all','predict'):
        encoder=core.Encoder(args.batch_size)
        predict(data,work,bundle,encoder)
        del encoder;gc.collect()
        import torch;torch.cuda.empty_cache()
    if args.stage in ('all','finalize'):
        merge_outputs(data,work);validate(data,work);package(work,meta)

if __name__=='__main__':main()
