from pathlib import Path
import ast,json
ROOT=Path(__file__).parent
old=Path('/workspace/scratch/bd62f3e52ed8/initial_submission/baseline.py').read_text()
tree=ast.parse(old)
names={'norm','canonical_address','block_keys','key_hash','BlockIndex'}
parts=[]
for node in tree.body:
 if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names:
  parts.append(ast.get_source_segment(old,node))
legacy='\n\n'.join(parts)
helpers='''import hashlib
import unicodedata
import re
import time
import numpy as np
import pandas as pd
COLS = ['entity_id', 'business_name', 'business_address', 'country']
LEGAL = set('inc incorporated llc llp ltd limited pvt private corp corporation company co plc sarl sas eurl'.split())
ADDR_ALIAS = {'road':'rd', 'street':'st', 'avenue':'ave', 'drive':'dr',
              'lane':'ln', 'boulevard':'blvd', 'apartment':'apt', 'suite':'ste'}
def log(s):
    print(s, flush=True)
'''+legacy+'''

def read_chunks(path, cols=COLS):
    return pd.read_csv(path,sep="\\t",dtype=str,keep_default_na=False,
                       usecols=cols,chunksize=100000)

def reproduce_reference_sample(data):
    # Same random sampling/split as the submitted baseline, independent of
    # old notebook memory. No old model or unsafe pickle load is needed.
    from sklearn.model_selection import train_test_split
    rng=np.random.default_rng(2026)
    sample=pd.DataFrame()
    for chunk in read_chunks(data/"train/train_source1.tsv"):
        chunk=chunk[COLS].copy()
        chunk["_priority"]=rng.random(len(chunk))
        sample=pd.concat([sample,chunk],ignore_index=True).nsmallest(4000,"_priority")
    sample=sample.drop(columns="_priority").reset_index(drop=True)
    assert len(sample)==4000 and sample.entity_id.is_unique
    ids=set(sample.entity_id)
    truth={}
    for chunk in read_chunks(data/"train/train_ground_truth.tsv",
                            ["source1_entity_id","matched_entity_ids"]):
        for row in chunk[chunk.source1_entity_id.isin(ids)].itertuples(index=False):
            truth[row.source1_entity_id]=set(x.strip() for x in row.matched_entity_ids.split(',') if x.strip())
    assert set(truth)==ids
    strata=sample.country+'|'+sample.entity_id.map(lambda x:str(not truth[x]))
    train_idx,held=train_test_split(np.arange(4000),test_size=.25,random_state=2026,stratify=strata)
    tune_idx,eval_idx=train_test_split(held,test_size=.5,random_state=2027,stratify=strata.iloc[held])
    sample['split']='train'
    sample.loc[tune_idx,'split']='tune'
    sample.loc[eval_idx,'split']='evaluation'
    return sample,truth

def prepare_pool(sample,truth,data,query_limit,random_per_source):
    index=BlockIndex(sample,max_block=30)
    selected=sample[sample.split=='evaluation'].sample(n=query_limit,random_state=7)
    q_original=selected.index.to_numpy()
    q_map={int(original):i for i,original in enumerate(q_original)}
    queries=selected.reset_index(drop=True)[COLS]
    q_truth={sid:truth[sid] for sid in queries.entity_id}
    required=set().union(*q_truth.values())
    exact=[set() for _ in range(len(queries))]
    retained=[]
    for source in [2,3]:
        rng=np.random.default_rng(180+source)
        extras=pd.DataFrame()
        scanned=0
        for chunk in read_chunks(data/f'train/train_source{source}.tsv'):
            chunk=chunk[COLS].reset_index(drop=True)
            rows=list(chunk.itertuples(index=False,name=None))
            matches=index.batch_candidates(rows)
            keep=chunk.entity_id.isin(required).to_numpy()
            for j,s1_rows in enumerate(matches):
                for original in s1_rows:
                    if original in q_map and norm(rows[j][3]) == norm(index.rows[original][3]):
                        i=q_map[original]
                        exact[i].add(rows[j][0])
                        keep[j]=True
            if keep.any():
                retained.append(chunk.loc[keep].copy())
            random_rows=chunk.loc[~keep].copy()
            random_rows['_priority']=rng.random(len(random_rows))
            extras=pd.concat([extras,random_rows],ignore_index=True).nsmallest(random_per_source,'_priority')
            scanned+=len(chunk)
            if scanned%1000000==0:
                log(f'Source {source}: scanned {scanned:,} rows')
        retained.append(extras.drop(columns='_priority'))
    pool=pd.concat(retained,ignore_index=True).drop_duplicates('entity_id').sort_values('entity_id').reset_index(drop=True)
    assert required<=set(pool.entity_id), 'Ground-truth target IDs missing'
    assert all(values<=set(pool.entity_id) for values in exact)
    return queries,q_truth,pool,exact

def clean_embedding_text(s):
    # Preserve accents and combining marks; do not use the legacy regex norm.
    return ' '.join(unicodedata.normalize('NFKC',str(s)).split())

def text_view(frame,view):
    if view=='name':
        return ['query: '+clean_embedding_text(x) for x in frame.business_name]
    return [
        'query: business name: '+clean_embedding_text(n)+'; address: '+clean_embedding_text(a)
        for n,a in zip(frame.business_name,frame.business_address)
    ]

def same_country_topk(query_vectors,pool_vectors,queries,pool,k=50,device="cuda"):
    # Exact GPU search for this small controlled pilot only. Full-pool
    # deployment needs ANN indexing; this is not the proposed production search.
    import torch
    results=[None]*len(queries)
    qc=queries.country.map(norm).to_numpy()
    pc=pool.country.map(norm).to_numpy()
    for country in sorted(set(qc)):
        qi=np.flatnonzero(qc==country)
        pi=np.flatnonzero(pc==country)
        if not len(pi):
            for i in qi: results[i]=[]
            continue
        p=torch.tensor(np.asarray(pool_vectors[pi]),device=device,dtype=torch.float32)
        for start in range(0,len(qi),32):
            ii=qi[start:start+32]
            q=torch.tensor(np.asarray(query_vectors[ii]),device=device,dtype=torch.float32)
            scores=q@p.T
            positions=torch.topk(scores,k=min(k,len(pi)),dim=1).indices.cpu().numpy()
            for i,local in zip(ii,positions):
                results[i]=pi[local].tolist()
        del p
    return results

def rrf(lists,limit=50):
    scores={}
    for values in lists:
        for rank,p in enumerate(values,1):
            scores[p]=scores.get(p,0.)+1/(60+rank)
    return sorted(scores,key=lambda p:(-scores[p],p))[:limit]

def hybrid(exact,embedding,limit):
    # Fixed half-budget per route, then fill from the embedding ranking first.
    out=[]; seen=set()
    for values in [exact[:limit//2],embedding[:limit-limit//2],embedding,exact]:
        for p in values:
            if p not in seen:
                out.append(p);seen.add(p)
                if len(out)==limit:return out
    return out

def lexical_rank(query,pool,positions):
    n=norm(query.business_name)
    a=canonical_address(query.business_address)
    def tri(s):return set(s[i:i+3] for i in range(max(0,len(s)-2)))
    ng,ag=tri(n),tri(a)
    def jac(x,y):return len(x&y)/len(x|y) if x or y else 0.
    scored=[]
    for p in positions:
        row=pool.iloc[p]
        m,b=norm(row.business_name),canonical_address(row.business_address)
        ns,ads=jac(ng,tri(m)),jac(ag,tri(b))
        score=max(ns,ads)+0.25*min(ns,ads)
        scored.append((-score,p))
    return [p for _,p in sorted(scored)]

def report_candidates(method,lists,queries,pool,truth,baseline,k=None):
    id_array=pool.entity_id.to_numpy()
    counts=[]; found=0; total=0; recovered=0; macro=[]; all_recovered=0; nonempty=0
    singleton=0
    details=[]
    for i,row in enumerate(queries.itertuples(index=False)):
        ids=set(id_array[lists[i]]) if len(lists[i]) else set()
        answer=truth[row.entity_id]
        hit=ids&answer
        found+=len(hit);total+=len(answer);counts.append(len(ids))
        recovered+=len(hit-baseline[i])
        if answer:
            macro.append(len(hit)/len(answer));nonempty+=1
            all_recovered+=int(answer<=ids)
        else:singleton+=1
        details.append({'method':method,'k':k,'source1_id':row.entity_id,
                        'candidate_count':len(ids),'true_count':len(answer),
                        'found':len(hit),'missed_ids':','.join(sorted(answer-ids)),
                        'candidate_ids':','.join(sorted(ids))})
    return {'method':method,'k':k,'found':found,'true_links':total,
            'link_recall':found/max(1,total),'macro_recall_non_singletons':float(np.mean(macro)) if macro else None,
            'mean_candidates':float(np.mean(counts)),'p95_candidates':float(np.percentile(counts,95)),
            'max_candidates':max(counts),'recovered_over_exact':recovered,
            'all_matches_retrieved_businesses':all_recovered,'non_singletons':nonempty,
            'singleton_businesses':singleton},details
'''
(ROOT/'helpers.py').write_text(helpers)
cells=[]
def md(s):cells.append({'cell_type':'markdown','metadata':{},'source':s.splitlines(keepends=True)})
def code(s):cells.append({'cell_type':'code','metadata':{},'source':s.splitlines(keepends=True),'outputs':[],'execution_count':None})
md('''# Embedding candidate-search experiment — pilot 1

**Question:** Can multilingual embeddings recover correct matches missed by exact
blocking, at a small fixed candidate budget?

Keep the submitted **0.756** baseline and its backup. This notebook does not modify
that submission, train a matching classifier, or generate a new leaderboard file.

Use a **new private Kaggle notebook**, attach the same challenge dataset, enable a
GPU and Internet for package/model downloads, and use **Save & Run All** to retain
outputs. No business text is sent to an embedding API. Inference runs locally on
the notebook GPU using downloaded model weights.

Model: `intfloat/multilingual-e5-small`, whose official model card declares MIT and
384-dimensional embeddings. The code records the exact model revision, checks the
license, and verifies parameter count. This satisfies the stated license/size
conditions for this component, not an independent organizer approval of pretrained
models. No external business lookup, geocoding, or enrichment is used.

Sources: https://huggingface.co/intfloat/multilingual-e5-small
and its README: https://huggingface.co/intfloat/multilingual-e5-small/raw/main/README.md

### Controlled experiment
- Reproduce the 4,000-business sample and splits from the initial notebook.
- Select 200 of its evaluation businesses (development reuse, not a fresh holdout).
- Include all their true matches, **all their exact-key candidates from the full
  training pool**, and 20,000 additional random records from each source.
- Search this SAME pool using exact keys, name embeddings, combined name/address
  embeddings, and fixed-budget hybrid retrieval. Country restriction applies to all.
- Report candidate recall and sizes at K=10,20,50, plus exact-key unbounded reference.

The pool is deliberately label-constructed for a feasibility test; labels are
never input to encoding/ranking. Results are optimistic relative to 10 million
records and are NOT directly comparable with 69.6% full-pool recall except the
exact-key reference on these particular queries. We include the baseline's actual
lookalikes to make this harder than a random-only pool. Learned embeddings are
frozen: there is no fine-tuning yet. France is not validated by US/India training.

This pilot uses exact cosine top-K on GPU to separate representation quality from
approximate-index errors. Production deployment would require ANN and a full-pool
benchmark before another submission; do not scale this exact search unchanged.
''')
code('''# Install only if the required modules are missing; keep Kaggle's existing Torch.
import importlib.util, subprocess, sys
packages = {"transformers":"transformers>=4.40,<5", "sentencepiece":"sentencepiece",
            "safetensors":"safetensors", "huggingface_hub":"huggingface_hub"}
missing = [requirement for module,requirement in packages.items()
           if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable,"-m","pip","install","-q",*missing])

from pathlib import Path
import json, os, hashlib, time
import numpy as np
import pandas as pd
import torch
from IPython.display import display, FileLink

DATA=Path("/kaggle/input/datasets/spoortimorabad/ml-challenge-dataset/student_resource/dataset")
OUT=Path("/kaggle/working/embedding_pilot_v1")
OUT.mkdir(parents=True,exist_ok=True)
QUERY_LIMIT=200
RANDOM_PER_SOURCE=20000
MODEL_ID="intfloat/multilingual-e5-small"
assert torch.cuda.is_available(), "Enable a GPU in notebook settings before running this experiment."
assert (DATA/"train/train_source1.tsv").exists(), "Correct the DATA path."
print("GPU:",torch.cuda.get_device_name(0))
''')
md('## Helper functions\nNo edits needed. Exact-key generation is copied from the submitted baseline; embedding text preserves Unicode combining marks.\n')
code(helpers)
md('''## Build the pilot pool
This scans the training files once. It reproduces the baseline sample without
requiring old notebook variables or unsafe pickle loading. Complete pool files
are cached; an interrupted pool scan restarts, but embedding batches later resume.
''')
code('''config={"query_limit":QUERY_LIMIT,"random_per_source":RANDOM_PER_SOURCE,"model":MODEL_ID,
        "data":str(DATA.resolve()),"version":1,
        "input_sizes":{p.name:p.stat().st_size for p in (DATA/"train").glob("*.tsv")}}
config_path=OUT/"config.json"
if config_path.exists():
    assert json.loads(config_path.read_text())==config, "Configuration changed: use a new OUT directory."
else:
    config_path.write_text(json.dumps(config,indent=2))

if (OUT/"pool_complete.json").exists():
    sample=pd.read_csv(OUT/"reference_sample.tsv",sep="\\t",dtype=str,keep_default_na=False)
    queries=pd.read_csv(OUT/"queries.tsv",sep="\\t",dtype=str,keep_default_na=False)
    pool=pd.read_csv(OUT/"pool.tsv.gz",sep="\\t",dtype=str,keep_default_na=False)
    q_truth={k:set(v) for k,v in json.loads((OUT/"truth.json").read_text()).items()}
    exact=[set(v) for v in json.loads((OUT/"exact_candidates.json").read_text())]
else:
    sample,truth=reproduce_reference_sample(DATA)
    queries,q_truth,pool,exact=prepare_pool(sample,truth,DATA,QUERY_LIMIT,RANDOM_PER_SOURCE)
    sample.to_csv(OUT/"reference_sample.tsv",sep="\\t",index=False)
    queries.to_csv(OUT/"queries.tsv",sep="\\t",index=False)
    pool.to_csv(OUT/"pool.tsv.gz",sep="\\t",index=False,compression="gzip")
    (OUT/"truth.json").write_text(json.dumps({k:sorted(v) for k,v in q_truth.items()}))
    (OUT/"exact_candidates.json").write_text(json.dumps([sorted(v) for v in exact]))
    (OUT/"pool_complete.json").write_text(json.dumps({"queries":len(queries),"pool":len(pool)}))

print("Queries:",len(queries),"Pool:",len(pool),"True links:",sum(map(len,q_truth.values())))
print("Exact-key average candidates:",np.mean([len(x) for x in exact]))
print("Countries:",queries.country.value_counts().to_dict())
assert len(pool)<=400000, "Pool is larger than planned; reduce QUERY_LIMIT in a new OUT directory."
''')
md('''## Download and verify the frozen model
The model card recommends the `query: ` prefix on both sides for symmetric
similarity tasks. We follow that convention. Cosine similarity is a ranking score,
not a calibrated probability of identity. No old 0.50 threshold is used here.
''')
code('''from huggingface_hub import HfApi
from transformers import AutoTokenizer, AutoModel
import importlib.metadata

meta_path=OUT/"model_info.json"
prior=json.loads(meta_path.read_text()) if meta_path.exists() else None
info=HfApi().model_info(MODEL_ID,revision=prior["revision"] if prior else None)
license_name=info.card_data.get("license") if info.card_data else None
assert license_name in {"mit","apache-2.0"}, f"Check model license: {license_name}"
revision=info.sha

tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=revision,trust_remote_code=False)
encoder=AutoModel.from_pretrained(MODEL_ID,revision=revision,
                                  trust_remote_code=False,use_safetensors=True)
parameters=sum(p.numel() for p in encoder.parameters())
assert parameters<=8_000_000_000
encoder=encoder.eval().to("cuda").half()
torch.set_num_threads(4)
meta={"model":MODEL_ID,"revision":revision,"license":license_name,
      "parameters":parameters,"dimensions":encoder.config.hidden_size,"max_length":256,
      "prefix":"query: on both sides","pooling":"attention-mask mean; L2 normalization",
      "versions":{p:importlib.metadata.version(p) for p in ["torch","transformers","numpy","pandas"]}}
meta_path.write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2))
''')
md('''## Encode name and combined name/address views
Embeddings are saved every 2,048 records and resume from completed chunks. Records
longer than 256 tokenizer tokens are truncated in this pilot. Batch size is 64;
if GPU memory is insufficient, set `BATCH_SIZE=32` and rerun this cell.
''')
code('''BATCH_SIZE=64

def encode_cached(texts,tag):
    h=hashlib.sha256(json.dumps(meta,sort_keys=True).encode())
    for text in texts:
        h.update(text.encode("utf-8"));h.update(b"\\0")
    signature=h.hexdigest()
    target=OUT/f"{tag}.npy"
    progress_path=OUT/f"{tag}_progress.json"
    n,d=len(texts),encoder.config.hidden_size
    completed=0
    if progress_path.exists():
        saved=json.loads(progress_path.read_text())
        assert saved["signature"]==signature,"Embedding inputs changed: use new OUT."
        completed=saved["done"]
        assert target.exists()
        result=np.load(target,mmap_mode="r+")
        assert result.shape==(n,d)
    else:
        result=np.lib.format.open_memmap(target,mode="w+",dtype=np.float32,shape=(n,d))
    started=time.time()
    for start in range(completed,n,2048):
        end=min(n,start+2048)
        for batch_start in range(start,end,BATCH_SIZE):
            batch_end=min(end,batch_start+BATCH_SIZE)
            batch=tokenizer(texts[batch_start:batch_end],padding=True,truncation=True,
                            max_length=256,return_tensors="pt").to("cuda")
            with torch.inference_mode():
                hidden=encoder(**batch).last_hidden_state.float()
                mask=batch["attention_mask"].unsqueeze(-1).float()
                vectors=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
                vectors=torch.nn.functional.normalize(vectors,p=2,dim=1)
            result[batch_start:batch_end]=vectors.cpu().numpy()
        result.flush()
        temp=progress_path.with_suffix(".tmp")
        temp.write_text(json.dumps({"signature":signature,"done":end}))
        os.replace(temp,progress_path)
        print(f"{tag}: {end:,}/{n:,} encoded; {(time.time()-started)/60:.1f} minutes",flush=True)
    return result

pool_name=encode_cached(text_view(pool,"name"),"pool_name")
query_name=encode_cached(text_view(queries,"name"),"query_name")
pool_combined=encode_cached(text_view(pool,"combined"),"pool_combined")
query_combined=encode_cached(text_view(queries,"combined"),"query_combined")
''')
md('''## Retrieve and compare at the same candidate budgets
`Exact keys (all)` reproduces the baseline's raw candidate sets for these queries.
`Exact keys + lexical top-K` ranks those candidates cheaply without labels.
`Embedding two views` combines name and full-record ranks. `Hybrid` allocates
roughly half the budget to each route, deduplicates, then fills remaining slots.

These are **candidate recall** measurements, not macro F0.5 or leaderboard scores.
Singletons stay in candidate-size statistics but are excluded from recall's
non-singleton denominator. An embedding search normally returns candidates for
singletons too; the future matching model must reject them.
''')
code('''started=time.time()
name_ranks=same_country_topk(query_name,pool_name,queries,pool,k=50)
combined_ranks=same_country_topk(query_combined,pool_combined,queries,pool,k=50)
embedding_ranks=[rrf([n,c],50) for n,c in zip(name_ranks,combined_ranks)]
id_to_pos={sid:i for i,sid in enumerate(pool.entity_id)}
exact_positions=[[id_to_pos[sid] for sid in sorted(values)] for values in exact]
exact_ranks=[lexical_rank(row,pool,positions) for row,positions in
             zip(queries.itertuples(index=False),exact_positions)]
print(f"Pilot search + lexical ranking: {time.time()-started:.1f} seconds")

summaries=[];details=[]
summary,rows=report_candidates("Exact keys (all)",exact_positions,queries,pool,q_truth,exact)
summaries.append(summary);details.extend(rows)

for k in [10,20,50]:
    methods={
        "Exact keys + lexical top-K":[r[:k] for r in exact_ranks],
        "Embedding name":[r[:k] for r in name_ranks],
        "Embedding combined":[r[:k] for r in combined_ranks],
        "Embedding two views":[r[:k] for r in embedding_ranks],
        "Hybrid":[hybrid(e,v,k) for e,v in zip(exact_ranks,embedding_ranks)],
    }
    for method,candidates in methods.items():
        assert all(len(x)<=k and len(x)==len(set(x)) for x in candidates)
        summary,rows=report_candidates(method,candidates,queries,pool,q_truth,exact,k)
        summaries.append(summary);details.extend(rows)

results=pd.DataFrame(summaries)
results.to_csv(OUT/"retrieval_results.csv",index=False)
pd.DataFrame(details).to_csv(OUT/"retrieval_details.tsv.gz",sep="\\t",index=False,compression="gzip")
display(results[["method","k","found","true_links","link_recall","mean_candidates",
                 "p95_candidates","max_candidates","recovered_over_exact"]])
''')
md('''## Inspect matches recovered by embeddings and save results
These examples help tell whether improvements involve typos, shortened addresses,
or different scripts. Do not infer typo robustness from aggregate recall alone.
Keep a new holdout for later model selection after inspecting these cases.
''')
code('''examples=[]
lookup=pool.set_index("entity_id")
for i,row in enumerate(queries.itertuples(index=False)):
    retrieved=set(pool.entity_id.iloc[embedding_ranks[i][:20]])
    recovered=(retrieved & q_truth[row.entity_id])-exact[i]
    for sid in sorted(recovered):
        other=lookup.loc[sid]
        examples.append({"source1_id":row.entity_id,"name":row.business_name,
                         "address":row.business_address,"recovered_id":sid,
                         "recovered_name":other.business_name,
                         "recovered_address":other.business_address})
examples=pd.DataFrame(examples)
examples.to_csv(OUT/"recovered_examples.tsv",sep="\\t",index=False)
with pd.option_context("display.max_colwidth",100):
    display(examples.head(15))

import zipfile
backup=Path("/kaggle/working/embedding_pilot_results.zip")
with zipfile.ZipFile(backup,"w",zipfile.ZIP_DEFLATED) as z:
    for name in ["config.json","model_info.json","retrieval_results.csv","retrieval_details.tsv.gz",
                 "recovered_examples.tsv","queries.tsv","reference_sample.tsv","truth.json",
                 "exact_candidates.json","pool.tsv.gz"]:
        z.write(OUT/name,name)
print("Download the results backup; retain the saved notebook outputs for embedding caches.")
display(FileLink(str(backup)))
print("Share retrieval_results.csv, model_info.json, and the encoding timings.")
''')
md('''### Next decision
If embeddings recover more true matches at the same candidate budget, keep the
0.756 baseline and proceed to a larger/full-pool ANN test, then retrain matching
on the new candidates. If not, inspect errors and compare character-based
retrieval before spending GPU time on millions of embeddings. No automatic
production replacement is made by this notebook.

The experiment pool construction and metric logic were checked locally with
synthetic records. GPU encoding and GPU search were not executed locally. The pretrained encoder itself and real-data recall were not
run in the assistant environment; the Kaggle results are the evidence needed.
''')
for i,c in enumerate(cells):c['id']=f'emb-{i:02d}'
notebook={'cells':cells,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},
'language_info':{'name':'python'},'kaggle':{'isInternetEnabled':True,'language':'python','sourceType':'notebook'}},
'nbformat':4,'nbformat_minor':5}
(ROOT/'amazon_ml_embedding_pilot.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1))
for c in cells:
 if c['cell_type']=='code':ast.parse(''.join(c['source']))
print('Notebook written; all cells parse.')
