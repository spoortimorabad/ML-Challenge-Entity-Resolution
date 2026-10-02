import hashlib
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
def norm(s):
    s = unicodedata.normalize('NFKC', str(s)).casefold()
    return ' '.join(re.findall(r'[^\W_]+', s))

def canonical_address(s):
    return ' '.join(ADDR_ALIAS.get(t, t) for t in norm(s).split() if t not in {'null', 'none'})

def block_keys(name, address, country):
    """At most six deterministic keys. No labels or business lookups."""
    n = norm(name)
    a = canonical_address(address)
    country = norm(country)
    if not country:
        return ()
    core = sorted(set(t for t in n.split() if t not in LEGAL))
    keys = set()
    if len(''.join(core)) >= 5:
        keys.add(country + '|N|' + ''.join(core))
    if len(n.replace(' ', '')) >= 5:
        keys.add(country + '|R|' + n.replace(' ', ''))
    if len(a) >= 12 and len(a.split()) >= 3:
        keys.add(country + '|A|' + ' '.join(sorted(a.split())))
    # Combine one distinctive-looking name token with address numbers.
    words = sorted((t for t in core if len(t) >= 4), key=lambda t: (-len(t), t))
    numbers = sorted(set(re.findall(r'\d+[a-z]*', a)))[:3]
    if words:
        for number in numbers:
            keys.add(country + '|P|' + words[0] + '|' + number)
    return tuple(sorted(keys))

def key_hash(key):
    return int.from_bytes(hashlib.blake2b(key.encode('utf-8'), digest_size=8).digest(), 'little')

class BlockIndex:
    """Compact sorted key hashes; skip keys shared by too many Source 1 rows."""
    def __init__(self, source1, max_block=30):
        self.source1 = source1
        self.max_block = max_block
        hashes, owners = [], []
        for i, row in enumerate(source1.itertuples(index=False)):
            keys = block_keys(row.business_name, row.business_address, row.country)
            for h in sorted(set(map(key_hash, keys))):
                hashes.append(h)
                owners.append(i)
        h = np.asarray(hashes, dtype=np.uint64)
        o = np.asarray(owners, dtype=np.int32)
        del hashes, owners
        order = np.argsort(h, kind='stable')
        h, o = h[order], o[order]
        unique, start, counts = np.unique(h, return_index=True, return_counts=True)
        allowed = counts <= max_block
        keep = np.repeat(allowed, counts)
        self.hashes, self.owners = h[keep], o[keep]
        self.rows = list(source1[COLS].itertuples(index=False, name=None))
        log(f'Block index: {len(source1):,} Source 1 rows; '
            f'{len(self.hashes):,} key entries; {int((~allowed).sum()):,} broad keys skipped.')

    def batch_candidates(self, others):
        q_hashes, q_rows = [], []
        for j, (_, name, address, country) in enumerate(others):
            for h in sorted(set(map(key_hash, block_keys(name, address, country)))):
                q_hashes.append(h)
                q_rows.append(j)
        out = [set() for _ in others]
        if not q_hashes or not len(self.hashes):
            return out
        q = np.asarray(q_hashes, dtype=np.uint64)
        left = np.searchsorted(self.hashes, q, side='left')
        right = np.searchsorted(self.hashes, q, side='right')
        for k in np.flatnonzero(right > left):
            target = out[q_rows[k]]
            target.update(map(int, self.owners[left[k]:right[k]]))
        return out

def read_chunks(path, cols=COLS):
    return pd.read_csv(path,sep="\t",dtype=str,keep_default_na=False,
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
