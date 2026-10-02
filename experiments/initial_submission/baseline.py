"""Initial entity-resolution submission baseline. No external business data.

Index Source 1 block keys; stream Sources 2/3 through the index. This reverses
lookup direction for efficiency but generates the same kind of S1 -> S2/S3 pairs.
All final candidates are scored; all scored candidates are exported.
"""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import json
import os
import pickle
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

VERSION = 'initial-block-baseline-v1'
COLS = ['entity_id', 'business_name', 'business_address', 'country']
LEGAL = set('inc incorporated llc llp ltd limited pvt private corp corporation company co plc sarl sas eurl'.split())
ADDR_ALIAS = {'road':'rd', 'street':'st', 'avenue':'ave', 'drive':'dr',
              'lane':'ln', 'boulevard':'blvd', 'apartment':'apt', 'suite':'ste'}
FEATURES = ['name_exact', 'core_name_exact', 'name_trigram_jaccard',
 'name_token_jaccard', 'name_token_containment', 'name_length_ratio',
 'address_exact', 'address_trigram_jaccard', 'address_token_jaccard',
 'address_token_containment', 'address_missing', 'number_jaccard',
 'number_disjoint', 'unit_equal', 'unit_conflict', 'country_equal',
 'name_token_count', 'candidate_name_token_count']


def log(s):
    print(s, flush=True)


def read_chunks(path, size=25000, columns=COLS):
    return pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False,
                       usecols=columns, chunksize=size)


def norm(s):
    s = unicodedata.normalize('NFKC', str(s)).casefold()
    return ' '.join(re.findall(r'[^\W_]+', s))


def canonical_address(s):
    return ' '.join(ADDR_ALIAS.get(t, t) for t in norm(s).split() if t not in {'null', 'none'})


def grams(s):
    return frozenset(s[i:i+3] for i in range(max(0, len(s)-2)))


def unit_id(s):
    # Preserve alpha-numeric unit IDs (3B2 versus 3B3). Not a hard rejection rule.
    m = re.search(r'\b(?:unit|suite|ste|apt|apartment|flat)\s*(?:no\.?\s*)?[:#-]?\s*([\w]+)',
                  unicodedata.normalize('NFKC', s).casefold())
    return m.group(1) if m else ''


@lru_cache(maxsize=30000)
def prepared(name, address, country):
    n = norm(name)
    a = canonical_address(address)
    nt, at = frozenset(n.split()), frozenset(a.split())
    core = tuple(sorted(t for t in nt if t not in LEGAL))
    nums = frozenset(re.findall(r'\d+[a-z]*', a))
    return n, a, nt, at, core, nums, grams(n), grams(a), unit_id(address), norm(country)


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


def jaccard(a, b):
    return len(a & b) / len(a | b) if a or b else 0.0


def containment(a, b):
    return len(a & b) / min(len(a), len(b)) if a and b else 0.0


def features(left, right):
    # tuples: ID, business_name, business_address, country
    n,a,nt,at,c,nums,ng,ag,u,country = prepared(*left[1:])
    m,b,mt,bt,d,other_nums,mg,bg,v,other_country = prepared(*right[1:])
    return [
        float(bool(n) and n == m), float(bool(c) and c == d),
        jaccard(ng, mg), jaccard(nt, mt), containment(nt, mt),
        min(len(n),len(m)) / max(len(n),len(m),1),
        float(bool(a) and a == b), jaccard(ag,bg), jaccard(at,bt), containment(at,bt),
        float(not a or not b), jaccard(nums,other_nums),
        float(bool(nums) and bool(other_nums) and not nums & other_nums),
        float(bool(u) and bool(v) and u == v), float(bool(u) and bool(v) and u != v),
        float(bool(country) and country == other_country), len(nt), len(mt),
    ]


def atomic_pickle(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def signature(paths, extra):
    data = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    return hashlib.sha256(json.dumps([VERSION, data, extra], sort_keys=True).encode()).hexdigest()


def open_db(path, sig, training):
    db = sqlite3.connect(path)
    db.execute('PRAGMA cache_size=-65536')
    db.execute('CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)')
    prior = db.execute("SELECT v FROM meta WHERE k='signature'").fetchone()
    if prior and prior[0] != sig:
        db.close()
        raise RuntimeError('Inputs/configuration changed. Use a NEW work directory.')
    db.execute("INSERT OR IGNORE INTO meta VALUES ('signature', ?)", (sig,))
    db.execute('CREATE TABLE IF NOT EXISTS progress (file TEXT PRIMARY KEY, done INTEGER, complete INTEGER)')
    if training:
        db.execute('CREATE TABLE IF NOT EXISTS pairs '
                   '(s1 INTEGER, candidate TEXT, label INTEGER, x BLOB, '
                   'PRIMARY KEY(s1,candidate)) WITHOUT ROWID')
    else:
        db.execute('CREATE TABLE IF NOT EXISTS pairs '
                   '(s1 INTEGER, candidate TEXT, score REAL, '
                   'PRIMARY KEY(s1,candidate)) WITHOUT ROWID')
    db.commit()
    return db


def disk_check(directory):
    if shutil.disk_usage(directory).free < 2 * 1024**3:
        raise RuntimeError('Less than 2 GB free disk. Progress is saved. Free space or move to a larger runtime.')


def prepare_sample(data, work, sample_n, seed):
    path = work / 'sample.pkl'
    paths = [data/'train/train_source1.tsv', data/'train/train_ground_truth.tsv']
    sig = signature(paths, {'n': sample_n, 'seed': seed})
    if path.exists():
        with path.open('rb') as f:
            saved = pickle.load(f)
        if saved['signature'] != sig:
            raise RuntimeError('Sample configuration changed. Use a new work directory.')
        return saved['frame'], saved['truth']
    rng = np.random.default_rng(seed)
    sample = pd.DataFrame()
    for chunk in read_chunks(paths[0], size=100000):
        chunk = chunk[COLS].copy()
        chunk['_priority'] = rng.random(len(chunk))
        sample = pd.concat([sample,chunk], ignore_index=True).nsmallest(sample_n, '_priority')
    sample = sample.drop(columns='_priority').reset_index(drop=True)
    if len(sample) != sample_n or not sample.entity_id.is_unique:
        raise RuntimeError('Not enough unique Source 1 rows for configured sample.')
    ids = set(sample.entity_id)
    truth = {}
    for chunk in read_chunks(paths[1], columns=['source1_entity_id','matched_entity_ids']):
        for row in chunk[chunk.source1_entity_id.isin(ids)].itertuples(index=False):
            if row.source1_entity_id in truth:
                raise RuntimeError('Duplicate ground-truth row.')
            truth[row.source1_entity_id] = set(x.strip() for x in row.matched_entity_ids.split(',') if x.strip())
    if ids != set(truth):
        raise RuntimeError('Missing ground truth for sampled businesses.')
    # Verify labeled entities cannot bridge training and evaluation groups.
    seen = set()
    for values in truth.values():
        if seen & values:
            raise RuntimeError('A target links multiple sampled references. A grouped split is needed.')
        seen.update(values)
    strata = sample.country + '|' + sample.entity_id.map(lambda x: str(not truth[x]))
    train_idx, held_idx = train_test_split(np.arange(sample_n), test_size=0.25,
                                          random_state=seed, stratify=strata)
    tune_idx, eval_idx = train_test_split(held_idx, test_size=0.5, random_state=seed+1,
                                        stratify=strata.iloc[held_idx])
    sample['split'] = 'train'
    sample.loc[tune_idx,'split'] = 'tune'
    sample.loc[eval_idx,'split'] = 'evaluation'
    atomic_pickle(path, {'signature':sig, 'frame':sample, 'truth':truth})
    sample.to_csv(work/'sample_source1.tsv', sep='\t', index=False)
    log('Sample split: ' + str(sample.split.value_counts().to_dict()))
    return sample, truth


def scan_sources(data, split, index, db, work, truth=None, model=None, limit=None):
    """Every generated candidate is featurized and, at test time, scored.
    SQLite stores each pair once and commits progress atomically per input batch.
    """
    training = truth is not None
    for source in [2,3]:
        file = data/split/f'{split}_source{source}.tsv'
        progress = db.execute('SELECT done,complete FROM progress WHERE file=?', (file.name,)).fetchone()
        resume = progress[0] if progress else 0
        if progress and progress[1]:
            log(file.name + ': already complete.')
            continue
        started = time.time()
        seen = processed = pairs_new = 0
        log(f'{file.name}: resuming at input row {resume:,}')
        for chunk in read_chunks(file, size=10000):
            end = seen + len(chunk)
            if end <= resume:
                seen = end
                continue
            chunk = chunk.iloc[max(0,resume-seen):][COLS]
            disk_check(work)
            others = list(chunk.itertuples(index=False,name=None))
            candidates = index.batch_candidates(others)
            pending = []
            meta = []
            xs = []

            def flush():
                if not meta:
                    return
                matrix = np.asarray(xs, dtype=np.float32)
                if training:
                    pending.extend((i,c,label,sqlite3.Binary(x.tobytes()))
                                   for (i,c,label),x in zip(meta,matrix))
                else:
                    with threadpool_limits(limits=4):
                        scores = model.predict_proba(matrix)[:,1]
                    pending.extend((i,c,float(score)) for (i,c),score in zip(meta,scores))
                meta.clear()
                xs.clear()

            for record, matches in zip(others,candidates):
                for i in sorted(matches):
                    left = index.rows[i]
                    # Hash equality only generates a candidate; country must agree.
                    if norm(left[3]) != norm(record[3]):
                        continue
                    xs.append(features(left,record))
                    if training:
                        meta.append((i,record[0],int(record[0] in truth[left[0]])))
                    else:
                        meta.append((i,record[0]))
                    if len(meta) >= 10000:
                        flush()
            flush()
            with db:
                if training:
                    db.executemany('INSERT INTO pairs VALUES (?,?,?,?)',pending)
                else:
                    db.executemany('INSERT INTO pairs VALUES (?,?,?)',pending)
                db.execute('INSERT OR REPLACE INTO progress VALUES (?,?,0)',(file.name,end))
            processed += len(others)
            pairs_new += len(pending)
            seen = end
            if processed % 100000 == 0 or processed == len(others):
                elapsed = max(time.time()-started,1e-6)
                log(f'  {seen:,} input rows; {pairs_new:,} new pairs; '
                    f'{processed/elapsed:,.0f} input rows/sec; {elapsed/60:.1f} min')
            if training and pairs_new > 2000000:
                raise RuntimeError('More than 2M new pairs in this scan. Use a smaller sample in a new work directory.')
            if limit is not None and processed >= limit:
                log('Preview stopped at a saved batch boundary. Full predict resumes here.')
                return
        with db:
            db.execute('INSERT OR REPLACE INTO progress VALUES (?,?,1)',(file.name,seen))
    log(f'{split} scan complete.')


def macro_score(sids, candidate_sids, labels, scores, truth_counts, threshold):
    selected = scores >= threshold
    pred = np.bincount(candidate_sids[selected], minlength=len(truth_counts))
    tp = np.bincount(candidate_sids[selected], weights=labels[selected], minlength=len(truth_counts))
    fp = pred - tp
    fn = truth_counts - tp
    values = np.zeros(len(truth_counts), dtype=float)
    single = truth_counts == 0
    values[single] = (pred[single] == 0)
    non = ~single
    values[non] = 1.25*tp[non]/(1.25*tp[non]+fp[non]+0.25*fn[non])
    return {'macro_F0.5':float(values[sids].mean()),
            'correct_accepted':int(tp[sids].sum()), 'incorrect_accepted':int(fp[sids].sum()),
            'missed_true_links':int(fn[sids].sum()),
            'singletons_correct':int((single[sids] & (pred[sids]==0)).sum()),
            'singletons':int(single[sids].sum())}


def train(data, work, sample_n=4000, seed=2026, max_block=30):
    sample, truth = prepare_sample(data,work,sample_n,seed)
    paths = [data/f'train/train_source{x}.tsv' for x in [2,3]] + [work/'sample.pkl']
    sig = signature(paths, {'max_block':max_block,'features':FEATURES})
    model_path = work/'model.pkl'
    if model_path.exists():
        with model_path.open('rb') as f:
            existing = pickle.load(f)
        if existing.get('signature') != sig:
            raise RuntimeError('Saved model inputs changed. Use a new work directory.')
        log('Saved training model is complete; reusing it.')
        log(json.dumps(existing['report'],indent=2))
        return
    db = open_db(work/'train_pairs.sqlite',sig,training=True)
    index = BlockIndex(sample,max_block)
    scan_sources(data,'train',index,db,work,truth=truth)
    count = db.execute('SELECT COUNT(*) FROM pairs').fetchone()[0]
    if count > 3000000:
        raise RuntimeError('Over 3M training pairs; reduce sample size in a new work directory.')
    if not count:
        raise RuntimeError('No candidates found; inspect input data and blocking.')
    X = np.empty((count,len(FEATURES)),dtype=np.float32)
    y = np.empty(count,dtype=np.int8)
    owners = np.empty(count,dtype=np.int32)
    for j,(i,label,blob) in enumerate(db.execute('SELECT s1,label,x FROM pairs ORDER BY s1,candidate')):
        X[j] = np.frombuffer(blob,dtype=np.float32)
        y[j], owners[j] = label,i
    db.close()
    splits = sample.split.to_numpy()
    train_mask = splits[owners] == 'train'
    if len(np.unique(y[train_mask])) != 2:
        raise RuntimeError('Training candidates need both matches and nonmatches.')
    log(f'Training on {int(train_mask.sum()):,} pairs, including {int(y[train_mask].sum()):,} positives.')
    model = HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,
        learning_rate=0.08,l2_regularization=1.0,early_stopping=False,random_state=seed)
    with threadpool_limits(limits=4):
        model.fit(X[train_mask],y[train_mask])
        scores = model.predict_proba(X)[:,1]
    truth_counts = np.array([len(truth[x]) for x in sample.entity_id])
    tune_ids = np.flatnonzero(splits=='tune')
    eval_ids = np.flatnonzero(splits=='evaluation')
    thresholds = list(np.linspace(0.05,0.95,37)) + [0.97,0.99,0.995,0.999,1.000001]
    results = [(t,macro_score(tune_ids,owners,y,scores,truth_counts,t)) for t in thresholds]
    threshold, tuning = max(results,key=lambda v:(v[1]['macro_F0.5'],v[0]))
    evaluation = macro_score(eval_ids,owners,y,scores,truth_counts,threshold)
    retrieved = np.bincount(owners,weights=y,minlength=len(sample))
    report = {'version':VERSION,'threshold':threshold,'features':FEATURES,
              'tuning':tuning,'evaluation':evaluation,'sample_businesses':sample_n,
              'sample_seed':seed,'max_block':max_block,
              'search_recall_evaluation':float(retrieved[eval_ids].sum()/max(1,truth_counts[eval_ids].sum())),
              'scope':'Full training S2/S3 streamed against sampled S1. Block-frequency cap depends on S1 pool size.',
              'countries':sample.country.value_counts().to_dict(),
              'versions':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'scikit-learn':sklearn.__version__}}
    atomic_pickle(work/'model.pkl',{'model':model,'report':report,'signature':sig})
    (work/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    pd.DataFrame([{'threshold':t,**m} for t,m in results]).to_csv(work/'thresholds.csv',index=False)
    log(json.dumps(report,indent=2))
    if tuning['correct_accepted'] == 0:
        log('NOTICE: tuning chose no links. Check metrics before uploading this baseline.')
    prepared.cache_clear()


def predict(data,work,max_block=30,preview=False):
    with (work/'model.pkl').open('rb') as f:
        saved = pickle.load(f)
    model,report = saved['model'],saved['report']
    if report['version'] != VERSION or report['features'] != FEATURES or report['max_block'] != max_block:
        raise RuntimeError('Model and retrieval settings differ. Use matching settings.')
    source1_path = data/'test/test_source1.tsv'
    log('Loading test Source 1 only (Sources 2/3 will stream in batches).')
    s1 = pd.read_csv(source1_path,sep='\t',dtype=str,keep_default_na=False,usecols=COLS)[COLS]
    if not s1.entity_id.is_unique:
        raise RuntimeError('Test Source 1 IDs are not unique.')
    index = BlockIndex(s1,max_block)
    paths = [data/f'test/test_source{x}.tsv' for x in [1,2,3]] + [work/'model.pkl']
    sig = signature(paths,{'max_block':max_block})
    db = open_db(work/'test_pairs.sqlite',sig,training=False)
    scan_sources(data,'test',index,db,work,model=model,limit=100000 if preview else None)
    if not preview:
        complete = db.execute('SELECT COUNT(*) FROM progress WHERE complete=1').fetchone()[0]
        if complete != 2:
            raise RuntimeError('Both test sources must finish before export.')
        export_outputs(db,s1,report['threshold'],work/'output')
    db.close()
    prepared.cache_clear()
    del index,s1
    gc.collect()


def export_outputs(db,s1,threshold,out):
    out.mkdir(exist_ok=True,parents=True)
    # Ordered streaming: no giant groupby and no missing singleton rows.
    iterator = iter(db.execute('SELECT s1,candidate,score FROM pairs ORDER BY s1,candidate'))
    current = next(iterator,None)
    mtemp = out/'matching_results.tsv.tmp'
    ctemp = out/'candidate_pairs.tsv.tmp'
    links = 0
    with mtemp.open('w',encoding='utf-8',newline='') as fm, ctemp.open('w',encoding='utf-8',newline='') as fc:
        mw,cw = csv.writer(fm,delimiter='\t',lineterminator='\n'),csv.writer(fc,delimiter='\t',lineterminator='\n')
        mw.writerow(['source1_entity_id','matched_entity_ids'])
        cw.writerow(['source1_entity_id','candidate_entity_ids'])
        for i,entity_id in enumerate(s1.entity_id):
            candidates,accepted = [],[]
            while current is not None and current[0] == i:
                candidates.append(current[1])
                if current[2] >= threshold:
                    accepted.append(current[1])
                current = next(iterator,None)
            mw.writerow([entity_id,','.join(accepted)])
            cw.writerow([entity_id,','.join(candidates)])
            links += len(accepted)
    if current is not None:
        raise RuntimeError('Pair table contains invalid Source 1 row indices.')
    os.replace(mtemp,out/'matching_results.tsv')
    os.replace(ctemp,out/'candidate_pairs.tsv')
    log(f'Exported {len(s1):,} Source 1 rows, {links:,} accepted links.')


def validate(data,work):
    validator = data.parent/'utils/validate_submission.py'
    if not validator.exists():
        raise FileNotFoundError(f'Official validator missing: {validator}')
    cmd = [sys.executable,str(validator),'--matching',str(work/'output/matching_results.tsv'),
           '--candidate',str(work/'output/candidate_pairs.tsv'),'--test-dir',str(data/'test')]
    subprocess.run(cmd,check=True)
    log('Official validator completed successfully. Upload output/matching_results.tsv to the portal.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=['train','preview','predict','validate'])
    parser.add_argument('--data',type=Path,required=True,help='Path to student_resource/dataset')
    parser.add_argument('--work',type=Path,default=Path('/kaggle/working/initial_submission_v1'))
    parser.add_argument('--sample',type=int,default=4000)
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--max-block',type=int,default=30)
    args = parser.parse_args()
    args.work.mkdir(parents=True,exist_ok=True)
    if args.stage == 'train':
        train(args.data,args.work,args.sample,args.seed,args.max_block)
    elif args.stage in ['preview','predict']:
        predict(args.data,args.work,args.max_block,preview=args.stage=='preview')
    else:
        validate(args.data,args.work)


if __name__ == '__main__':
    main()
