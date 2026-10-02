import csv
import importlib.util
import json
import pickle
import sqlite3
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import baseline as b


def write_tsv(path,columns,rows):
    pd.DataFrame(rows,columns=columns).to_csv(path,sep='\t',index=False)


def fixture(root):
    data = root/'student_resource/dataset'
    for split in ['train','test']:
        (data/split).mkdir(parents=True)
        s1,s2,s3,gt=[],[],[],[]
        for i in range(120):
            country = 'India' if i%2 else 'US'
            if split == 'test' and i%3==0:
                country = 'France'
            name = f'Companyword{i} Trading Private Limited'
            addr = f'{i+100} Uniqueavenue{i} Road Unit 3B2 Cityname{i}'
            s1.append([f'S1-{i}',name,addr,country])
            # true same name and true completely different name at exact address
            s2.append([f'S2-{i}',name,addr,country])
            s3.append([f'S3-{i}',f'Alternateword{i}',addr,country])
            s2.append([f'S2-neg{i}',name,f'{i+9999} Different Road Othercity',country])
            # singleton S1 has only decoys in target files
            if i%10==0:
                s2[-2][2] = '888 Distant Avenue Othercity'
                s3[-1][2] = '777 Other Avenue Elsewhere'
                gt.append([f'S1-{i}',''])
            else:
                gt.append([f'S1-{i}',f'S2-{i},S3-{i}'])
        s1.append(['S1-nocandidates','Unfindable Zyxwv','9999999 Nowhere','France'])
        gt.append(['S1-nocandidates',''])
        for source,rows in [(1,s1),(2,s2),(3,s3)]:
            write_tsv(data/split/f'{split}_source{source}.tsv',b.COLS,rows)
        if split=='train':
            write_tsv(data/split/'train_ground_truth.tsv',['source1_entity_id','matched_entity_ids'],gt)
    return data


def test_all():
    # Features distinguish suite conflicts and missingness; Unicode survives.
    a=('S1-a','Alpha Ltd','12 Main Road Suite 315','India')
    c=('S2-a','Alpha Limited','12 Main Rd Suite 127','India')
    f=dict(zip(b.FEATURES,b.features(a,c)))
    assert f['unit_conflict']==1 and f['number_disjoint']==0
    assert b.norm('ग्रीन 商店')
    frame=pd.DataFrame([a],columns=b.COLS)
    idx=b.BlockIndex(frame)
    assert idx.batch_candidates([c])[0]=={0}
    assert not idx.batch_candidates([(c[0],c[1],c[2],'US')])[0]
    # Broad keys are skipped rather than silently truncated.
    broad=pd.DataFrame([(f'S1-{i}',a[1],a[2],a[3]) for i in range(4)],columns=b.COLS)
    assert not b.BlockIndex(broad,max_block=3).batch_candidates([a])[0]
    # Entity-level metric includes missing candidates and singleton semantics.
    result=b.macro_score(np.array([0,1,2]),np.array([0,1]),np.array([1,0]),
                         np.array([.9,.9]),np.array([2,0,0]),.5)
    assert abs(result['macro_F0.5']-((1.25/1.5)+0+1)/3)<1e-9
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        data=fixture(root)
        work=root/'work';work.mkdir()
        # stratified 80-business test fixture
        b.train(data,work,sample_n=80,seed=19)
        timestamp=(work/'model.pkl').stat().st_mtime_ns
        b.train(data,work,sample_n=80,seed=19)
        assert timestamp==(work/'model.pkl').stat().st_mtime_ns
        # Preview commits a partial scan; predict resumes it.
        original_scan = b.scan_sources
        def short_preview(*args, **kwargs):
            if kwargs.get('limit') is not None:
                kwargs['limit'] = 1
            return original_scan(*args, **kwargs)
        b.scan_sources = short_preview
        b.predict(data,work,preview=True)
        partial = sqlite3.connect(work/'test_pairs.sqlite')
        assert partial.execute('SELECT complete FROM progress').fetchone()[0] == 0
        partial.close()
        b.scan_sources = original_scan
        b.predict(data,work)
        m=pd.read_csv(work/'output/matching_results.tsv',sep='\t',keep_default_na=False)
        c=pd.read_csv(work/'output/candidate_pairs.tsv',sep='\t',keep_default_na=False)
        assert len(m)==121 and m.source1_entity_id.is_unique
        assert list(m.columns)==['source1_entity_id','matched_entity_ids']
        assert list(c.columns)==['source1_entity_id','candidate_entity_ids']
        assert (m.source1_entity_id==c.source1_entity_id).all()
        for matched,candidates in zip(m.matched_entity_ids,c.candidate_entity_ids):
            ms=set(filter(None,matched.split(',')));cs=set(filter(None,candidates.split(',')))
            assert ms<=cs
            assert len(cs)==len(list(filter(None,candidates.split(','))))
            assert all(x.startswith(('S2-','S3-')) for x in cs)
        assert m.loc[m.source1_entity_id=='S1-nocandidates','matched_entity_ids'].iloc[0]==''
        assert c.loc[c.source1_entity_id=='S1-nocandidates','candidate_entity_ids'].iloc[0]==''
        db=sqlite3.connect(work/'test_pairs.sqlite')
        scored=db.execute('SELECT COUNT(*) FROM pairs').fetchone()[0]
        assert scored==sum(len(list(filter(None,x.split(',')))) for x in c.candidate_entity_ids)
        previous=(work/'output/matching_results.tsv').read_bytes()
        b.predict(data,work)
        assert previous==(work/'output/matching_results.tsv').read_bytes()
        # France rows are processed even though absent from training fixture.
        assert 'S1-3' in set(m.source1_entity_id)
        db.close()
    print('PASS: end-to-end synthetic training, resume, metrics, Unicode, block caps, and TSV invariants.')

if __name__=='__main__':
    test_all()
