import numpy as np
import pandas as pd
import helpers as h

queries=pd.DataFrame([
 ['S1-a','Alpha Inc','12 Main Road','US'],
 ['S1-b','Singleton','Other address','India'],
],columns=h.COLS)
pool=pd.DataFrame([
 ['S2-a','Alpha','12 Main Rd','US'],
 ['S3-b','Alhpa','12 Main Road','US'],
 ['S2-c','Alpha','12 Main Rd','India'],
],columns=h.COLS)
truth={'S1-a':{'S2-a','S3-b'},'S1-b':set()}
base=[{'S2-a'},set()]
r,details=h.report_candidates('test',[[0,1],[2]],queries,pool,truth,base,2)
assert r['found']==2 and r['true_links']==2 and r['link_recall']==1
assert r['recovered_over_exact']==1 and r['mean_candidates']==1.5
assert r['singleton_businesses']==1
assert h.hybrid([1,2,3],[1,4,5,6],4)==[1,2,4,5]
assert len(h.rrf([[1,2],[2,3]],2))==2
assert 'ग्रीन' in h.clean_embedding_text('ग्रीन')
assert h.text_view(queries,'name')[0]=='query: Alpha Inc'
assert h.text_view(queries,'combined')[0].startswith('query: business name:')
assert len(h.block_keys('Alpha Inc','12 Main Road','US'))>0

from pathlib import Path
import tempfile
with tempfile.TemporaryDirectory() as td:
    data=Path(td);(data/'train').mkdir()
    pool.iloc[:2].to_csv(data/'train/train_source2.tsv',sep='\t',index=False)
    extra=pd.DataFrame([
        ['S3-decoy','Alpha Ltd','999 Elsewhere','US'],
        ['S3-other','Zebras','100 Other','India'],
        ['S3-extra','Strange','800 Nope','US'],
    ],columns=h.COLS)
    extra.to_csv(data/'train/train_source3.tsv',sep='\t',index=False)
    sample=queries.copy();sample['split']='evaluation'
    out_q,out_t,out_p,out_e=h.prepare_pool(sample,truth,data,2,1)
    assert {'S2-a','S3-b','S3-decoy'}<=set(out_p.entity_id)
    index=list(out_q.entity_id).index('S1-a')
    assert 'S3-decoy' in out_e[index]
    assert out_t['S1-a']==truth['S1-a']
print('PASS: exact-candidate retention, true-match retention, hybrid dedup/budgets, recall/singleton/recovery metrics, and Unicode formatting. GPU search and encoding remain unexecuted locally.')
