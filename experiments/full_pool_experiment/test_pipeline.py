from pathlib import Path
import tempfile, sqlite3, json
import numpy as np, pandas as pd
import pipeline as p
sample=pd.DataFrame([
 ['S1-a','Alpha Ltd','5 Main Road','US','train'],
 ['S1-b','Beta','7 Main Rd','US','tune'],
 ['S1-c','Gamma','9 Main Road','US','evaluation'],
 ['S1-d','Delta','11 Main Road','US','evaluation'],
],columns=p.COLS+['split'])
truth={'S1-a':{'S2-a'},'S1-b':{'S2-b'},'S1-c':{'S2-c','S3-c'},'S1-d':set()}
records={0:('S2-a','Alpha','5 Main Rd','us'),1:('S2-b','Beta','7 Main Rd','us'),
2:('S2-c','Gamma','9 Main Rd','us'),3:('S2-x','Different','9 Main Rd','us')}
ids=np.array([[0,-1],[1,-1],[2,3],[-1,-1]])
scores=np.array([[.99,-1],[.99,-1],[.99,.7],[-1,-1]])
x,y,owners,candidates=p.make_pairs(sample,truth,ids,scores,records)
assert x.shape==(4,19) and y.tolist()==[1,1,1,0]
m=p.metrics(sample,truth,owners,candidates,[.99,.99,.9,.8],.5,'evaluation')
assert m['correct_accepted']==1 and m['incorrect_accepted']==1 and m['missed_true_links']==1
assert abs(m['macro_F05']-.75)<1e-10
assert m['singletons_correct']==1
r=p.retrieval_report(sample,truth,ids,records)
assert r.query("split=='evaluation' and k==20").iloc[0]['recall']==.5
assert 'தமிழ்' in p.text_view(pd.DataFrame([['id','தமிழ்','é','India']],columns=p.COLS),'combined')[0]
with tempfile.TemporaryDirectory() as td:
 w=Path(td); db=p.open_records(w)
 db.executemany('INSERT INTO records VALUES (?,?,?,?,?)',[(k,*v) for k,v in records.items()]);db.commit();db.close()
 assert p.fetch_records(w,ids)==records
 p.atomic_json(w/'test.json',{'x':1}); assert json.loads((w/'test.json').read_text())=={'x':1}
print('PASS: pair labels, 19 features, macro F0.5, singleton handling, search recall, Unicode preservation, SQLite lookup, atomic JSON.')
