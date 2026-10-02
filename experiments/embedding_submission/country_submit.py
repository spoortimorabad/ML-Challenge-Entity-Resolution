"""Same evaluated pipeline, processing one test country at a time to bound disk usage."""
import argparse,csv,gc,json,os,shutil,sqlite3,sys,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import embedding_core as c
import submit_pipeline as s


def prepare_country(data,folder,country):
    done=folder/'ready.json'
    if done.exists():return json.loads(done.read_text())
    target=folder/'dataset/test';target.mkdir(parents=True,exist_ok=True)
    counts={}
    for source in (1,2,3):
        out=target/f'test_source{source}.tsv';tmp=Path(str(out)+'.tmp');n=0
        with tmp.open('w',encoding='utf-8',newline='') as f:
            pd.DataFrame(columns=c.COLS).to_csv(f,sep='\t',index=False)
            for frame in c.read_chunks(data/f'test/test_source{source}.tsv'):
                frame=frame.loc[frame.country.map(c.norm)==country,c.COLS]
                frame.to_csv(f,sep='\t',index=False,header=False);n+=len(frame)
        os.replace(tmp,out);counts[str(source)]=n
        c.log(f'{country}: selected {n:,} Source {source} records')
    c.atomic_json(done,counts)
    return counts


def remove_test_temporaries(folder):
    # Only caches created by THIS country-processing stage; never the training WORK.
    if not (folder/'imported.json').exists():return
    for name in ['input','cache']:
        path=folder/name
        if path.exists():shutil.rmtree(path)


def import_country(db,out,country):
    if db.execute('SELECT 1 FROM complete WHERE country=?',(country,)).fetchone():return
    with (out/'candidate_pairs.tsv').open(newline='',encoding='utf-8') as cf, (out/'matching_results.tsv').open(newline='',encoding='utf-8') as mf:
        cr=csv.DictReader(cf,delimiter='\t');mr=csv.DictReader(mf,delimiter='\t')
        from itertools import zip_longest
        rows=[];total=0
        # One transaction: an interrupted import rolls back, so a rerun is safe.
        with db:
            for a,b in zip_longest(cr,mr):
                assert a is not None and b is not None
                assert a['source1_entity_id']==b['source1_entity_id']
                candidates=a['candidate_entity_ids'];matches=b['matched_entity_ids']
                cs=candidates.split(',') if candidates else []
                ms=matches.split(',') if matches else []
                assert len(cs)<=20 and len(set(cs))==len(cs) and set(ms)<=set(cs)
                rows.append((a['source1_entity_id'],candidates,matches));total+=1
                if len(rows)==1000:
                    db.executemany('INSERT INTO results VALUES (?,?,?)',rows);rows=[]
            if rows:db.executemany('INSERT INTO results VALUES (?,?,?)',rows)
            db.execute('INSERT INTO complete VALUES (?,?)',(country,total))


def export_all(data,work,db):
    out=work/'output';out.mkdir(exist_ok=True)
    mp=out/'matching_results.tsv';cp=out/'candidate_pairs.tsv'
    count=empty=emptyc=links=pairs=0;hist=np.zeros(21,dtype=np.int64)
    with Path(str(mp)+'.tmp').open('w',newline='',encoding='utf-8') as mf, Path(str(cp)+'.tmp').open('w',newline='',encoding='utf-8') as cf:
        mw=csv.writer(mf,delimiter='\t',lineterminator='\n');cw=csv.writer(cf,delimiter='\t',lineterminator='\n')
        mw.writerow(['source1_entity_id','matched_entity_ids']);cw.writerow(['source1_entity_id','candidate_entity_ids'])
        for frame in c.read_chunks(data/'test/test_source1.tsv',['entity_id']):
            all_ids=frame.entity_id.tolist()
            for start in range(0,len(all_ids),800):
                ids=all_ids[start:start+800]
                found={row[0]:row[1:] for row in db.execute('SELECT * FROM results WHERE sid IN ('+','.join('?'*len(ids))+')',ids)}
                assert len(found)==len(ids),'Missing prediction or duplicate Source 1 ID.'
                for sid in ids:
                    candidates,matches=found[sid]
                    mw.writerow([sid,matches]);cw.writerow([sid,candidates])
                    nc=len(candidates.split(',')) if candidates else 0
                    nm=len(matches.split(',')) if matches else 0
                    count+=1;pairs+=nc;links+=nm;empty+=int(nm==0);emptyc+=int(nc==0);hist[nc]+=1
    assert count==db.execute('SELECT COUNT(*) FROM results').fetchone()[0]
    os.replace(str(mp)+'.tmp',mp);os.replace(str(cp)+'.tmp',cp)
    c.atomic_json(out/'submission_summary.json',dict(rows=count,candidate_pairs=pairs,predicted_links=links,
        empty_matches=empty,empty_candidates=emptyc,mean_candidates=pairs/count,candidate_histogram=hist.tolist()))
    c.log(f'Exported {count:,} Source 1 businesses in original test order.')


def run(data,work,model_dir,batch_size=64):
    data=Path(data);work=Path(work);model_dir=Path(model_dir)
    bundle,meta,config=s.initialize(data,work,model_dir)
    marker=work/'country_plan.json'
    if marker.exists():plan=json.loads(marker.read_text())
    else:
        frequencies={}
        for frame in c.read_chunks(data/'test/test_source1.tsv',['country']):
            for country,n in frame.country.map(c.norm).value_counts().items():
                frequencies[country]=frequencies.get(country,0)+int(n)
        plan=sorted(frequencies,key=lambda country:(-frequencies[country],country))
        c.atomic_json(marker,plan)
    db=sqlite3.connect(work/'all_predictions.sqlite')
    db.execute('PRAGMA cache_size=-65536')
    db.execute('CREATE TABLE IF NOT EXISTS results(sid TEXT PRIMARY KEY,candidates TEXT,matches TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS complete(country TEXT PRIMARY KEY,rows INTEGER)');db.commit()
    try:
        for country in plan:
            token=s.hashlib.sha256(country.encode()).hexdigest()[:12]
            folder=work/'countries'/token;folder.mkdir(parents=True,exist_ok=True)
            if db.execute('SELECT 1 FROM complete WHERE country=?',(country,)).fetchone():
                c.atomic_json(folder/'imported.json',{'country':country})
                remove_test_temporaries(folder)
                c.log(f'{country}: completed predictions reused');continue
            c.log(f'Processing country {country!r}. Training files are retained.')
            counts=prepare_country(data,folder/'input',country)
            cache=folder/'cache';cache.mkdir(exist_ok=True)
            subset=folder/'input/dataset'
            if counts['2']+counts['3']==0:
                out=cache/'output';out.mkdir(exist_ok=True)
                for name,col in [('matching_results.tsv','matched_entity_ids'),('candidate_pairs.tsv','candidate_entity_ids')]:
                    with (out/name).open('w',newline='',encoding='utf-8') as f:
                        w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['source1_entity_id',col])
                        for frame in c.read_chunks(subset/'test/test_source1.tsv',['entity_id']):
                            w.writerows((sid,'') for sid in frame.entity_id)
            else:
                # Reserve 3 GiB for metadata, index, output and safety; vectors are country-sized.
                encoder=c.Encoder(batch_size)
                c.encode_pool(subset,cache,encoder,split='test',reserve_gib=3)
                del encoder;gc.collect()
                import torch;torch.cuda.empty_cache()
                c.build_indexes(cache)
                encoder=c.Encoder(batch_size)
                s.predict(subset,cache,bundle,encoder)
                del encoder;gc.collect();torch.cuda.empty_cache()
                out=s.merge_outputs(subset,cache)
            import_country(db,out,country)
            c.atomic_json(folder/'imported.json',{'country':country})
            # Results are durably committed above before disposable test caches are removed.
            remove_test_temporaries(folder)
        export_all(data,work,db)
    finally:db.close()
    s.validate(data,work)
    archive=s.package(work,meta)
    with zipfile.ZipFile(archive,'a',zipfile.ZIP_DEFLATED) as z:
        z.write(work/'country_submit.py','country_submit.py')
        z.writestr('COUNTRY_PROCESSING.txt',
          'Test countries are processed sequentially, with identical model/search settings.\n'
          'Intermediate test caches are removed only after country predictions are committed.\n'
          'Training caches are preserved. Final files are reordered to original test S1 order.\n'
          'To repeat this disk-bounded route, place code plus training_experiment.ipynb in WORK, then run:\n'
          'python country_submit.py --data DATASET --work WORK --model-dir MODEL\n')
    c.log('Finished. Upload output/matching_results.tsv; retain candidate_pairs.tsv and the backup.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',required=True,type=Path)
    parser.add_argument('--work',required=True,type=Path);parser.add_argument('--model-dir',required=True,type=Path)
    parser.add_argument('--batch-size',type=int,default=64)
    a=parser.parse_args();run(a.data,a.work,a.model_dir,a.batch_size)
