import json
from pathlib import Path
root=Path(__file__).resolve().parent
cells=[]
def md(s): cells.append({'cell_type':'markdown','metadata':{},'source':s.splitlines(True)})
def code(s): cells.append({'cell_type':'code','metadata':{},'source':s.splitlines(True),'execution_count':None,'outputs':[]})
md('''# Amazon ML — full-pool embedding experiment

**Purpose:** check the promising embedding pilot against every training Source 2/3 record, then train a new matching classifier. This is an experiment, not a portal submission.

1. Encode all training S2/S3 records with the same combined-name-and-address E5 model as the pilot.
2. Build a compressed FAISS IVF-PQ index for each country. Search selected cells, rather than comparing every pair.
3. Retrieve up to 200 approximate neighbors, reorder these by cosine similarity, and keep **20 candidates per Source 1 business for the matcher**. Measure retrieval at 10/20/50 too.
4. Train on 3,000 Source 1 businesses; choose the acceptance threshold on 500; report results on another 500. All target records participate in retrieval, without using ground truth to select them.

**Kaggle setup:** new notebook; attach the challenge dataset; enable a GPU and Internet; use a fresh output folder. The old 0.756 submission remains separate. Allow several hours: actual time depends on GPU, CPU and text lengths. Progress logs estimate encoding time after work starts.

The model runs locally. No business records go to an external API. This uses the pilot's fixed MIT-licensed E5 revision. It requires roughly 13 GiB free disk initially, including 7.4 GiB for vectors; the notebook checks space. Do not run the old full pipeline again in the same session.

**Saving:** choose **Save Version → Save & Run All**. Checkpoints survive a cell rerun in the same filesystem; they do not survive an unsaved, discarded Kaggle session. Keep the saved version's outputs. Restoring a run requires the entire work folder, not just the small results ZIP.
''')
code('''%pip install -q "faiss-cpu==1.11.0" "transformers>=4.48,<6" safetensors sentencepiece
''')
code('''from pathlib import Path
import os, sys, shutil, torch
DATA = Path('/kaggle/input/datasets/spoortimorabad/ml-challenge-dataset/student_resource/dataset')
WORK = Path('/kaggle/working/full_pool_embedding_v1')
BATCH_SIZE = 64
assert torch.cuda.is_available(), 'Enable a GPU in Kaggle Settings before running.'
assert (DATA/'train/train_source1.tsv').exists(), 'Update DATA to your attached dataset directory.'
WORK.mkdir(parents=True, exist_ok=True)
os.environ['TOKENIZERS_PARALLELISM']='false'
print('GPU:', torch.cuda.get_device_name(0))
print('Free disk GiB:', round(shutil.disk_usage(WORK).free/2**30, 2))
''')
md('''## Load the self-contained pipeline
No variables or files from the old notebook are required. If you change model settings or code, use a new WORK folder to avoid mixing caches.''')
source=(root/'pipeline.py').read_text()
code('PIPELINE_SOURCE = '+repr(source)+'''\n(WORK/'pipeline.py').write_text(PIPELINE_SOURCE)
import importlib.util
spec=importlib.util.spec_from_file_location('full_pool_pipeline', WORK/'pipeline.py')
p=importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
DATA, WORK=p.initialize(DATA,WORK)
''')
md('''## 1. Encode the full target pool
This is the main GPU stage. Source 2 and Source 3 are read in batches. Vectors and record text are saved on disk; both must be retained for resume. A completed 10,000-record batch is checkpointed. If GPU memory runs out, reduce BATCH_SIZE to 32 and rerun from the encoder cell. Do not change the model or text format within an existing WORK folder.''')
code('''encoder=p.Encoder(batch_size=BATCH_SIZE)
p.encode_pool(DATA,WORK,encoder)
sample,truth,query_vectors=p.prepare_queries(DATA,WORK,encoder)
# Release the GPU model after saving the query vectors.
del encoder
import gc
gc.collect()
torch.cuda.empty_cache()
''')
md('''## 2. Build the approximate indexes
This stage uses CPU FAISS. IVF splits vectors into cells; PQ compresses the vectors stored in those cells. Only one country's index is built in RAM at a time.

Completed country indexes and trained quantizers are saved. If interrupted while adding vectors, that country's additions restart; the expensive embeddings remain reusable. The 200-neighbor search is an intermediate retrieval step. Only the final 20 go to the classifier.
''')
code('''p.build_indexes(WORK)
''')
md('''## 3. Measure candidate recall against the full pool
No ground-truth matches are injected. The labels are consulted only after retrieval to measure success. Recall here is the fraction of true links retrieved, not the challenge macro F0.5 score.''')
code('''candidate_rows, candidate_scores=p.retrieve(WORK,sample,query_vectors,nprobe=64,overfetch=200,save_k=50)
records=p.fetch_records(WORK,candidate_rows)
retrieval=p.retrieval_report(sample,truth,candidate_rows,records)
retrieval.to_csv(WORK/'retrieval_summary.csv',index=False)
print(retrieval.to_string(index=False))
''')
md('''## 4. Check whether a more thorough ANN search helps
On tuning businesses only, double the cells searched and increase intermediate retrieval to 400. This diagnoses approximation losses. It still checks selected cells, not every vector. Similar recall does not prove that exact global nearest neighbors were recovered. The matcher below keeps the predeclared 64-cell / 200-neighbor configuration; this diagnostic informs the next experiment.''')
code('''import numpy as np
mask=np.flatnonzero(sample.split.to_numpy()=='tune')
tune_sample=sample.iloc[mask].reset_index(drop=True)
audit_ids,audit_scores=p.retrieve(WORK,tune_sample,query_vectors[mask],nprobe=128,overfetch=400,save_k=50)
audit_records=p.fetch_records(WORK,audit_ids)
audit=p.retrieval_report(tune_sample,truth,audit_ids,audit_records)
audit['search']='128 cells / 400 intermediate neighbors'
reference=retrieval[retrieval.split=='tune'].copy()
reference['search']='64 cells / 200 intermediate neighbors'
import pandas as pd
ann_audit=pd.concat([reference,audit],ignore_index=True)
ann_audit.to_csv(WORK/'ann_tuning_audit.csv',index=False)
print(ann_audit.to_string(index=False))
del audit_records, audit_ids, audit_scores
''')
md('''## 5. Train a new matching classifier and evaluate
The classifier learns from 18 lexical comparison features plus embedding cosine similarity. Each of the 20 retrieved pairs receives its match/nonmatch label from the training ground truth. Unretrieved true matches remain misses in end-to-end evaluation.

The classifier is a HistGradientBoostingClassifier. Its acceptance cutoff is chosen using tuning businesses, not evaluation scores. Empty predictions and singleton businesses are included in macro F0.5. No old cutoff or trained matcher is reused.

The 500 evaluation businesses reproduce the earlier baseline split, so this is a useful development comparison. Because that split has been inspected in earlier work, it is not a newly untouched final holdout. A larger fresh holdout will be needed for the next major model selection.
''')
code('''results=p.fit_and_evaluate(WORK,sample,truth,candidate_rows,candidate_scores,records)
print('\nPASTE BACK THESE RESULTS:')
print(retrieval.to_string(index=False))
print(ann_audit.to_string(index=False))
print(results['evaluation'])
print('Chosen cutoff:',results['threshold'])
'''.replace("print('\nPASTE", "print('\\nPASTE"))
md('''## 6. Save the small experiment report
Download this ZIP after the run completes. It contains metrics, held-out pair scores, the trained matcher, sample rows and runnable pipeline code. It excludes the large vector/database/index caches; those remain in the saved notebook outputs for reuse.

**Do not upload this ZIP to the challenge portal.** There are no test predictions in this experiment. Next, use the full-pool results to decide whether to generate a replacement submission. For that submission, the exported candidate file must contain exactly the candidate sets passed to the matcher, with one row for every test Source 1 entity.
''')
code('''import json, zipfile
from importlib.metadata import version
(WORK/'requirements_run.txt').write_text('\n'.join(f'{name}=={version(name)}' for name in
    ['torch','transformers','numpy','pandas','scikit-learn','faiss-cpu','safetensors','sentencepiece'])+'\n')
(WORK/'README.txt').write_text(
    'Full-pool embedding experiment, not a challenge submission.\n'
    'See experiment_results.json, retrieval_summary.csv and ann_tuning_audit.csv.\n'
    'matcher.pkl is generated by this run. Only load trusted pickle files.\n'
    'Full resume requires vectors.f16, records.sqlite, manifests, index files and configuration, '\
    'which are NOT included in this small archive. Retain the saved Kaggle version outputs.\n')
files=['configuration.json','pool_manifest.json','experiment_results.json','retrieval_summary.csv',
       'ann_tuning_audit.csv','threshold_tuning.csv','heldout_pair_scores.tsv','matcher.pkl',
       'sample.tsv','pipeline.py','requirements_run.txt','README.txt']
archive=Path('/kaggle/working/full_pool_embedding_results.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for name in files:
        assert (WORK/name).exists(), f'Missing result: {name}'
        z.write(WORK/name,arcname=name)
print('Experiment complete. Download:',archive)
from IPython.display import FileLink,display
display(FileLink(str(archive)))
'''.replace("'\n'", "'\\n'").replace("submission.\n'", "submission.\\n'").replace("audit.csv.\n'", "audit.csv.\\n'").replace("files.\n'", "files.\\n'").replace("outputs.\n'", "outputs.\\n'"))
md('''### Interpreting the result
- Compare full-pool **top-20 recall** with the old exact-key recall (~0.696 on its 500-business evaluation). The pilot's 0.969 recall came from a smaller pool and is not an expectation for this run.
- Compare development **macro F0.5** with the old baseline's ~0.788. Your **0.756 leaderboard score** is on different, hidden data.
- `missed_by_search` measures retrieval failures; `retrieved_but_rejected` measures true candidates the matcher discarded.
- A low incorrect-acceptance count matters, but rejecting too much can also reduce the score.
- France appears in test but not training. These training results do not establish French performance.

Reference implementations: [E5 model card](https://huggingface.co/intfloat/multilingual-e5-small) and [FAISS index documentation](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes). No challenge business data is sourced from these links.
''')
nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'},'kaggle':{'accelerator':'gpu','isInternetEnabled':True,'language':'python','sourceType':'notebook'}},'nbformat':4,'nbformat_minor':5}
for i,c in enumerate(cells):
 c['id']=f'cell-{i:02d}'
 if c['cell_type']=='code':
  s=''.join(c['source'])
  if not s.startswith('%'): compile(s,f'cell-{i}','exec')
path=root/'amazon_ml_full_pool_embeddings.ipynb'
path.write_text(json.dumps(nb,indent=1))
print(path)
print(f'{len(cells)} cells; all Python cells compile')
