import json
from pathlib import Path
root=Path(__file__).resolve().parent
cells=[]
def md(s):cells.append({'cell_type':'markdown','metadata':{},'source':s.splitlines(True)})
def code(s):cells.append({'cell_type':'code','metadata':{},'source':s.splitlines(True),'execution_count':None,'outputs':[]})
md('''# Amazon ML — submit the evaluated embedding model

This notebook produces predictions for **all test Source 1 businesses** using the matcher from your completed full-pool embedding experiment (local macro F0.5 ~0.8864).
It does not retrain, increase the training sample, or change the cutoff/search settings.

**Attach two inputs to a new Kaggle notebook:**
1. The original challenge dataset, including `student_resource/utils/validate_submission.py`.
2. The **saved output of your completed full-pool embedding notebook**. It must contain `matcher.pkl` and `experiment_results.json`. These are the model you already trained and its metadata. The small results ZIP can also supply these files if extracted into an attached input.

Enable **GPU** and **Internet**. Use **Save Version → Save & Run All** so completed outputs are retained. Keep the previous experiment saved separately.

The notebook uses only the saved matcher and metadata from the old run, without copying its large training caches. It creates new embeddings/indexes for test S2/S3, because those records differ from the training pool.
Allow several hours; prediction logs estimate remaining time after the first batch. Start with at least ~13 GiB free working disk. Nothing is uploaded to the challenge portal automatically.
''')
code('''%pip install -q "numpy==2.0.2" "pandas==2.3.3" "scikit-learn==1.6.1" "faiss-cpu==1.11.0" "transformers==5.0.0" safetensors sentencepiece
''')
code('''from pathlib import Path
import os,sys,shutil
DATA=Path('/kaggle/input/datasets/spoortimorabad/ml-challenge-dataset/student_resource/dataset')
WORK=Path('/kaggle/working/embedding_submission_v1')
# Leave blank to find the previous completed experiment automatically.
# If more than one is attached, paste the folder containing its matcher.pkl here.
MODEL_DIR=''
BATCH_SIZE=64
WORK.mkdir(parents=True,exist_ok=True)
os.environ['TOKENIZERS_PARALLELISM']='false'
import torch
assert torch.cuda.is_available(),'Enable a GPU before running this notebook.'
assert (DATA/'test/test_source1.tsv').exists(),'Update DATA to your challenge dataset path.'
assert (DATA.parent/'utils/validate_submission.py').exists(),'Attach the original challenge utils folder too.'
print('GPU:',torch.cuda.get_device_name(0))
print('Free working disk GiB:',round(shutil.disk_usage(WORK).free/2**30,2))
''')
md('''## Write and check the submission pipeline
The code is embedded here, so the notebook is self-contained apart from the dataset and your trained matcher. Existing caches require the same inputs and code; use a new WORK folder if changing the experiment.''')
core=(root/'embedding_core.py').read_text();runner=(root/'submit_pipeline.py').read_text()
training=(root.parent/'full_pool_experiment/amazon_ml_full_pool_embeddings.ipynb').read_text()
code('CORE_SOURCE='+repr(core)+'\nRUNNER_SOURCE='+repr(runner)+'\nTRAINING_NOTEBOOK='+repr(training)+'''\n(WORK/'embedding_core.py').write_text(CORE_SOURCE)
(WORK/'submit_pipeline.py').write_text(RUNNER_SOURCE)
(WORK/'training_experiment.ipynb').write_text(TRAINING_NOTEBOOK)
# Use a subprocess for dependency isolation and to release memory between stages.
import subprocess,json
probe=subprocess.run([sys.executable,'-c',
    "import submit_pipeline as s; print(s.locate_model("+repr(MODEL_DIR or None)+"))"],
    cwd=WORK,text=True,capture_output=True)
if probe.returncode:
    print(probe.stderr)
    raise RuntimeError('Attach the saved experiment output, or set MODEL_DIR to its model folder.')
MODEL_DIR=Path(probe.stdout.strip())
meta=json.loads((MODEL_DIR/'experiment_results.json').read_text())
print('Using trained model:',MODEL_DIR/'matcher.pkl')
print('Previous local evaluation:',meta['evaluation']['macro_F05'])
print('Cutoff:',meta['threshold'])
assert meta['candidate_k']==20 and meta['ann_nprobe']==64 and meta['ann_overfetch']==200

def run_stage(stage):
    cmd=[sys.executable,str(WORK/'submit_pipeline.py'),'--data',str(DATA),
         '--work',str(WORK),'--model-dir',str(MODEL_DIR),'--stage',stage,
         '--batch-size',str(BATCH_SIZE)]
    with (WORK/f'{stage}.log').open('a') as log:
        process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            print(line,end='',flush=True)
            log.write(line);log.flush()
        rc=process.wait()
    if rc:raise RuntimeError(f'{stage} failed. Check the log above. Completed checkpoints remain in WORK.')
''')
md('''## 1. Encode test Source 2/3
All test target records are encoded. Every 10,000 completed records are checkpointed. If a GPU memory error occurs, reduce BATCH_SIZE to 32 and rerun this stage. Encoding does not fine-tune the E5 model.

Progress can resume only when WORK and its caches still exist. A new Save & Run All session may start with empty working storage; do not assume it inherits an unsaved interactive run.''')
code("run_stage('encode')\n")
md('''## 2. Index the test pool
Country-specific IVF-PQ indexes are created dynamically, including France. Index training is an unlabeled compression/search step; the saved matching classifier is unchanged.
Completed country indexes are reusable. An interrupted country's additions restart from its saved trained quantizer.''')
code("run_stage('index')\n")
md('''## 3. Predict all test businesses
For each Source 1 business, retrieve up to 200 approximate neighbors, rerank them by cosine similarity, and keep at most **20**. Score those 20 with the saved matcher and apply its saved cutoff (0.70 in your run).

The candidate file records **all pairs scored by the classifier**, including rejected pairs. Both files preserve one row per Source 1 business, including empty results. Prediction shards are saved every 2,000 businesses, allowing a same-filesystem rerun to skip completed batches.''')
code("run_stage('predict')\n")
md('''## 4. Merge, validate, and package
This runs the organizer's validator with `--check-ids`, so IDs are checked against the test dataset too. If validation fails, the cell stops; do not upload an unvalidated file.

Only `matching_results.tsv` goes to the leaderboard portal. Keep `candidate_pairs.tsv` and the backup for final review. The backup includes runnable prediction code, the training notebook, model, dependency versions, methodology and validation log. At final closure, complete the organizer's documentation template and any additional packaging instructions.''')
code("run_stage('finalize')\n")
md('''## Download your submission
If a TSV link opens a preview, download **matching_results_download.zip**, extract it on your computer, and upload the extracted **matching_results.tsv** to the challenge portal. Do not upload the download ZIP as the leaderboard prediction file.

The backup ZIP excludes large vector/index caches. Keep the completed saved Kaggle version if you want to reuse those caches later.''')
code('''from IPython.display import FileLink,display
print('PORTAL: upload this TSV after validator PASS')
display(FileLink(str(WORK/'output/matching_results.tsv')))
print('EASIER DOWNLOAD: extract this ZIP to obtain that same TSV')
display(FileLink(str(WORK/'output/matching_results_download.zip')))
print('KEEP FOR FINAL REVIEW:')
display(FileLink(str(WORK/'output/candidate_pairs.tsv')))
display(FileLink(str(WORK.parent/'embedding_submission_v1_backup.zip')))
print(json.dumps(json.loads((WORK/'output/submission_summary.json').read_text()),indent=2))
''')
for i,cell in enumerate(cells):
 cell['id']=f'cell-{i:02d}'
 if cell['cell_type']=='code':
  src=''.join(cell['source'])
  if not src.startswith('%'):compile(src,f'cell-{i}','exec')
notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'},'kaggle':{'accelerator':'gpu','isInternetEnabled':True,'language':'python','sourceType':'notebook'}},'nbformat':4,'nbformat_minor':5}
path=root/'amazon_ml_embedding_submission.ipynb'
path.write_text(json.dumps(notebook,indent=1))
print(path)
print(len(cells),'cells; Python cell compilation passed')
