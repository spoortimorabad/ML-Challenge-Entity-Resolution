import json
from pathlib import Path
root=Path(__file__).resolve().parent
cells=[]
def md(x):cells.append({'cell_type':'markdown','metadata':{},'source':x.splitlines(True)})
def code(x):
 if not x.startswith('%'):compile(x,'notebook-cell','exec')
 cells.append({'cell_type':'code','metadata':{},'source':x.splitlines(True),'outputs':[],'execution_count':None})
md('''# Amazon ML — deadline upgrade, cached training + submission

Keep your submitted **0.84** result as the fallback. This notebook attempts one measured improvement:
- **30,000 training businesses**, 2,000 tuning businesses, and 2,000 fresh evaluation businesses.
- Same fixed E5 model and 19 comparison features; more labeled examples for the matcher.
- Compare standard search (64 cells / 200 intermediate neighbors) with stronger search (128 / 400), always **20 final candidates**.
- Choose search and cutoff on tuning data and estimated runtime. Compare the chosen new pipeline to the saved old model on the same fresh evaluation businesses.
- Start full test prediction automatically **only if evaluation improves and the estimated time fits the target**.

## Inputs and settings
Use a new Kaggle notebook. Attach **(1) the original challenge dataset and (2) the full saved output of the previous notebook**. The second input needs `full_pool_embedding_v1/vectors.f16`, `records.sqlite`, country `.faiss` files, `sample.tsv`, `matcher.pkl`, and metadata. The small results ZIP alone is insufficient.

Enable GPU + Internet. Do not copy the large training caches to working storage: they are read directly from the attached input. Training target embeddings/indexes are reused. Test embeddings still need to be generated.

The planned finish is **27 September 2026, 10:30 p.m. IST**, leaving time before the stated 11:59 p.m. deadline. Estimates include a margin but are not guarantees. Use **Save Version → Save & Run All**. Expect the evaluation stage to finish much sooner than full submission prediction; watch its report.
''')
code('''%pip install -q "numpy==2.0.2" "pandas==2.3.3" "scikit-learn==1.6.1" "faiss-cpu==1.11.0" "transformers==5.0.0" safetensors sentencepiece
''')
code('''from pathlib import Path
import os,sys,json,subprocess,shutil
DATA=Path('/kaggle/input/datasets/spoortimorabad/ml-challenge-dataset/student_resource/dataset')
# Leave blank for automatic discovery. If needed, set to the attached full_pool_embedding_v1 folder.
CACHE_DIR=''
ROOT=Path('/kaggle/working/ml_deadline_upgrade_v2')
CODE_DIR=ROOT/'code'
UPGRADE_WORK=ROOT/'training_upgrade'
SUBMISSION_WORK=ROOT/'submission'
TARGET_FINISH='2026-09-27T22:30:00+05:30'
BATCH_SIZE=64
CODE_DIR.mkdir(parents=True,exist_ok=True)
os.environ['TOKENIZERS_PARALLELISM']='false'
import torch
assert torch.cuda.is_available(),'Enable a GPU in notebook settings.'
assert (DATA/'train/train_source1.tsv').exists(),'Update DATA to the challenge dataset path.'
assert (DATA.parent/'utils/validate_submission.py').exists(),'The original challenge utils folder is required.'
print('GPU:',torch.cuda.get_device_name(0))
print('Free working disk GiB:',round(shutil.disk_usage(ROOT).free/2**30,2))
''')
modules={name:(root/name).read_text() for name in ['embedding_core.py','submit_pipeline.py','upgrade_train.py']}
code('MODULES='+repr(modules)+'''
for name,text in MODULES.items():
    (CODE_DIR/name).write_text(text)
probe=subprocess.run([sys.executable,'-c',
    'import upgrade_train as u; print(u.resolve_cache('+repr(CACHE_DIR or None)+'))'],
    cwd=CODE_DIR,text=True,capture_output=True)
if probe.returncode:
    print(probe.stderr)
    raise RuntimeError('Attach the FULL saved previous notebook output and check CACHE_DIR.')
CACHE_DIR=Path(probe.stdout.strip())
print('Read-only training cache:',CACHE_DIR)
print('Training target vectors will NOT be regenerated or copied.')

def run_logged(command,name):
    with (ROOT/name).open('a') as log:
        proc=subprocess.Popen(command,cwd=CODE_DIR,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in proc.stdout:
            print(line,end='',flush=True);log.write(line);log.flush()
        returncode=proc.wait()
    if returncode:
        raise RuntimeError('Stage stopped. See the preceding log. Completed checkpoints remain in ROOT.')
''')
md('''## 1. Train and compare using the saved training pool
This draws a new sample excluding **all 4,000 businesses from the old experiment**. It generates real retrieval candidates, without injecting ground-truth matches. Training and pair-feature batches are checkpointed.

The old model is scored using its original cutoff and standard search. The new model's cutoff and search profile use only the tuning businesses. Both are then compared on the same fresh evaluation businesses. This is a local comparison, not a guarantee of a leaderboard gain; France remains absent from labeled training.

The runtime benchmark includes query encoding, retrieval, feature computation, and classifier prediction. Its forecast adds 35% to benchmarked prediction time plus two hours for test encoding/indexing/export. Cold disk access and GPU variation can still make the full run slower.
''')
code('''run_logged([sys.executable,str(CODE_DIR/'upgrade_train.py'),
    '--data',str(DATA),'--cache',str(CACHE_DIR),'--work',str(UPGRADE_WORK),
    '--target-finish',TARGET_FINISH,'--batch-size',str(BATCH_SIZE)],'upgrade_training.log')
report=json.loads((UPGRADE_WORK/'upgrade_report.json').read_text())
print(json.dumps(report,indent=2))
''')
# The training notebook in the backup reproduces the new training procedure.
training_nb={'cells':json.loads(json.dumps(cells)),'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
for i,cell in enumerate(training_nb['cells']):cell['id']=f'training-{i}'
training_text=json.dumps(training_nb,indent=1)
md('''## 2. Conditional test submission run
If the new model improves fresh evaluation and the runtime estimate fits, this starts the full test run with the selected model, cutoff and search profile. It does not upload anything to the portal.

The 20 candidates scored for each business are all written to `candidate_pairs.tsv`. The accepted subset goes to `matching_results.tsv`. All test S1 businesses get rows, including empty predictions. Test predictions checkpoint every 2,000 businesses. Encoding also checkpoints.

If the gate says to keep the fallback, no expensive test run starts. Share the report rather than overriding the gate under deadline pressure. A better local score does not guarantee a better leaderboard score.
''')
code('TRAINING_NOTEBOOK='+repr(training_text)+'''
from datetime import datetime,timezone
report=json.loads((UPGRADE_WORK/'upgrade_report.json').read_text())
selected=report['chosen_profile']
remaining=(datetime.fromisoformat(TARGET_FINISH)-datetime.now(timezone.utc)).total_seconds()/3600
needed=report['estimated_test_hours_with_margin'][selected]
RUN_SUBMISSION=report['promote_to_test'] and remaining>=needed
if RUN_SUBMISSION:
    SUBMISSION_WORK.mkdir(parents=True,exist_ok=True)
    for name,text in MODULES.items():
        (SUBMISSION_WORK/name).write_text(text)
    (SUBMISSION_WORK/'training_experiment.ipynb').write_text(TRAINING_NOTEBOOK)
    run_logged([sys.executable,str(SUBMISSION_WORK/'submit_pipeline.py'),
        '--data',str(DATA),'--work',str(SUBMISSION_WORK),
        '--model-dir',str(UPGRADE_WORK/'model'),'--stage','all',
        '--batch-size',str(BATCH_SIZE)],'test_submission.log')
else:
    print('Full test run NOT started. Keep the existing 0.84 submission.')
    print('Evaluation gate:',report['promote_to_test'])
    print('Hours remaining until target:',round(remaining,2),'Estimated hours needed:',round(needed,2))
''')
md('''## 3. Download results
Only upload the new **matching_results.tsv** after the official validator passes. If a TSV opens a preview, download its ZIP and extract it. Keep the candidate file and backup for final review.

If the evaluation/time gate did not pass, you still get the training report and saved new model for inspection, but your existing 0.84 submission remains the fallback. No leaderboard submission is changed automatically.
''')
code('''import zipfile
from IPython.display import FileLink,display
# Small upgrade report backup, independent of whether test prediction ran.
report_zip=ROOT/'upgrade_training_results.zip'
with zipfile.ZipFile(report_zip,'w',zipfile.ZIP_DEFLATED) as z:
    for name in ['upgrade_config.json','upgrade_report.json','country_evaluation.json',
                 'threshold_tuning.csv','sample.tsv','model/matcher.pkl','model/experiment_results.json']:
        if (UPGRADE_WORK/name).exists():z.write(UPGRADE_WORK/name,name)
    for name in MODULES:z.write(CODE_DIR/name,name)
    z.writestr('training_experiment.ipynb',TRAINING_NOTEBOOK)
display(FileLink(str(report_zip)))
out=SUBMISSION_WORK/'output'
if (out/'validated.json').exists():
    print('VALIDATED — upload matching_results.tsv:')
    display(FileLink(str(out/'matching_results.tsv')))
    display(FileLink(str(out/'matching_results_download.zip')))
    print('Retain for final review:')
    display(FileLink(str(out/'candidate_pairs.tsv')))
    display(FileLink(str(ROOT/'embedding_submission_v2_backup.zip')))
else:
    print('No validated replacement submission yet. Your existing 0.84 remains the fallback.')
''')
for i,cell in enumerate(cells):cell['id']=f'cell-{i:02d}'
nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'},'kaggle':{'accelerator':'gpu','isInternetEnabled':True,'language':'python','sourceType':'notebook'}},'nbformat':4,'nbformat_minor':5}
path=root/'amazon_ml_deadline_upgrade.ipynb';path.write_text(json.dumps(nb,indent=1))
print(path);print(len(cells),'cells; all Python cells compile')
