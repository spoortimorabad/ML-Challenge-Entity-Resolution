import json
from pathlib import Path
root=Path(__file__).resolve().parent
path=root.parent/'full_pool_experiment/amazon_ml_full_pool_embeddings.ipynb'
original=path.read_text()
nb=json.loads(original)
# Idempotent build: never append the same section twice.
nb['cells']=[cell for cell in nb['cells'] if not cell.get('id','').startswith('submission-')]
training=json.dumps(nb,indent=1)
core=(root/'embedding_core.py').read_text().replace('def encode_pool(data,work,encoder,split="train"):',
 'def encode_pool(data,work,encoder,split="train",reserve_gib=5):').replace('needed+5*2**30','needed+reserve_gib*2**30').replace("'Need about 13 GiB free before full-pool encoding. Use a fresh notebook.'", "f'Need {needed/2**30+reserve_gib:.1f} GiB free for this encoding stage.'")
runner=(root/'submit_pipeline.py').read_text().replace("work.parent/'embedding_submission_v1_backup.zip'", "work.parent/(work.name+'_backup.zip')")
country=(root/'country_submit.py').read_text()
# Scratch helpers for the synthetic integration test.
(root/'test_country_core_source.txt').write_text(core)

def md(text):nb['cells'].append({'cell_type':'markdown','metadata':{},'source':text.splitlines(True),'id':f'submission-{len(nb["cells"])}'})
def code(text):
 compile(text,'appended-cell','exec')
 nb['cells'].append({'cell_type':'code','metadata':{},'source':text.splitlines(True),'execution_count':None,'outputs':[],'id':f'submission-{len(nb["cells"])}'})
md('''# 7. Generate the challenge submission — NEW

**This is the submission extension to the same notebook.** Run All now runs the original experiment and then predicts every test Source 1 business. The trained matcher and cutoff from above are reused directly; you do not need to upload a ZIP or create another notebook.

The ~0.8864 result above is local validation. Test predictions must still be generated before a leaderboard submission exists. This section uses the evaluated settings: 64 searched cells, up to 200 intermediate neighbors, **20 final candidates**, and the saved cutoff (0.70 in your completed run).

To fit alongside the training caches, test countries are processed one at a time. Completed country predictions are saved before their temporary **test** vectors/indexes are removed. The training model, training embeddings and training indexes are retained. Country processing also supports France.

Expect additional runtime for the whole test set. Use GPU + Internet and **Save Version → Save & Run All**. Do not run this concurrently with another submission pipeline in the same working filesystem.
''')
code('''from pathlib import Path
import json,sys,subprocess,shutil
# These default paths are the same ones used earlier in this notebook.
TRAINED_WORK=Path(globals().get('WORK','/kaggle/working/full_pool_embedding_v1'))
SUBMISSION_DATA=Path(globals().get('DATA','/kaggle/input/datasets/spoortimorabad/ml-challenge-dataset/student_resource/dataset'))
SUBMISSION_WORK=TRAINED_WORK.parent/'full_pool_embedding_submission'
SUBMISSION_WORK.mkdir(parents=True,exist_ok=True)
assert (TRAINED_WORK/'matcher.pkl').exists(),'Run the training/evaluation cells above first.'
assert (TRAINED_WORK/'experiment_results.json').exists()
assert (SUBMISSION_DATA.parent/'utils/validate_submission.py').exists(),'The original challenge utils folder must be attached.'
print('Using trained model:',TRAINED_WORK/'matcher.pkl')
print('Cutoff:',json.loads((TRAINED_WORK/'experiment_results.json').read_text())['threshold'])
print('Free disk GiB:',round(shutil.disk_usage(SUBMISSION_WORK).free/2**30,2))
'''+ '\nSUBMISSION_CORE='+repr(core)+'\nSUBMISSION_RUNNER='+repr(runner)+'\nCOUNTRY_RUNNER='+repr(country)+'\nTRAINING_REPRODUCTION_NOTEBOOK='+repr(training)+'''
(SUBMISSION_WORK/'embedding_core.py').write_text(SUBMISSION_CORE)
(SUBMISSION_WORK/'submit_pipeline.py').write_text(SUBMISSION_RUNNER)
(SUBMISSION_WORK/'country_submit.py').write_text(COUNTRY_RUNNER)
(SUBMISSION_WORK/'training_experiment.ipynb').write_text(TRAINING_REPRODUCTION_NOTEBOOK)
print('Submission code ready.')
''')
code('''# This is the long-running cell. It resumes committed batches/countries if the files still exist.
cmd=[sys.executable,str(SUBMISSION_WORK/'country_submit.py'),
     '--data',str(SUBMISSION_DATA),'--work',str(SUBMISSION_WORK),
     '--model-dir',str(TRAINED_WORK),'--batch-size','64']
with (SUBMISSION_WORK/'submission_run.log').open('a') as log:
    process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in process.stdout:
        print(line,end='',flush=True);log.write(line);log.flush()
    exit_code=process.wait()
if exit_code:
    raise RuntimeError('Submission stage stopped. See the log above; completed checkpoints are retained.')
''')
md('''## 8. Download after validator PASS
Upload **matching_results.tsv** to the leaderboard portal. If the browser opens a preview, download the small matching-results ZIP, extract it, and upload the TSV inside.

Retain **candidate_pairs.tsv** and the backup ZIP for final review. The candidate file contains exactly the pairs scored by the matcher. The backup includes code, the trained matcher, training notebook, methodology, dependency versions, both TSVs and the validation log. The validator includes test ID existence checks.
''')
code('''from IPython.display import FileLink,display
submission_output=SUBMISSION_WORK/'output'
assert (submission_output/'validated.json').exists(),'Wait for the submission cell and official validator to finish successfully.'
print('UPLOAD THIS TSV:')
display(FileLink(str(submission_output/'matching_results.tsv')))
print('OR download this ZIP and extract the same TSV:')
display(FileLink(str(submission_output/'matching_results_download.zip')))
print('KEEP FOR FINAL REVIEW:')
display(FileLink(str(submission_output/'candidate_pairs.tsv')))
display(FileLink(str(SUBMISSION_WORK.parent/(SUBMISSION_WORK.name+'_backup.zip'))))
print(json.dumps(json.loads((submission_output/'submission_summary.json').read_text()),indent=2))
''')
first=''.join(nb['cells'][0]['source'])
first=first.replace('This is an experiment, not a portal submission.',
 'Sections 1–6 evaluate the model; the new Sections 7–8 generate and validate portal submission files.')
nb['cells'][0]['source']=first.splitlines(True)
for i,cell in enumerate(nb['cells']):
 if cell['cell_type']=='code':
  src=''.join(cell['source'])
  if not src.startswith('%'):compile(src,f'cell-{i}','exec')
path.write_text(json.dumps(nb,indent=1))
print(path)
print(len(nb['cells']),'cells; appended setup, full-test prediction/validation, and download cells.')
