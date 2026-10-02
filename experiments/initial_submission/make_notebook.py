import json
from pathlib import Path
root=Path(__file__).parent
source=(root/'baseline.py').read_text()
cells=[]
def md(text):
    cells.append({'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)})
def code(text):
    cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
                  'source':text.splitlines(keepends=True)})
md('''# Amazon ML Challenge — first submission baseline

**Goal:** train a small supervised model, predict for every test Source 1 business,
produce both required TSV files, and run the official validator.

Use a **fresh private Kaggle notebook**, attach the supplied challenge dataset,
and select CPU. No GPU, external business data, pretrained weights, or internet
access is required. Do not rerun the old TF-IDF/FTS notebooks first.

This is a new conservative baseline, **not the earlier model with a measured
0.9584 score**. Its real score and runtime must be measured on your dataset.
The matching model is still `HistGradientBoostingClassifier`, trained anew with
18 text-comparison features and a newly tuned threshold.

### How candidates are found
Instead of searching millions of records separately for each business, build a
compact index of Source 1 name/address keys and stream Sources 2 and 3 through it.
A shared key generates the same `(Source 1, Source 2/3)` candidate pair as a forward
lookup. Examples: normalized complete name, normalized complete address, or one
name token plus an address number. Country is included in every key. Keys shared
by more than 30 Source 1 businesses are skipped, not truncated.

This avoids broad ranked text queries but can miss typos, transliterations,
incomplete addresses, and common names. It is a precision-oriented starting
point. Do not interpret its score as a guarantee about the leaderboard.

### Run order
1. Configure paths.
2. Write the self-contained source file (long helper cell; no edits needed).
3. Train and evaluate: 4,000 sampled Source 1 businesses split 3,000 / 500 / 500.
   Both full training Source 2/3 files are scanned for candidates.
4. Preview 100,000 test-source records to check throughput.
5. Resume and finish predictions for the full test set.
6. Run the official validator.
7. Download `matching_results.tsv` and save a backup bundle.

**Only upload `matching_results.tsv` to the leaderboard.** Keep `candidate_pairs.tsv`
and the code for the final package. The backup ZIP below is not a substitute for
completing the organizers' final methodology template.

Checkpoints survive cell interruption only while `/kaggle/working` remains
available. Save/download notebook outputs before the session is discarded.
''')
code('''from pathlib import Path
import os
import sys
import shutil
import subprocess
import json

DATA = Path("/kaggle/input/datasets/spoortimorabad/ml-challenge-dataset/student_resource/dataset")
WORK = Path("/kaggle/working/initial_submission_v1")
SCRIPT = Path("/kaggle/working/baseline.py")
SAMPLE_BUSINESSES = 4000
SEED = 2026
MAX_BLOCK = 30

WORK.mkdir(parents=True, exist_ok=True)
for split in ["train", "test"]:
    for source in [1, 2, 3]:
        path = DATA / split / f"{split}_source{source}.tsv"
        assert path.is_file(), f"Check DATA path: {path} does not exist"
assert (DATA / "train/train_ground_truth.tsv").is_file()
assert (DATA.parent / "utils/validate_submission.py").is_file()
print(f"Free disk: {shutil.disk_usage(WORK).free / 1024**3:.1f} GB")
print("The earlier full_training_search_v1.sqlite file is NOT needed.")
print("This notebook does not delete prior files.")

ENV = os.environ.copy()
ENV.update({"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})

def run_stage(stage):
    cmd = [sys.executable, "-u", str(SCRIPT), stage,
           "--data", str(DATA), "--work", str(WORK),
           "--sample", str(SAMPLE_BUSINESSES), "--seed", str(SEED),
           "--max-block", str(MAX_BLOCK)]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1, env=ENV) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
        status = process.wait()
    if status:
        raise RuntimeError(f"Stage {stage} failed (exit {status}). See the message above; do not skip ahead.")
''')
md('''## Write the implementation
This cell writes a Python module; it does not start training. It contains the
complete pipeline and has no dependency on variables from your earlier notebook.
''')
code('%%writefile /kaggle/working/baseline.py\n'+source)
md('''## Train, tune, evaluate
Candidate pairs are labeled using training ground truth. The model learns only
from the 3,000 training businesses. A separate 500 choose the cutoff; another 500
provide evaluation. Metrics include true matches never retrieved by blocking.

This scans 10.3 million target records, but only builds candidate features for
the sampled Source 1 businesses. It is not full-data model training.

The Source 1 block-frequency cap is applied to the sampled reference pool during
training and to all test references during prediction, so test blocks may be
suppressed more often. Training validation does not measure France performance.

A saved model is reused on reruns. To change seed, sample size, key cap, or feature
code, use a NEW work directory rather than mixing checkpoints.
''')
code('''run_stage("train")
metrics = json.loads((WORK / "metrics.json").read_text())
print("\\nNEW baseline evaluation:", metrics["evaluation"])
print("Candidate recall:", metrics["search_recall_evaluation"])
print("New cutoff:", metrics["threshold"])
''')
md('''## Preview test throughput
Builds the full test Source 1 key index, then scores the first 100,000 unprocessed
Source 2/3 records. These scored pairs are checkpointed and reused in the next
cell. All country labels, including France, are supported automatically.

Read the input-rows/second messages. Full-data runtime is not established by the
synthetic checks used to test this code. If throughput is unsuitable for your
available session, stop and share the logs before committing to a longer run.
''')
code('run_stage("preview")\n')
md('''## Complete predictions and export
Resume the test scan. The matching file contains exactly one row per test Source 1
business, including empty predictions. The candidate file contains all pairs
actually scored by the matching model. There is no additional hidden candidate
filter after inference.

Restarting this cell resumes committed input batches. Export itself restarts
safely if interrupted. Neither the model nor threshold is tuned on test labels.
''')
code('run_stage("predict")\n')
md('''## Validate before uploading
This invokes the challenge's own `utils/validate_submission.py` against the full
test files. Do not upload until it reports PASS. No test data was available to
run this official check in the assistant's environment.
''')
code('''run_stage("validate")
from IPython.display import FileLink, display
print("Upload this file to the challenge portal:")
display(FileLink(str(WORK / "output/matching_results.tsv")))
print("Retain this file for the final package:")
display(FileLink(str(WORK / "output/candidate_pairs.tsv")))
''')
md('''## Save a reproducible backup
This ZIP includes code, actual dependency versions, trained model, metrics,
sampled references, and both outputs. It omits large regenerable SQLite caches
and raw challenge data. Run in the same environment to preserve feature/model
compatibility. Model pickle files should only be loaded from trusted runs.

The final competition archive also needs the organizers' completed methodology
template. Use the supplied template and document this approach and any later
changes; do not submit this backup as though the final documentation were done.
''')
code('''import importlib.metadata
import zipfile

requirements = "\\n".join(
    f"{package}=={importlib.metadata.version(package)}"
    for package in ["numpy", "pandas", "scikit-learn", "threadpoolctl"]
) + "\\n"
(WORK / "requirements.txt").write_text(requirements)

readme = f"""Initial entity-resolution baseline

Python: {sys.version}
Install pinned requirements.txt. Raw challenge data is not included.
The model is trained from scratch; no external pretrained model is used.

Reproduce from scratch, supplying your dataset and a new work directory:
python baseline.py train --data PATH_TO_DATASET --work RUN_DIRECTORY --sample {SAMPLE_BUSINESSES} --seed {SEED} --max-block {MAX_BLOCK}
python baseline.py predict --data PATH_TO_DATASET --work RUN_DIRECTORY --max-block {MAX_BLOCK}
python baseline.py validate --data PATH_TO_DATASET --work RUN_DIRECTORY

Upload only output/matching_results.tsv to the live leaderboard.
This backup is not the final competition package: complete the provided methodology template.

Method: country-specific deterministic name/address blocks; skip keys shared by
more than {MAX_BLOCK} reference rows; 18 lexical features; supervised histogram
gradient boosting; entity-disjoint train/tune/evaluation split; macro F0.5 cutoff
selection including singletons. Every inference candidate is exported.
No external business databases, geocoding, or entity lookups are used.

Limitations: exact/partial blocking misses noisy/transliterated records; countries
unseen in training are unvalidated; block cap is reference-pool dependent; this
is a submission-first baseline with no claimed leaderboard score.
"""
(WORK / "README.txt").write_text(readme)

bundle = Path("/kaggle/working/initial_baseline_backup.zip")
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
    archive.write(SCRIPT, "baseline.py")
    for filename in ["model.pkl", "metrics.json", "thresholds.csv",
                     "sample_source1.tsv", "sample.pkl", "requirements.txt", "README.txt"]:
        archive.write(WORK / filename, filename)
    for filename in ["matching_results.tsv", "candidate_pairs.tsv"]:
        archive.write(WORK / "output" / filename, "output/" + filename)

display(FileLink(str(bundle)))
''')
md('''### Code verification
The pipeline was exercised on synthetic files for training, cutoff scoring,
partial-scan recovery, rerunning completed stages, country filtering, Unicode,
unit conflicts, duplicate-free candidate export, empty rows, and match-subset
invariants. This establishes implementation checks, **not challenge performance**.
The complete challenge data and official validator must run in your Kaggle session.
''')
notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
 'language_info':{'name':'python','version':'3.11'},
 'kaggle':{'accelerator':'none','isInternetEnabled':False,'language':'python','sourceType':'notebook'}},
 'nbformat':4,'nbformat_minor':5}
for i,cell in enumerate(cells):
    cell['id']=f'cell-{i:02d}'
p=root/'amazon_ml_initial_submission.ipynb'
p.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
print(p)
