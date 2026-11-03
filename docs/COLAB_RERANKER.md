# Colab Strata reranker arm

This is a fourth, fresh listwise arm over the frozen controlled cases. Run these commands **manually** in the Colab runtime where Qwen3.8-Flash-Next IQ3_S with MTP is deployed. The script uses only this repository and Python's standard library; it makes one chat request per case and never reads qrels during ranking. No local GPU or remote Colab run is claimed here.

## Start the existing deployment

Use the existing [Colab deployment notebook](../notebooks/01_strata_colab.ipynb). Set its `REPO_REF` to `codex/colab-reranker` after that branch is published. Set the Strata launch port to **8081**, and edit its smoke subprocess command to include `--base-url http://127.0.0.1:8081`; the earlier successful runtime used 8081 because 8080 was occupied. Keep the same IQ3_S and MTP deployment settings. Run notebook setup, preflight, launch and smoke cells. Leave the optional stop/cleanup cell until after this experiment. The smoke test itself makes chat requests before the benchmark, so the model may already be warm; the reranker adds no further warmup. The new runner records actual `/health`, `/v1/models`, `/props`, and `/metrics` metadata and does not assume the previous engine version still applies.

The notebook's setup cell defines the randomly named `ROOT`. In a new Colab cell, set the working directory to that checkout. Its test cell installs `.[test]` and runs tests; if you need to rerun them, use:

```python
import os
os.chdir(ROOT)
```

```bash
pip install -e '.[test]'
python -m pytest -q
```

`STRATA_API_KEY` must be set in the Colab environment for the already running server. The deployment notebook may generate and set this key in memory; do not paste a literal key into a cell or save it in the export. The runner reads it only from that variable. The server defaults to `http://127.0.0.1:8081`. If `/v1/models` reports more than one loaded model, pass `--model MODEL_ID` explicitly.

## Import the frozen data

The private ZIP is `完整实验材料.zip` in the Drive folder `MyDrive/research-agent-reranker-controlled-20261103`, if that folder is visible in the mounted account. You can instead upload/copy the same ZIP into `/content` and change `zip_path` below. Do not fetch public data or regenerate cases. For the Drive route, first run:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then run this Python cell:

```python
from pathlib import Path
from zipfile import ZipFile
import hashlib

zip_path = Path('/content/drive/MyDrive/research-agent-reranker-controlled-20261103/完整实验材料.zip')
dest = Path('/content/reranker-data')
expected_zip = 'f604a4109bf13721d85876b48d58cd0a39d950b2b5143cd363730b046b9e95ed'
assert hashlib.sha256(zip_path.read_bytes()).hexdigest() == expected_zip
members = {
    'experiment-v1/cases.json': 'cases.json',
    'experiment-v1/qrels.json': 'qrels.json',
    'experiment-v1/results/voyage_results.json': 'voyage_results.json',
    'experiment-v1/results/jev_results.json': 'jev_results.json',
    'experiment-v1/results/assistant_results.json': 'assistant_results.json',
    'report/LABEL_LIMITATIONS.md': 'LABEL_LIMITATIONS.md',
}
dest.mkdir(parents=True, exist_ok=True)
with ZipFile(zip_path) as archive:
    for member, target in members.items():
        (dest / target).write_bytes(archive.read(member))
assert hashlib.sha256((dest / 'cases.json').read_bytes()).hexdigest() == '87ae227c0cf7eff47d50d4c41717f2c2c90a5c065100a13bcff1530ef29b6757'
assert hashlib.sha256((dest / 'qrels.json').read_bytes()).hexdigest() == '1154d6dafadcbfda773b62d08473d830dba15becc9a5fef4b59b017b92cae2df'
print('Frozen cases, qrels, and three baselines verified in', dest)
```

The exact member allowlist reads only the required files. The qrels are used only after rankings have been frozen.

## Run and inspect

With the Strata server still running, from the repository root:

```bash
python scripts/run_strata_reranker.py --input /content/reranker-data/cases.json --output-dir /content/strata-reranker-none --base-url http://127.0.0.1:8081
```

The default request uses `temperature=0`, `reasoning_effort=none`, 4096 maximum output tokens and a 600 second HTTP timeout. A different reasoning effort is a **separate run** with a new empty output directory, for example `--reasoning-effort medium --output-dir /content/strata-reranker-medium`. There is no hidden chat warmup, cache, retry, partial pool, repaired JSON, padded ranking, or score. The output directory must be new or empty, preventing accidental overwrite. A failed case does not stop later cases; every completed request has its raw response and failure reason saved. Any failure makes the CLI exit nonzero. If initialization fails, the summary records the initialization error and no case calls occur.

The CLI prints each case's start and completion to stderr. It writes `runtime_metadata.json` before ranking, and each completed result immediately to `cases/001.json`, etc. `strata_results.json` then records the full neutral requests, raw responses, usage, model metadata, fresh per-case time from request preparation through parsing and validation, separate HTTP wall time, initialization time, and whole experiment time. It filters the API key from artifacts. Check `success_count=10` and `failed_count=0` before evaluating. If any case fails, review its raw output and rerun the **whole arm** into a new directory; do not silently patch rankings or combine runs.

Optional hardware snapshot before and after the run, saved outside the results directory:

```bash
nvidia-smi > /content/strata-reranker-gpu-before.txt
# run the experiment command above
nvidia-smi > /content/strata-reranker-gpu-after.txt
```

## Evaluate four arms

Only after a complete Strata run, evaluate the frozen 10 cases with the existing metrics:

```bash
python scripts/evaluate_controlled_rerank.py \
  --cases /content/reranker-data/cases.json --qrels /content/reranker-data/qrels.json \
  --result voyage=/content/reranker-data/voyage_results.json \
  --result jev=/content/reranker-data/jev_results.json \
  --result assistant=/content/reranker-data/assistant_results.json \
  --result strata=/content/strata-reranker-none/strata_results.json \
  --output /content/strata-reranker-none/four_arm_metrics.json
```

The primary result includes all 10 cases. `CC2` has a known label limitation: read `LABEL_LIMITATIONS.md` before interpretation. A nine-case exclusion is a clearly labeled **post hoc sensitivity analysis**, separate from the primary result. The following Python cell writes filtered copies under a separate directory:

```python
from pathlib import Path
import json

base = Path('/content/reranker-data')
run = Path('/content/strata-reranker-none')
out = run / 'sensitivity_excluding_CC2'
out.mkdir(exist_ok=False)
inputs = {'cases.json': base / 'cases.json', 'qrels.json': base / 'qrels.json',
          'voyage_results.json': base / 'voyage_results.json',
          'jev_results.json': base / 'jev_results.json',
          'assistant_results.json': base / 'assistant_results.json',
          'strata_results.json': run / 'strata_results.json'}
for name, path in inputs.items():
    document = json.loads(path.read_text(encoding='utf-8'))
    key = 'cases' if name in ('cases.json', 'qrels.json') else 'results'
    assert sum(row['query_id'] == 'CC2' for row in document[key]) == 1
    document[key] = [row for row in document[key] if row['query_id'] != 'CC2']
    document['sensitivity_note'] = 'Post hoc exclusion of CC2; primary analysis retains all 10 cases.'
    (out / name).write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding='utf-8')
```

Run the nine-case evaluator in a new Python cell:

```python
import subprocess
import sys

subprocess.run([
    sys.executable, str(ROOT / 'scripts/evaluate_controlled_rerank.py'),
    '--cases', str(out / 'cases.json'), '--qrels', str(out / 'qrels.json'),
    '--result', f'voyage={out / "voyage_results.json"}',
    '--result', f'jev={out / "jev_results.json"}',
    '--result', f'assistant={out / "assistant_results.json"}',
    '--result', f'strata={out / "strata_results.json"}',
    '--output', str(out / 'four_arm_metrics_excluding_CC2.json'),
], check=True)
```

Preserve the original 10-case files and metric report. The evaluator requires complete, matching case/result/qrels coverage and will reject an incomplete arm.

## Export the Colab evidence

After the ranking and evaluation cells finish, run this Python cell. It archives only the reranker output directory, optional GPU snapshots, and the checked-out code SHA into the already mounted experiment folder. The destination must already exist; the cell refuses to replace a prior ZIP. The model weights and API key are not included.

```python
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from datetime import datetime, timezone
import json
import subprocess

run = Path('/content/strata-reranker-none')
drive_dir = Path('/content/drive/MyDrive/research-agent-reranker-controlled-20261103')
assert drive_dir.is_dir(), 'Mount Drive and verify the experiment folder first'
assert (run / 'strata_results.json').is_file(), 'Run the reranker first'
commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
archive_path = drive_dir / f'strata-reranker-{stamp}.zip'
assert not archive_path.exists(), 'Choose a new export filename'
with ZipFile(archive_path, 'x', compression=ZIP_DEFLATED) as archive:
    for path in sorted(run.rglob('*')):
        if path.is_file():
            archive.write(path, f'strata-reranker-none/{path.relative_to(run).as_posix()}')
    for name in ('strata-reranker-gpu-before.txt', 'strata-reranker-gpu-after.txt'):
        path = Path('/content') / name
        if path.is_file():
            archive.write(path, f'gpu/{name}')
    archive.writestr('code_commit.json', json.dumps({'commit': commit}, indent=2))
print('Exported', archive_path)
```
