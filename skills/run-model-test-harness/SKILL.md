# Run and read the model test harness

Use this to check that every model in the catalog still runs, uses the GPU it
should, and produces the expected output. It is a manual, per-OS check (not CI):
it needs real weights and real micromamba envs, which no other test touches. Run
it after adding or changing a model, or after an env bump, on each OS.

Script: `backend/scripts/test_models.py`. Read its module docstring too.

## How to run

```
cd backend
source venv/bin/activate            # Windows: venv\Scripts\activate
python scripts/test_models.py               # every cls model in the catalog
python scripts/test_models.py --model AFR-DFV-v2   # one (repeatable)
python scripts/test_models.py --skip-missing       # only what is downloaded
python scripts/test_models.py --model-dir ~/AddaxAI/models/cls/<id>          # a local dir, bypass the catalog
python scripts/test_models.py --model-dir ./staged/<id> --record-reference   # freeze this run as the baseline
```

The parent process (the harness) needs a supported Python (3.11 to 3.13, not
3.14) with `requirements.txt` installed. The model workers run in the micromamba
envs, which the script builds itself if missing or drifted.

## What it does

- Reads the **on-disk** `models.json` (repo working tree), not GitHub.
- Brings each needed env in line with its yaml first: rebuilds a drifted env,
  builds a missing one (multi-GB, slow on first run).
- For catalog runs (`--model` / all), re-fetches each model's `inference.py` and
  `taxonomy.csv` fresh from HuggingFace before testing, so it always tests the
  live model code. `--model-dir` tests the local dir as-is (no refetch); use it
  for a model you are still editing.
- Downloads missing weights, then runs each model in its worker subprocess and
  compares the top-5 (labels + confidences) against the recorded baseline.

## Baselines

Baselines live in `backend/tests/data/model_expectations.json`, keyed by
model_id, with a `source`:
- `legacy`: generated from the old AddaxAI on the same crop (a faithful-port
  check).
- `reference`: this pipeline's own frozen output, for models with no legacy
  equivalent (record with `--record-reference`).

Comparison uses `CONF_TOLERANCE` (0.02) on confidence and stops at
`NEGLIGIBLE_CONF` (1e-3) so near-zero tail ranks do not flap. Baselines are
recorded on one OS and compared on others within that tolerance; that is why the
tolerance is loose, do not tighten it.

## Reading the table

Columns: model, name, env, device, status.
- device: `GPU` good; `CPU ok` means CPU-only here by design (in `EXPECTED_CPU`,
  or tensorflow-v2 on Windows which has no native GPU); `CPU!` an unexpected
  fallback (worth a look, not a fail).
- status: `PASS` (top-5 agrees), `FAIL` (ran but output diverged), `ERROR`
  (crashed / did not run), `RAN` (no baseline recorded), `SKIP` (weights not
  downloaded, with `--skip-missing`).

## Gotchas

- **Your model must be in the on-disk `models.json`** (and on the test box) or it
  is silently skipped. Commit/sync your catalog change first.
- **A new model shows `RAN` until you record a baseline** (`--record-reference`).
  That writes into `model_expectations.json`, which ships with the release.
- **First run is heavy**: env rebuilds (cu128 wheels are GBs) plus weight
  downloads. Reruns are fast.
- **The shared test image is a plumbing check, not an accuracy benchmark.** It is
  one out-of-region crop used to confirm the pipeline runs and is deterministic;
  do not read the predictions as model quality.
- **tensorflow-v1 GPU on Windows** needs the env's conda CUDA DLLs on PATH
  (handled in `custom_classification_model.py`); `tensorflow-v2` is CPU-only on
  Windows by design.

## Key files

- `backend/scripts/test_models.py`
- `backend/tests/data/model_expectations.json`
- `backend/app/ml/inference/custom_classification_model.py`
- `backend/app/ml/environment_manager.py` (env build / drift)
