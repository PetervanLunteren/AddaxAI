# Add a classification model to the zoo

Use this when a model author sends a new (or updated) camera-trap species
classifier to ship in AddaxAI. It is a recurring, gotcha-heavy job; follow the
steps and the gotcha list rather than re-deriving it.

Scope: classification (cls) models. Embedding (emb) models reuse the vendored
torch.hub pattern in step 2 (see `~/AddaxAI/models/emb/DINOV2-*`). Detection
models are out of scope.

## Mental model

- The catalog is `models.json` at the repo root, served from GitHub `main`, so
  committing an entry publishes it to every installed client. Read
  `backend/app/ml/schemas/model_manifest.py` for the fields.
- Each model has its own HuggingFace repo (`Addax-Data-Science/<model_id>` by
  default, see `resolve_hf_repo`) holding: the weights, an `inference.py`
  implementing `ModelInference`, a `taxonomy.csv`, and (for torch.hub models)
  the vendored architecture source.
- At analysis time the model runs as a one-shot subprocess in a micromamba env
  named by `manifest.env`, driven by
  `backend/app/ml/inference/custom_classification_model.py`.
- `backend/scripts/test_models.py` is the manual per-OS harness. It is the
  regression gate for this whole job.

## Steps

1. **Get the artifacts and identify the framework.** Weights file, the author's
   own inference/preprocessing code, the class list, the license. Note whether
   it is PyTorch (transformers / torch.hub / timm / torchvision / YOLOv8) or
   TensorFlow/Keras. That decides the env and the inference.py pattern.

2. **Write `inference.py` by copying the closest existing model.** Do not write
   from scratch. The required `ModelInference` interface (see
   `backend/templates/inference_template.py`): `__init__(model_dir, model_path)`,
   `check_gpu()`, `load_model()`, `get_crop(image, bbox)`,
   `get_classification(crop) -> [[name, conf], ...]` for all classes,
   `get_class_names() -> {"1": name, ...}` (1-indexed). Add the optional batch
   path (`get_tensor` + `classify_batch`) for a 5 to 15x GPU speedup; copy it
   from any example. Match the author's preprocessing exactly (resize size,
   interpolation, normalization, crop shape), then prove it in step 7.

   | framework | copy from (in `~/AddaxAI/models/cls/`) |
   |---|---|
   | transformers (Dinov2 etc., local `config.json`) | `AFR-DFV-v1/inference.py` |
   | vendored torch.hub (DINOv3, `source="local"`) | `AFR-DFV-v2/inference.py` |
   | timm backbone | `AWC135-AWC-v1/inference.py`, `EUR-DF-v1-3/inference.py` |
   | torchvision (EfficientNet etc.) | `SWUSA-SDZWA-v3/inference.py` |
   | YOLOv8 (ultralytics) | `NAM-ADS-v1/inference.py` |
   | Keras / TensorFlow | `TAS-BB-v1` (tf-v2), `PAM-SDZWA-v1` (tf-v1) |
   | SpeciesNet-style fx GraphModule | `HWI-ADS-v1/inference.py` |

   For a **vendored torch.hub** model (like DFV2): ship the arch source next to
   the weights (`hubconf.py` + the package dir), load with
   `torch.hub.load(model_dir, "<entry>", source="local", pretrained=False)`, and
   slim `hubconf.py` to only the one entry point you use (the full one pulls
   training/eval deps like termcolor). See `AFR-DFV-v2/README.md`.

3. **Check env fit with a load-smoke.** In the model's target env
   (`~/AddaxAI/envs/env-<name>/bin/python`), load the arch and strict-load the
   weights, confirm 0 missing / 0 unexpected keys and a forward pass. Envs:
   `pytorch` (most), `tensorflow-v1` (Keras 2.10, GPU on Win/Linux),
   `tensorflow-v2` (TF 2.16, CPU-only on Windows), `pywildlife` (PytorchWildlife).
   Prefer an env that already fits with no yaml change; adding a dep re-hashes
   the env and forces every user of it to rebuild (see the env yamls under
   `backend/app/ml/envs/`). A model needing a newer Python (e.g. DINOv3 needs
   3.10+) must gate on `min_app_version` (step 6).

4. **Strip training checkpoints.** If the `.pth` carries optimizer / scheduler /
   scaler state, re-save just what inference reads (usually `model_state_dict`
   and the label list). DFV2 went 1.0 GB to 343 MB this way. Document it in the
   model's README.

5. **Build `taxonomy.csv`.** Follow the `build-taxonomy-csv` skill. Reuse rows
   from a sibling model where class names overlap.

6. **Add the catalog entry to `models.json`.** Copy a sibling cls entry. Required
   fields: `model_id`, `friendly_name`, `env`, `model_fname`, `description`,
   `developer`, `info_url`, `min_app_version`; plus cls conventions `emoji`,
   `description_short`, `region` (one of global/africa/americas/asia/europe/
   oceania). Omit `hf_repo` to use the default org. Set
   `"torch_hub_model": "<entry>"` for vendored torch.hub models so
   `check_weights_ready` gates on `hubconf.py` being present.
   `min_app_version` is a real gate: set it to the release that ships the env
   this model needs, or old clients will offer a model their env cannot run.
   A model whose `taxonomy.csv` uses the `variant` column (classes below
   species, e.g. adult / juvenile) needs `min_app_version` at the first release
   with variant support: older apps ignore the column and show two classes
   with the same scientific name.

7. **Validate (the part that catches a botched port).**
   - Fidelity: run the author's own code and your `inference.py` on the same
     crop, confirm identical top-k. This is the ground truth when there is no
     legacy AddaxAI baseline.
   - Harness: `python backend/scripts/test_models.py --model-dir
     ~/AddaxAI/models/cls/<id> --record-reference`, then a plain `--model-dir`
     run must PASS. Records the reference baseline into
     `backend/tests/data/model_expectations.json`.
   - `ruff check app tests` and `pytest tests/ml/test_catalog_schema.py` (it
     validates the new entry) plus the full `pytest`.

8. **Hand off.** Assemble the model dir (weights + inference.py + taxonomy.csv +
   vendored source + LICENSE + README) as the HuggingFace upload folder; Peter
   uploads (no HF token in-session). Then verify the live repo: download it via
   the production path and run `test_models.py --model <id>`. Commit `models.json`
   and `model_expectations.json` with the release that ships the env.

## Gotchas (the expensive ones)

- **torch >= 2.6 flips `weights_only` to True.** A checkpoint pickling any
  non-tensor object (Compose, numpy scalar, omegaconf) fails to load. Add
  `weights_only=False` to that one `torch.load` (trusted first-party file).
- **Windows-pickled checkpoints** carry `WindowsPath` objects. Guard with
  `if platform.system() != "Windows": pathlib.WindowsPath = pathlib.PosixPath`.
- **ultralytics version drift.** Old YOLOv8 `.pt` files reference moved symbols
  (`yaml_load`) and pickle stale transforms. See the gated shims in
  `NAM/NZI/TKM inference.py`.
- **cp1252 on Windows.** A bare `open(labels_file)` mojibakes non-ASCII species
  names on Windows. Always `open(..., encoding="utf-8")` for any text read.
- **`mixed_float16` Keras checkpoints go NaN on CPU.** fp16 overflows off-GPU.
  Rebuild the model as float32 from its config, then load weights. See
  `PAM-SDZWA-v1/inference.py`. `set_global_policy` alone does not work.
- **tensorflow-v1 on Windows needs its conda CUDA DLLs on PATH.** Handled in
  `custom_classification_model.py` (`_worker_path_prefix`); the env carries
  cudatoolkit 11.2 + cudnn 8.1.
- **Baselines are device-dependent.** A reference baseline recorded on one OS is
  compared elsewhere within `CONF_TOLERANCE` (0.02). Never tighten it.
- **Vendored torch.hub entry points download pretrained weights by default.**
  Always pass `pretrained=False`; the strict weight load supplies everything and
  keeps inference offline.
- **The harness reads the on-disk `models.json`, not GitHub.** Your entry must
  be in the local working tree (and on the test box) or the model is silently
  skipped.

## Key files

- `models.json`, `backend/app/ml/schemas/model_manifest.py`
- `backend/templates/inference_template.py`
- `backend/app/ml/inference/custom_classification_model.py`,
  `classification_worker.py`
- `backend/scripts/test_models.py`, `backend/tests/data/model_expectations.json`
- `backend/app/ml/envs/<env>/<platform>/environment.yml`
- example models under `~/AddaxAI/models/cls/` (see the table above)
- `DEVELOPERS.md` sections "Creating a custom classification model" and "Label
  taxonomy and the hierarchical filter tree"
