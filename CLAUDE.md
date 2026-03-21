# AddaxAI — Developer Handbook

## What This App Does

AddaxAI is a desktop GUI application that helps field ecologists process camera trap images
using local computer vision. Users point it at a folder of images; it runs MegaDetector
(an object detection model) to find animals, people, and vehicles, then optionally runs a
species classification model on the detections. Results are written to JSON, and the app
provides postprocessing tools to separate images by detection type, export to CSV/XLSX/COCO,
generate pie charts and GPS heatmaps, and review detections in a human-in-the-loop (HITL)
workflow using LabelImg. The app supports English, Spanish, and French, runs fully offline,
and is packaged via PyInstaller for non-technical ecologists on Windows and macOS. The
classification models are heterogeneous — different models require different conda environments
and ML frameworks — so each model runs as a subprocess, often in an isolated conda env.

---

## What Was Refactored and Why

The application originally lived in a single ~11,200-line file (`AddaxAI_GUI.py`) with no
separation between business logic, UI, model deployment, data processing, and localization.
All state was managed via `global` declarations (35 of them). This made the codebase
impossible to unit test, extremely fragile to modify, and impenetrable for new contributors.

**Phase 1** extracted 58 pure backend functions (~1,378 lines) into 12 modules under
`addaxai/`. Each function was parameterized (globals like `AddaxAI_files` became explicit
arguments), then wired back into `AddaxAI_GUI.py` as drop-in replacements. The Phase 1
behavioral change log in the git history documents every case where behavior changed during
extraction (bug fixes, return type changes, stripped UI side-effects) — read it before
touching `models/registry.py`, `models/deploy.py`, or `processing/export.py`.

**Phase 2** replaced 662 inline `["English", "Spanish", "French"][lang_idx]` array lookups
with a proper i18n system: a `t("key")` function backed by three JSON files
(`addaxai/i18n/{en,es,fr}.json`, ~300 keys each). The `lang_idx` global was eliminated;
all language state now goes through `addaxai.i18n`.

**Phase 3** extracted 7 widget/dialog/tab classes from the monolith into `addaxai/ui/`:
`ProgressWindow`, `SpeciesSelectionFrame`, `InfoButton`/`CancelButton`/`GreyTopButton`,
`TextButtonWindow`, `CustomWindow`, `ModelInfoFrame`/`DonationPopupFrame`, help tab,
about tab, and the entire simple-mode window (`build_simple_mode()`).

**Phase 4** eliminated all global state. An `AppState` dataclass (`addaxai/core/state.py`)
now owns all 35 former globals — tkinter variables, cancel flags, deployment state, HITL
state, dropdown option lists, widget references, and caches. The instance is created once
after the root window is built and passed wherever needed. `SpeciesNetOutputWindow` was
also extracted as the final dialog class.

**Phase 5** added production-quality polish: full type annotations on all 39 `addaxai/`
modules (Python 3.8 compatible, using `typing` generics throughout); a `logging`
infrastructure (`addaxai/core/logging.py`) that writes to both stdout and
`AddaxAI_files/addaxai.log`, replacing all `print()` calls across the entire codebase;
and a GitHub Actions CI pipeline (`.github/workflows/test.yml`) that runs unit tests on
Python 3.9 and 3.11, `ruff` lint, and `mypy` type checking on every push.

---

## Repository Layout

```
AddaxAI_GUI.py              # Main entry point — still ~8,500 lines of GUI code
addaxai/
├── core/
│   ├── config.py           # load_global_vars / write_global_vars / load_model_vars_for
│   ├── logging.py          # setup_logging() — call once in main() with log_dir=AddaxAI_files
│   ├── paths.py            # Path resolution helpers
│   ├── platform.py         # OS detection, DPI scaling, Python interpreter lookup
│   └── state.py            # AppState dataclass — single instance owns all mutable state
├── models/
│   ├── deploy.py           # cancel_subprocess, switch_yolov5_version, imitate_object_detection
│   └── registry.py         # fetch_known_models, set_up_unknown_model, environment_needs_downloading
├── processing/
│   ├── annotations.py      # Pascal VOC / COCO / YOLO XML conversion
│   ├── export.py           # csv_to_coco
│   └── postprocess.py      # move_files, format_size
├── analysis/
│   └── plots.py            # fig2img, overlay_logo, calculate_time_span
├── i18n/
│   ├── __init__.py         # t("key") translation function, lang_idx(), i18n_set_language()
│   ├── en.json
│   ├── es.json
│   └── fr.json
├── hitl/
│   └── __init__.py         # stub — HITL data logic remains in AddaxAI_GUI.py for now
├── ui/
│   ├── widgets/
│   │   ├── buttons.py      # InfoButton, CancelButton, GreyTopButton
│   │   ├── frames.py       # MyMainFrame, MySubFrame, MySubSubFrame
│   │   └── species_selection.py  # SpeciesSelectionFrame (scrollable checkbox list)
│   ├── dialogs/
│   │   ├── custom_window.py       # CustomWindow (generic popup)
│   │   ├── download_progress.py   # EnvDownloadProgressWindow
│   │   ├── info_frames.py         # ModelInfoFrame, DonationPopupFrame
│   │   ├── patience.py            # PatienceWindow
│   │   ├── progress.py            # ProgressWindow (deploy + postprocess progress)
│   │   ├── speciesnet_output.py   # SpeciesNetOutputWindow
│   │   └── text_button.py         # TextButtonWindow
│   ├── advanced/
│   │   ├── about_tab.py    # write_about_tab()
│   │   └── help_tab.py     # write_help_tab(), HyperlinkManager
│   └── simple/
│       └── simple_window.py  # build_simple_mode() → returns dict of widget refs
└── utils/
    ├── files.py            # is_valid_float, get_size, shorten_path, natural_sort_key,
    │                       #   remove_ansi_escape_sequences, sort_checkpoint_files
    ├── images.py           # is_image_corrupted, get_image_timestamp, find_series_images, blur_box
    └── json_ops.py         # merge_jsons, append_to_json, get_hitl_var_in_json, etc.
```

**Why `AddaxAI_GUI.py` is still ~8,500 lines:** Everything that remains is tightly coupled
to tkinter widget construction and event wiring. The extractable surface is exhausted. The
deployment orchestrators (`start_deploy`, `deploy_model`, `classify_detections`) coordinate
subprocess spawning, progress updates, messagebox calls, and cancel handling in a way that
cannot be cleanly separated without introducing a callback/event bus first. The HITL window
(`open_hitl_settings_window`) is ~400 lines of widget construction alone. This line count
is normal for a ~40-dialog, 3-language desktop app with no UI framework abstraction layer.

---

## Development Setup

```bash
# Clone the fork
git clone https://github.com/TeodoroTopa/AddaxAI.git
cd AddaxAI
git checkout refactor/modularize

# Unit test environment (Python 3.14, no GUI deps needed)
python -m venv .venv
.venv/Scripts/pip install pytest numpy pandas requests Pillow ruff mypy

# Run unit tests
.venv/Scripts/python -m pytest tests/ -v \
  --ignore=tests/test_gui_smoke.py \
  --ignore=tests/test_gui_integration.py

# Run linter
.venv/Scripts/ruff check addaxai/

# Run type checker
.venv/Scripts/mypy addaxai/ --ignore-missing-imports --no-strict-optional

# GUI environment (Python 3.8, full deps including MegaDetector)
# This is the installed app's conda env — do not modify it
C:\Users\Topam\AddaxAI_files\envs\env-base\python.exe

# Launch the GUI for manual testing
C:\Users\Topam\AddaxAI_files\envs\env-base\python.exe dev_launch.py

# Run GUI integration tests (boots real GUI, ~15s per test)
C:\Users\Topam\AddaxAI_files\envs\env-base\python.exe -m pytest tests/test_gui_integration.py -v

# Run GUI smoke test (starts GUI, waits 10s, asserts no crash)
C:\Users\Topam\AddaxAI_files\envs\env-base\python.exe -m pytest tests/test_gui_smoke.py -v
```

**Remotes:**
- `origin` → `TeodoroTopa/AddaxAI` (fork — push here)
- `upstream` → `PetervanLunteren/AddaxAI` (original — pull updates from here)

---

## Test Suite

Tests are split by runtime because the GUI requires a specific conda env that is not
available in CI.

**Unit tests** (`tests/test_*.py`, excluding GUI tests): Run with `.venv` Python 3.14.
Fast (~12s). Import `addaxai/` modules directly. No tkinter, no conda, no models.
**Current count: 325 passing, 9 skipped** (optional deps: cv2, matplotlib, customtkinter).

**GUI integration tests** (`tests/test_gui_integration.py`): Run with env-base Python 3.8.
The `tests/gui_test_runner.py` harness `exec()`s `AddaxAI_GUI.py` with a patched
`AddaxAI_files` path, suppresses `main()`, initializes frame states manually, then
schedules each test via `root.after()`. The test writes results to a temp JSON file and
calls `root.quit()`. The pytest file reads that JSON and asserts. **Current count: 8 passing.**

| Test | What it covers |
|------|----------------|
| `test_language_cycling` | EN→ES→FR→EN, checks 12 advanced + 5 simple widget texts per language |
| `test_mode_switching` | Advanced↔simple toggle, window visibility |
| `test_folder_selection` | `update_frame_states()` on folder change |
| `test_model_dropdown_population` | `update_model_dropdowns()` populates `state.dpd_options_*` |
| `test_toggle_frames` | sep/vis postprocessing frame toggle callbacks |
| `test_reset_values` | `reset_values()` reverts 5 vars to defaults |
| `test_deploy_validation` | `start_deploy()` shows error on empty folder (doesn't crash) |
| `test_state_attributes` | 24 AppState attributes have correct types/defaults after boot |

**GUI smoke test** (`tests/test_gui_smoke.py`): Launches GUI as subprocess, waits 10s,
asserts process is still alive. **1 passing.**

**What is NOT tested:** The actual MegaDetector/classification subprocess pipeline, HITL
workflow, postprocessing file moves, results viewer, export to XLSX, and SpeciesNet
deployment. These require real models and are too slow for automated CI. See the "Future
Work" section for how to address this.

**CI** (`.github/workflows/test.yml`): Runs on every push to `refactor/modularize` and
PRs to `main`. Three jobs: unit tests (Python 3.9 + 3.11), ruff lint, mypy type check.
GUI/integration/smoke tests are excluded — they require env-base and a display.

---

## Development Conventions

- **TDD:** Write tests first, implement to make them pass, run full suite, commit.
- **One commit per logical step** — small, immediately pushable. Conventional commit
  prefixes: `feat`, `fix`, `refactor`, `ci`, `docs`, `chore`.
- **Extraction rule:** When moving a function out of `AddaxAI_GUI.py`, parameterize all
  globals (e.g. `AddaxAI_files` → `base_path`, `var_choose_folder.get()` → `base_folder`).
  Do not change behavior — pure mechanical moves only. Document exceptions in the commit message.
- **Type hints:** Use `typing` module generics throughout for Python 3.8 compatibility
  (`List`, `Dict`, `Optional`, not `list[str]`, `dict[str, Any]`, `X | None`).
  Use `Any` for all tkinter/customtkinter widget parameters.
- **Logging:** Use `logging.getLogger(__name__)` at the top of each module. Map levels as:
  function traces → `DEBUG`, subprocess output → `INFO`, warnings → `WARNING`,
  caught exceptions → `ERROR(..., exc_info=True)`.
- **No `global` declarations:** All mutable state goes through `AppState`. If you find
  yourself adding a global, add it to `AppState` instead.
- **Update CLAUDE.md** after any significant change to architecture, conventions, or status.

---

## Watchouts and Known Issues

**Python version split:** Unit tests run on Python 3.14 (`.venv`). The GUI runs on Python
3.8 (`env-base`). All `addaxai/` code must be Python 3.8 compatible — use `typing` generics,
not built-in generic syntax. `mypy` is configured with `--ignore-missing-imports` and
`--no-strict-optional`; tightening these will reveal real issues.

**Phase 1 behavioral changes:** Several functions behave differently from the original:
- `utils/files.py`: `sort_checkpoint_files` uses index `[2]` not `[1]` (bug fix — original would sort incorrectly).
- `utils/json_ops.py`: `get_hitl_var_in_json` returns `"never-started"` gracefully instead of crashing when no metadata key exists.
- `processing/export.py`: `csv_to_coco` uses `math.isnan()` instead of `type(val) == float` (bug fix — original treated any float as NA date).
- `models/deploy.py`: `cancel_subprocess` no longer re-enables UI buttons or closes the progress window — the caller in `AddaxAI_GUI.py` handles that.
- `models/registry.py`: `environment_needs_downloading` returns a `tuple` not a `list`. `set_up_unknown_model` silently swallows download errors — should be improved.

**Flaky test:** `test_non_tk_attr_instantiated` occasionally skips depending on Tk
availability in the test environment. The count fluctuates between 325/9 and 326/8 skipped.
This is pre-existing, not introduced by the refactoring.

**customtkinter import pattern:** All UI modules use a try/except fallback pattern so they
can be imported without customtkinter installed (enabling unit tests). This causes `mypy`
`no-redef` and `valid-type`/`misc` errors on the stub class definitions and subclasses —
suppressed with `# type: ignore` comments. Do not remove these.

**`HITL` and `analysis` modules are stubs:** `addaxai/hitl/__init__.py` is nearly empty;
the HITL data logic remains in `AddaxAI_GUI.py`. `addaxai/analysis/` has only `plots.py`;
maps.py was planned but not needed.

**Model adapters:** `classification_utils/model_types/` is untouched. Each adapter runs
as a subprocess in its own conda env with a different ML framework. The boilerplate
duplication is intentional for subprocess isolation — don't consolidate it.

---

## Ideas for Future Development

### More substantial testing and CI/CD
The most impactful next step. Commit a small fixture image set (~5 images, ~1MB) and a
tiny model checkpoint (~30MB, e.g. MDv5-tiny). Add `test_deploy_pipeline.py` that:
runs detection on the fixtures, asserts the output JSON structure, and diffs against a
golden file. Run this on a self-hosted GitHub Actions runner (with models pre-downloaded)
on every merge to `main`. This would catch 80% of behavioral regressions. A monthly canary
run against a curated 100-image set with known labels would track model integration quality
over time.

### Cloud inference backend
The architecture is ready for this. `models/deploy.py` is the right place to introduce an
`InferenceBackend` interface with `LocalBackend` and `CloudBackend` implementations. The
intended model: MegaDetector runs locally (fast, no uploads), classification runs in the
cloud (upload crops only, ~10KB per crop). Candidate hosting: HuggingFace Inference
Endpoints or Replicate (users bring their own API key and pay for compute). This eliminates
the multi-GB conda environment download for most users while keeping detection offline.

### User accounts and model management
A login system (OAuth via GitHub/Google, or email/password against a hosted backend) would
enable: syncing user settings and selected models across machines, a model registry hosted
centrally instead of per-user downloads, usage analytics (which models are popular, which
species are being detected where), and a future paid tier for cloud inference credits.
AddaxAI already has a donation popup — a freemium model is a natural next step.

### Additional languages
The i18n system (`addaxai/i18n/`) makes adding languages cheap: create a new JSON file,
add the language index to `i18n/__init__.py`, and update the language dropdown. Portuguese
and German would cover the largest remaining camera trap user communities. The main cost
is translation quality review, not engineering work.

### UI framework migration
The Phase 1–4 extraction means all business logic is now in `addaxai/` and completely
framework-agnostic. Migrating the UI layer from customtkinter to PySide6 (or Dear PyGui
for a lighter option) is now a contained effort limited to `AddaxAI_GUI.py` and
`addaxai/ui/`. PySide6 would enable a proper MVC architecture, better theming, and
native-feeling widgets on macOS — which currently has rough edges with customtkinter.

### Faster inference
MegaDetector is already fast, but classification is a bottleneck for large deployments.
Options: batch inference (the current pipeline processes one image at a time for many
models), ONNX export for models that support it (eliminates conda env overhead), and
async/concurrent processing (detect N images while classifying the previous N). The
subprocess architecture already isolates classification — a concurrent queue with worker
processes is a natural extension.

### HITL improvements
The human-in-the-loop workflow is the most complex remaining area. It currently relies on
LabelImg (a separate tool) for annotation. A native annotation UI built into AddaxAI
(using tkinter canvas or a web-based approach via a local Flask server) would eliminate
the LabelImg dependency, enable real-time sync between reviewed images and the results
JSON, and allow batch-review workflows (e.g., "accept all detections above 0.9 conf").

---

## Phase 6 Recommendations: Open-Source Readiness

The refactoring phases (1–5) established solid foundations: extracted business logic,
centralized state, added type hints, logging, and i18n. The next phase focuses on making
the project contributor-friendly, robustly tested, and ready for external integration.
Work items are ordered by impact — each builds on the previous.

### 6.1 — Project packaging and contributor onboarding

**Problem:** No `pyproject.toml`, no `requirements.txt`, no issue templates, no PR template.
A new contributor clones the repo and has no standard way to install dependencies or
understand the contribution workflow.

**Work items:**
1. Create `pyproject.toml` with project metadata, dependencies, and optional dependency
   groups (`[project.optional-dependencies]`): `test` (pytest, ruff, mypy), `gui`
   (customtkinter, Pillow, etc.), `dev` (all of the above). This replaces the ad-hoc
   `pip install` commands scattered in CLAUDE.md and CI.
2. Create `requirements.txt` (pinned) for reproducible CI builds, generated from
   `pip freeze` of the test environment.
3. Add `.github/ISSUE_TEMPLATE/bug_report.yml` and `feature_request.yml` (structured
   YAML forms, not freeform markdown) so bug reports include OS, Python version, model
   used, and steps to reproduce.
4. Add `.github/pull_request_template.md` with sections: Summary, Test Plan, Checklist
   (ran tests, ran linter, updated CLAUDE.md if applicable).
5. Add a `DEVELOPMENT.md` (or expand CONTRIBUTING.md) with a contributor-oriented setup
   guide: fork → clone → create venv → install `[test]` extras → run tests → open PR.
   Keep CLAUDE.md as the internal developer handbook; DEVELOPMENT.md is the public face.

### 6.2 — Fix and extend CI/CD

**Problem:** CI currently runs only 2 jobs (unit tests + ruff lint). CLAUDE.md claims
3 jobs including mypy, but the mypy job does not exist in `test.yml`. No coverage
reporting, no pre-commit hooks, no branch protection guidance.

**Work items:**
1. Add a `mypy` job to `.github/workflows/test.yml` — this was documented as done in
   CLAUDE.md but is actually missing. Run `mypy addaxai/ --ignore-missing-imports
   --no-strict-optional` on Python 3.11. Fix the CLAUDE.md claim.
2. Add `pytest-cov` and generate a coverage report. Upload to Codecov or Coveralls.
   Set a coverage floor (current coverage is ~75–80% for `addaxai/`; start there and
   ratchet up). Display the badge in README.
3. Add a `pre-commit` configuration (`.pre-commit-config.yaml`) with ruff, mypy, and
   trailing-whitespace hooks. Document in DEVELOPMENT.md. This catches issues before
   they reach CI.
4. Add branch protection rules documentation: require CI pass + 1 review before merge
   to `main`. Contributors should target `main` (not `refactor/modularize`).
5. Add a `test-gui-smoke` job that runs on `ubuntu-latest` with `xvfb-run` — the smoke
   test only needs tkinter (not customtkinter) and can catch import-time crashes.
   Conditionally install `python3-tk` via apt.

### 6.3 — JSON schema validation and data contracts

**Problem:** Configuration files (`global_vars.json`, `variables.json`) and recognition
output JSON have no schema definitions. Unknown keys are silently ignored. This makes it
impossible for external tools to integrate reliably, and contributors can introduce
malformed configs without tests catching them.

**Work items:**
1. Write JSON Schema files for: `global_vars.json`, `variables.json` (per-model config),
   and the MegaDetector recognition output format. Place in `addaxai/schemas/`.
2. Add a `validate_config()` function in `addaxai/core/config.py` that validates loaded
   JSON against the schema. Use `jsonschema` (lightweight, pure Python). Warn on unknown
   keys instead of silently ignoring.
3. Add a `validate_recognition_output()` function for the detection/classification JSON.
   This enables external tools (R packages, cloud dashboards, other Python apps) to
   validate their output before feeding it to AddaxAI.
4. Commit example/fixture JSON files in `tests/fixtures/` — valid `global_vars.json`,
   valid `variables.json`, valid recognition output, and intentionally invalid versions
   of each. Use these in schema validation tests.
5. Document the JSON formats in a `docs/data-formats.md` file aimed at integrators
   (not contributors). Include field descriptions, value ranges, and examples.

### 6.4 — Model adapter protocol and plugin documentation

**Problem:** Adding a new classification model requires creating files in 3 locations
(`models/cls/`, `classification_utils/model_types/`, and model config JSON) with no
template, no interface definition, and no documentation. Contributors reverse-engineer
existing adapters.

**Work items:**
1. Define an `InferenceBackend` protocol (abstract base class or `typing.Protocol`) in
   `addaxai/models/backend.py`. Methods: `detect(image_paths, config) → ResultJSON`,
   `classify(crops, config) → ClassificationResult`. This does not replace the subprocess
   architecture — it documents the contract that adapters must fulfill.
2. Create a template adapter in `classification_utils/model_types/_template/` with
   annotated `classify_detections.py`, `variables.json`, and a README explaining each
   field.
3. Write `docs/adding-a-model.md` — step-by-step guide: create directory, fill in
   variables.json, implement classify_detections.py, test with fixture images.
4. Add a CI test that validates all `variables.json` files under `models/` against the
   schema from 6.3. This catches broken model configs before they ship.

### 6.5 — Integration test infrastructure with fixtures

**Problem:** The test suite covers extracted `addaxai/` functions well (325 tests) but
cannot test the actual detection/classification pipeline, postprocessing file moves,
or export workflows. These are the features most likely to regress.

**Work items:**
1. Commit a small fixture dataset in `tests/fixtures/images/` (~5 camera trap images,
   ~1MB total, Creative Commons licensed). Include one with EXIF GPS, one without, one
   corrupted, one video frame.
2. Commit a golden-file recognition JSON (`tests/fixtures/golden_output.json`) — the
   expected detection result for the fixture images. This does not require a real model;
   it can be hand-crafted to match the schema from 6.3.
3. Write `tests/test_postprocess_pipeline.py` — given the golden JSON and fixture images,
   test `move_files()`, CSV export, COCO export, and verify output structure. These are
   pure-function tests that run without models or GUI.
4. Write `tests/test_export_roundtrip.py` — export to CSV, re-import, verify lossless.
   Export to COCO JSON, validate against schema. Export to XLSX, read back with openpyxl.
5. Add a `self-hosted` CI job label for future use: when a self-hosted runner with GPU
   and models is available, run `test_deploy_pipeline.py` against real MegaDetector.
   Document the runner setup in DEVELOPMENT.md.

### 6.6 — Event bus for UI decoupling

**Problem:** The deploy/classify orchestrators (`start_deploy`, `deploy_model`,
`classify_detections`) are 400+ lines each in `AddaxAI_GUI.py`. They interleave
subprocess management, progress updates, messagebox calls, and cancel handling. This
is the single largest barrier to further modularization and testability.

**Work items:**
1. Introduce a lightweight event bus in `addaxai/core/events.py`. A simple
   publish/subscribe pattern: `emit("deploy.progress", pct=50)`,
   `on("deploy.progress", callback)`. No external dependencies — use stdlib `queue`
   or a simple listener dict.
2. Refactor `deploy_model()` to emit events (`deploy.started`, `deploy.progress`,
   `deploy.image_complete`, `deploy.error`, `deploy.finished`) instead of directly
   calling `progress_window.update()` and `messagebox.showerror()`.
3. The GUI subscribes to these events and updates widgets. This lets tests subscribe
   to the same events and assert the deployment sequence without a GUI.
4. Repeat for `classify_detections()` and `start_postprocess()`.
5. This is the architectural prerequisite for cloud inference (6.4's `InferenceBackend`)
   and for the UI framework migration to PySide6. Prioritize it over those features.

### 6.6b — Break apart AddaxAI_GUI.py

**Problem:** `AddaxAI_GUI.py` is 8,500 lines. The filename itself (`_GUI` suffix,
inconsistent casing) looks unprofessional for an open-source project. The CLAUDE.md
"Why `AddaxAI_GUI.py` is still ~8,500 lines" section explains that the *business logic*
extraction surface is exhausted — but the remaining UI code can still be split by feature
area once the event bus (6.6) breaks the coupling.

**Work items:**
1. Rename `AddaxAI_GUI.py` → `addaxai/app.py` (or `addaxai/__main__.py` for
   `python -m addaxai` support). Update `main.py`, `dev_launch.py`, PyInstaller configs,
   build workflows, and test harnesses. This is a high-touch rename — do it as a single
   focused commit.
2. Define view protocols for each feature area in `addaxai/ui/protocols.py` using
   `typing.Protocol`. These are the contracts between orchestration logic and the UI:
   - `DeployView`: `show_progress(pct, message)`, `show_error(msg)`,
     `set_model_list(models)`, `on_cancel(callback)`, `reset()`
   - `PostprocessView`: `show_progress(pct, message)`, `show_error(msg)`,
     `show_completion(summary)`
   - `HITLView`: `load_annotations(data)`, `show_image(path, boxes)`,
     `on_save(callback)`
   - `ResultsView`: `display(recognition_json)`, `set_filters(species, confidence)`
   The protocols define *what* the UI must do, not *how*. No tkinter types appear in
   any protocol signature — only plain Python types, dataclasses, and callbacks.
3. After 6.6's event bus is in place, split `addaxai/app.py` by feature area. Each
   module implements its protocol with tkinter/customtkinter:
   - `addaxai/ui/deploy_tab.py` — implements `DeployView`, subscribes to deploy events
   - `addaxai/ui/postprocess_tab.py` — implements `PostprocessView`
   - `addaxai/ui/hitl_window.py` — implements `HITLView`
   - `addaxai/ui/results_viewer.py` — implements `ResultsView`
   - `addaxai/app.py` — main window construction, menu bar, tab assembly, wires
     concrete view implementations to the event bus
4. Target: `app.py` should shrink to ~1,500 lines (window setup + tab wiring). Each
   feature module should be 500–1,500 lines. No module over 2,000 lines.
5. This abstraction is what makes a future UI framework migration (PySide6, web-based)
   a contained effort: write new implementations of the same protocols, swap them in
   at the wiring layer in `app.py`. The orchestration logic, event bus, and business
   logic modules never change. Without these protocols, a migration means rewriting
   every feature module — with them, it means reimplementing a known interface.

### 6.7 — REST API layer for external integration

**Problem:** AddaxAI is a closed desktop app. External tools (R packages, web dashboards,
mobile apps, cloud pipelines) cannot trigger detection, retrieve results, or monitor
progress without the GUI.

**Work items:**
1. Add a lightweight local API server in `addaxai/api/server.py` using FastAPI (or
   Flask for Python 3.8 compat). Endpoints: `POST /detect` (submit folder, return job
   ID), `GET /jobs/{id}` (poll status + progress), `GET /jobs/{id}/results` (return
   recognition JSON), `GET /models` (list available models).
2. The API server reuses the same `addaxai/` modules — it is an alternative entry point
   alongside the GUI, not a separate codebase. Both GUI and API call the same
   `deploy_model()` function (via the event bus from 6.6).
3. Add OpenAPI/Swagger docs (FastAPI generates these automatically). This is the
   integration contract for external developers.
4. This is a medium-term goal — it depends on the event bus (6.6) being in place so
   that deployment can run without a GUI event loop. Start with a read-only API
   (`GET /models`, `GET /results`) that works today, then add write endpoints after 6.6.

### 6.8 — Developer experience polish

**Problem:** Small friction points that individually are minor but collectively discourage
contributions.

**Work items:**
1. Add a `Makefile` (or `justfile`) with common commands: `make test`, `make lint`,
   `make typecheck`, `make test-gui`, `make dev` (launch GUI). Reduces cognitive load
   for new contributors.
2. Add `mypy.ini` (or `[tool.mypy]` in `pyproject.toml`) so mypy config is in version
   control, not just in CLAUDE.md prose.
3. Add `py.typed` marker to `addaxai/` so downstream consumers get type information.
4. Add GitHub Actions status badges to README: tests, lint, coverage.
5. Create a `CHANGELOG.md` starting from the refactoring phases. Use Keep a Changelog
   format. This gives contributors context on what changed and when.
6. Add `.editorconfig` for consistent formatting across editors (indent style, line
   endings, trailing whitespace).

### Implementation order

The items above are ordered by dependency and impact. A practical execution sequence:

| Priority | Item | Depends on | Estimated scope |
|----------|------|------------|-----------------|
| **P0** | 6.1 — Packaging + onboarding | — | 1–2 sessions |
| **P0** | 6.2 — Fix CI/CD | 6.1 (for deps) | 1 session |
| **P1** | 6.3 — JSON schemas | — | 2–3 sessions |
| **P1** | 6.5 — Integration test fixtures | 6.3 (for schemas) | 2–3 sessions |
| **P1** | 6.8 — DX polish | 6.1 (for pyproject) | 1 session |
| **P2** | 6.4 — Model adapter protocol | 6.3 (for schemas) | 2 sessions |
| **P2** | 6.6 — Event bus | — | 3–4 sessions |
| **P2** | 6.6b — Break apart GUI file | 6.6 (for decoupling) | 2–3 sessions |
| **P3** | 6.7 — REST API | 6.6 (for headless deploy) | 3–4 sessions |

P0 items should be done before merging `refactor/modularize` into `main` and inviting
contributors. P1 items make the first wave of contributions productive. P2/P3 items
unlock the architectural future (cloud inference, external integrations, UI migration).

---

## Phase 6 — Step-by-Step Implementation Plan

This section contains exact instructions for implementing each Phase 6 item. Each step
specifies the branch, files to create/modify, success criteria, and when to commit. Follow
these instructions literally — do not improvise or skip steps.

**Branch strategy:**
- All P0 work happens on `refactor/modularize` (the current branch).
- After P0 is done, merge `refactor/modularize` → `main` via PR.
- All P1/P2/P3 work happens on feature branches off `main`, named
  `phase6/<item>` (e.g. `phase6/json-schemas`, `phase6/event-bus`).
- Each feature branch merges to `main` via PR when its success criteria pass.

---

## Phase 6 Progress Tracker

**Current Status:** P0 Complete, Ready for merge to main (2026-03-21)

### P0 — Pre-merge work (must complete before `refactor/modularize` → `main`)

| Step | Task | Status | Date | Notes |
|------|------|--------|------|-------|
| 1 | Create `pyproject.toml` | ✅ Done | 2026-03-21 | Commit: 6199c8ee |
| 2 | Add mypy job to CI | ✅ Done | 2026-03-21 | Commit: 52ce500b |
| 3 | Add coverage reporting to CI | ✅ Done | 2026-03-21 | Commit: 52ce500b |
| 4 | Add GitHub issue/PR templates | ✅ Done | 2026-03-21 | Commit: 4cc11190 |
| 5 | Add developer tooling files | ✅ Done | 2026-03-21 | Commit: f8a60288 |
| 6 | Add pre-commit configuration | ✅ Done | 2026-03-21 | Commit: 0babd0a8 |
| 7 | Merge to main (P0 gate) | ⏳ In Progress | 2026-03-21 | Ready for PR |

### P1 — High-impact foundation work

| Step | Task | Status | Date | Notes |
|------|------|--------|------|-------|
| 8 | JSON schema validation (6.3) | ⏳ Pending | — | Branch: `phase6/json-schemas` after merge |
| 9 | Integration test fixtures (6.5) | ⏳ Pending | — | Branch: `phase6/test-fixtures` |
| 10 | Model adapter protocol (6.4) | ⏳ Pending | — | Branch: `phase6/model-protocol` |
| 11 | Event bus infrastructure (6.6) | ⏳ Pending | — | Branch: `phase6/event-bus` (multi-commit) |

### P2 — Architectural improvements

| Step | Task | Status | Date | Notes |
|------|------|--------|------|-------|
| 12 | View protocols (6.6b) | ⏳ Pending | — | Branch: `phase6/view-protocols` |
| 13 | Rename & break apart GUI (6.6b) | ⏳ Pending | — | Branch: `phase6/gui-restructure` (high-risk) |
| 14 | REST API layer (6.7) | ⏳ Pending | — | Branch: `phase6/rest-api` |

---

### Step 1: Create `pyproject.toml` (part of 6.1)

**Branch:** `refactor/modularize`

1. Create `pyproject.toml` at the repo root with this content:

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "addaxai"
version = "5.0.0"
description = "Camera trap image classification for ecologists"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.8"
dependencies = [
    "numpy",
    "pandas",
    "requests",
    "Pillow",
]

[project.optional-dependencies]
test = [
    "pytest",
    "pytest-cov",
    "ruff",
    "mypy",
]
gui = [
    "customtkinter",
]
dev = [
    "addaxai[test,gui]",
    "pre-commit",
]

[tool.mypy]
python_version = "3.8"
ignore_missing_imports = true
no_strict_optional = true
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--ignore=tests/test_gui_smoke.py --ignore=tests/test_gui_integration.py --ignore=tests/gui_test_runner.py"
```

2. Verify: run `.venv/Scripts/python -m pytest` (no extra flags needed — `pyproject.toml`
   provides defaults). All 325+ tests pass.

3. Commit: `chore: add pyproject.toml with project metadata and tool config`

---

### Step 2: Add mypy job to CI (part of 6.2)

**Branch:** `refactor/modularize`

1. Edit `.github/workflows/test.yml`. Add this job after the `lint` job:

```yaml
  typecheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - run: pip install mypy numpy pandas requests Pillow

      - name: Type-check addaxai/
        run: mypy addaxai/ --ignore-missing-imports --no-strict-optional
```

2. Also update the `test` job's dependency install step to use the optional deps:

```yaml
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pytest pytest-cov numpy pandas requests Pillow
```

3. Verify locally: run `.venv/Scripts/mypy addaxai/ --ignore-missing-imports --no-strict-optional`.
   Must exit 0 with no errors.

4. Commit: `ci: add mypy type-checking job to CI pipeline`

---

### Step 3: Add coverage reporting to CI (part of 6.2)

**Branch:** `refactor/modularize`

1. Edit `.github/workflows/test.yml`. In the `test` job, change the pytest run step to:

```yaml
      - name: Run unit tests with coverage
        run: |
          python -m pytest tests/ -v \
            --ignore=tests/test_gui_smoke.py \
            --ignore=tests/test_gui_integration.py \
            --ignore=tests/gui_test_runner.py \
            --cov=addaxai --cov-report=xml --cov-report=term-missing
```

2. Add a step after the test step to upload coverage (only on one Python version to
   avoid duplicate uploads):

```yaml
      - name: Upload coverage
        if: matrix.python-version == '3.11'
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
          fail_ci_if_error: false
```

3. Commit: `ci: add pytest-cov coverage reporting and Codecov upload`

---

### Step 4: Add GitHub issue and PR templates (part of 6.1)

**Branch:** `refactor/modularize`

1. Create `.github/ISSUE_TEMPLATE/bug_report.yml`:

```yaml
name: Bug Report
description: Report a bug in AddaxAI
labels: ["bug"]
body:
  - type: input
    id: os
    attributes:
      label: Operating System
      placeholder: "e.g. Windows 11, macOS 14.2"
    validations:
      required: true
  - type: input
    id: python-version
    attributes:
      label: Python Version
      placeholder: "e.g. 3.8.19 (env-base)"
    validations:
      required: true
  - type: input
    id: model
    attributes:
      label: Model Used
      placeholder: "e.g. MegaDetector v5a, DeepFaune v1.4"
  - type: textarea
    id: description
    attributes:
      label: What happened?
      description: Describe the bug. Include error messages if any.
    validations:
      required: true
  - type: textarea
    id: steps
    attributes:
      label: Steps to reproduce
      description: Minimal steps to trigger the bug.
    validations:
      required: true
  - type: textarea
    id: expected
    attributes:
      label: Expected behavior
```

2. Create `.github/ISSUE_TEMPLATE/feature_request.yml`:

```yaml
name: Feature Request
description: Suggest a new feature or enhancement
labels: ["enhancement"]
body:
  - type: textarea
    id: description
    attributes:
      label: What would you like?
      description: Describe the feature and why it would be useful.
    validations:
      required: true
  - type: textarea
    id: alternatives
    attributes:
      label: Alternatives considered
      description: Any workarounds or alternative approaches you've tried.
```

3. Create `.github/pull_request_template.md`:

```markdown
## Summary
<!-- 1-3 bullet points describing what this PR does and why -->

## Test plan
<!-- How did you verify this works? Check all that apply: -->
- [ ] Unit tests pass (`pytest tests/`)
- [ ] Lint passes (`ruff check addaxai/`)
- [ ] Type check passes (`mypy addaxai/ --ignore-missing-imports --no-strict-optional`)
- [ ] GUI smoke test passes (manual or `pytest tests/test_gui_smoke.py`)
- [ ] Tested manually in the GUI

## Checklist
- [ ] No new `global` declarations (use `AppState` instead)
- [ ] Type hints use `typing` generics (Python 3.8 compatible)
- [ ] Updated CLAUDE.md if architecture or conventions changed
```

4. Commit: `chore: add GitHub issue templates and PR template`

---

### Step 5: Add developer tooling files (part of 6.8)

**Branch:** `refactor/modularize`

1. Create `.editorconfig` at repo root:

```ini
root = true

[*]
indent_style = space
indent_size = 4
end_of_line = lf
charset = utf-8
trim_trailing_whitespace = true
insert_final_newline = true

[*.{yml,yaml}]
indent_size = 2

[*.{json,toml}]
indent_size = 2

[Makefile]
indent_style = tab
```

2. Create `Makefile` at repo root (use tabs for indentation, not spaces):

```makefile
.PHONY: test lint typecheck dev test-gui test-smoke

VENV_PYTHON = .venv/Scripts/python
GUI_PYTHON = C:/Users/Topam/AddaxAI_files/envs/env-base/python.exe

test:
	$(VENV_PYTHON) -m pytest tests/ -v

lint:
	$(VENV_PYTHON) -m ruff check addaxai/

typecheck:
	$(VENV_PYTHON) -m mypy addaxai/ --ignore-missing-imports --no-strict-optional

test-gui:
	$(GUI_PYTHON) -m pytest tests/test_gui_integration.py -v

test-smoke:
	$(GUI_PYTHON) -m pytest tests/test_gui_smoke.py -v

dev:
	$(GUI_PYTHON) dev_launch.py

all: lint typecheck test
```

3. Create `addaxai/py.typed` as an empty file (zero bytes). This is a PEP 561 marker
   that tells type checkers this package ships inline type annotations.

4. Commit: `chore: add .editorconfig, Makefile, and py.typed marker`

---

### Step 6: Add pre-commit configuration (part of 6.2)

**Branch:** `refactor/modularize`

1. Create `.pre-commit-config.yaml` at repo root:

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.0
    hooks:
      - id: ruff
        args: [--fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        args: [--ignore-missing-imports, --no-strict-optional]
        additional_dependencies: [numpy, pandas-stubs, types-requests]
        files: ^addaxai/
```

2. Do NOT run `pre-commit install` — that modifies the user's local git hooks.
   Just commit the config file so contributors can opt in.

3. Commit: `chore: add pre-commit configuration`

---

### Step 7: Merge `refactor/modularize` → `main` (P0 gate)

**This is the merge point.** All P0 work is complete.

1. Open a PR from `refactor/modularize` → `main` on `TeodoroTopa/AddaxAI`.
   Title: `refactor: complete Phases 1–6 (P0) — modular architecture + CI + DX`
   Body: summarize Phases 1–5 (already done) plus the P0 items from Phase 6
   (pyproject.toml, CI fixes, templates, tooling).

2. Verify CI passes on the PR (unit tests, lint, mypy).

3. Merge the PR. From this point forward, `main` is the base branch for all new work.

4. After merge, delete the `refactor/modularize` branch.

---

### Step 8: JSON schema validation (6.3)

**Branch:** Create `phase6/json-schemas` off `main`.

1. Create directory `addaxai/schemas/`.

2. Create `addaxai/schemas/__init__.py` (empty file).

3. Create `addaxai/schemas/global_vars.schema.json`. To build this schema, read
   `addaxai/core/config.py` — the `load_global_vars()` function documents every key
   it reads from `global_vars.json`. Also read a real `global_vars.json` file at
   `C:\Users\Topam\AddaxAI_files\AddaxAI\global_vars.json` (if accessible) for
   reference values. The schema should define every known key with its type and a
   description. Set `"additionalProperties": false` so unknown keys are flagged.

4. Create `addaxai/schemas/model_vars.schema.json`. To build this, read
   `addaxai/core/config.py` — the `load_model_vars_for()` function documents the keys.
   Also read any `variables.json` under `classification_utils/model_types/` for examples.

5. Create `addaxai/schemas/recognition_output.schema.json`. The MegaDetector output
   format is: `{"images": [{"file": str, "detections": [{"category": str, "conf": float,
   "bbox": [float, float, float, float]}]}], "detection_categories": {"1": "animal", ...}}`.
   Read `addaxai/utils/json_ops.py` for the exact keys the app reads from this JSON.

6. Create `addaxai/schemas/validate.py`:

```python
"""JSON schema validation for AddaxAI configuration and output files."""

import json
import logging
import os
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

_SCHEMA_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_schema(name: str) -> Dict[str, Any]:
    path = os.path.join(_SCHEMA_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_global_vars(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate global_vars.json data against schema.

    Returns (is_valid, list_of_error_messages).
    Does NOT require jsonschema — uses manual validation against the schema
    definition so there is no added dependency.
    """
    schema = _load_schema("global_vars.schema.json")
    errors = []
    props = schema.get("properties", {})

    # Check for unknown keys
    for key in data:
        if key not in props:
            errors.append(f"Unknown key: '{key}'")

    # Check types of known keys
    for key, prop_def in props.items():
        if key in data:
            expected_type = prop_def.get("type")
            value = data[key]
            if expected_type == "string" and not isinstance(value, str):
                errors.append(f"Key '{key}': expected string, got {type(value).__name__}")
            elif expected_type == "number" and not isinstance(value, (int, float)):
                errors.append(f"Key '{key}': expected number, got {type(value).__name__}")
            elif expected_type == "boolean" and not isinstance(value, bool):
                errors.append(f"Key '{key}': expected boolean, got {type(value).__name__}")

    return (len(errors) == 0, errors)
```

   Write similar `validate_model_vars()` and `validate_recognition_output()` functions.
   Use manual validation — do NOT add `jsonschema` as a dependency. The schemas are
   reference documentation; the validation functions check types and unknown keys.

7. Create `tests/fixtures/` directory. Add:
   - `tests/fixtures/global_vars_valid.json` — a valid config (copy key structure
     from `load_global_vars()` defaults).
   - `tests/fixtures/global_vars_invalid.json` — has wrong types and unknown keys.
   - `tests/fixtures/recognition_output_valid.json` — small valid detection result
     (2 images, 3 detections).
   - `tests/fixtures/recognition_output_invalid.json` — missing required fields.

8. Create `tests/test_schemas.py`:
   - Test that valid fixtures pass validation.
   - Test that invalid fixtures fail with specific error messages.
   - Test that every key in a real `global_vars.json` (from fixtures) is in the schema.
   - Target: 12–15 tests.

9. Run: `.venv/Scripts/python -m pytest tests/test_schemas.py -v` — all pass.
   Run: `.venv/Scripts/python -m pytest tests/ -v` — all existing tests still pass.
   Run: `.venv/Scripts/mypy addaxai/ --ignore-missing-imports --no-strict-optional` — clean.

10. Commit: `feat: add JSON schema validation for config and recognition output`

11. Open PR `phase6/json-schemas` → `main`. Merge when CI passes.

---

### Step 9: Integration test fixtures (6.5)

**Branch:** Create `phase6/test-fixtures` off `main`.

1. Add fixture images to `tests/fixtures/images/`. You need 3–5 small JPEG images
   (< 200KB each). Options:
   - Use Creative Commons camera trap images from LILA Science (https://lila.science).
   - Or create synthetic test images using Pillow in a setup script.
   For the implementation, create a helper `tests/fixtures/create_test_images.py` that
   generates synthetic images using Pillow (colored rectangles, ~10KB each) so there
   are no licensing concerns. Run it once and commit the generated images.

2. Create `tests/fixtures/golden_output.json` — a hand-crafted recognition result
   matching the fixture images. Follow the schema from step 8. Example structure:

```json
{
  "images": [
    {
      "file": "test_animal.jpg",
      "detections": [
        {"category": "1", "conf": 0.95, "bbox": [0.1, 0.2, 0.5, 0.6]}
      ]
    },
    {
      "file": "test_empty.jpg",
      "detections": []
    },
    {
      "file": "test_person.jpg",
      "detections": [
        {"category": "2", "conf": 0.88, "bbox": [0.3, 0.1, 0.7, 0.8]}
      ]
    }
  ],
  "detection_categories": {
    "1": "animal",
    "2": "person",
    "3": "vehicle"
  }
}
```

3. Create `tests/test_postprocess_pipeline.py`:
   - Import `addaxai.processing.postprocess.move_files` and
     `addaxai.processing.postprocess.format_size`.
   - Test `move_files()` in a temp directory: copy fixture images to temp, run
     `move_files()` with the golden JSON, verify files are moved to correct
     subdirectories (animal/, person/, empty/).
   - Test that `move_files()` with `copy=True` preserves originals.
   - Test that `move_files()` handles missing image files gracefully.
   - Target: 6–8 tests.

4. Create `tests/test_export_roundtrip.py`:
   - Test CSV export: write golden JSON → CSV using the export functions in
     `addaxai/processing/export.py`, read back, verify column names and row count.
   - Test COCO export: call `csv_to_coco()`, validate output against the
     recognition output schema from step 8.
   - Test that exported data round-trips without data loss (dates, confidence
     values, bounding box coordinates).
   - Target: 8–10 tests.

5. Run full test suite: `.venv/Scripts/python -m pytest tests/ -v` — all pass.

6. Commit: `feat: add integration test fixtures and pipeline/export tests`

7. Open PR `phase6/test-fixtures` → `main`. Merge when CI passes.

---

### Step 10: Model adapter protocol (6.4)

**Branch:** Create `phase6/model-protocol` off `main`.

1. Create `addaxai/models/backend.py`:

```python
"""Inference backend protocol — the contract all model adapters must fulfill."""

from typing import Any, Dict, List, Optional, Protocol


class DetectionResult:
    """Structured detection result for a single image."""

    def __init__(
        self,
        file: str,
        detections: List[Dict[str, Any]],
    ) -> None:
        self.file = file
        self.detections = detections


class InferenceBackend(Protocol):
    """Protocol that all inference backends must implement.

    Current implementations:
    - LocalSubprocessBackend (each model adapter in classification_utils/model_types/)

    Future implementations:
    - CloudBackend (HuggingFace / Replicate API)
    - ONNXBackend (local ONNX runtime, no conda env needed)
    """

    def detect(
        self,
        image_paths: List[str],
        model_path: str,
        confidence_threshold: float,
        **kwargs: Any,
    ) -> List[DetectionResult]:
        """Run object detection on a list of images."""
        ...

    def classify(
        self,
        crops: List[Dict[str, Any]],
        model_path: str,
        class_threshold: float,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run species classification on detection crops."""
        ...

    def is_available(self) -> bool:
        """Check if this backend is ready (model downloaded, env exists, etc.)."""
        ...
```

2. Create `classification_utils/model_types/_template/` directory with:

   `classification_utils/model_types/_template/classify_detections.py`:
   ```python
   """
   Template classification adapter.

   Copy this directory to create a new model adapter. Replace all TODO comments.

   This script is invoked as a subprocess by AddaxAI:
       python classify_detections.py <AddaxAI_files> <model_path> <det_thresh>
           <cls_thresh> <smooth> <json_path> <temp_frame_folder>
           <tax_fallback> <tax_levels_idx>

   Arguments (positional, via sys.argv):
       1. AddaxAI_files     - base path to AddaxAI installation
       2. cls_model_fpath   - path to model checkpoint file
       3. cls_detec_thresh   - detection confidence threshold (float)
       4. cls_class_thresh   - classification confidence threshold (float)
       5. smooth_bool       - "True"/"False" — smooth predictions across sequence
       6. json_path         - path to input/output recognition JSON
       7. temp_frame_folder - path to temp video frames (or "None")
       8. cls_tax_fallback  - "True"/"False" — use taxonomic fallback
       9. cls_tax_levels_idx - taxonomy level index (int)
   """

   import json
   import sys
   from PIL import ImageFile

   ImageFile.LOAD_TRUNCATED_IMAGES = True

   # ── Parse CLI arguments (same for all adapters — do not change) ────
   AddaxAI_files = str(sys.argv[1])
   cls_model_fpath = str(sys.argv[2])
   cls_detec_thresh = float(sys.argv[3])
   cls_class_thresh = float(sys.argv[4])
   smooth_bool = True if sys.argv[5] == 'True' else False
   json_path = str(sys.argv[6])
   temp_frame_folder = None if str(sys.argv[7]) == 'None' else str(sys.argv[7])
   cls_tax_fallback = True if sys.argv[8] == 'True' else False
   cls_tax_levels_idx = int(sys.argv[9])

   # ── TODO: Model-specific imports ──────────────────────────────────
   # import torch / tensorflow / etc.

   # ── TODO: Load model ──────────────────────────────────────────────
   # model = load_your_model(cls_model_fpath)

   # ── Read input JSON ───────────────────────────────────────────────
   with open(json_path, "r") as f:
       data = json.load(f)

   # ── TODO: Classify each detection ─────────────────────────────────
   # For each image in data["images"]:
   #   For each detection in image["detections"]:
   #     if detection["conf"] >= cls_detec_thresh:
   #       crop the detection region from the image
   #       prediction = model.predict(crop)
   #       if prediction.confidence >= cls_class_thresh:
   #         detection["classifications"] = [
   #           [prediction.label, prediction.confidence]
   #         ]

   # ── Write output JSON ─────────────────────────────────────────────
   with open(json_path, "w") as f:
       json.dump(data, f, indent=1)
   ```

   `classification_utils/model_types/_template/README.md`:
   ```markdown
   # Model Adapter Template

   Copy this directory and rename it to your model's name
   (e.g. `my-model-v1.0/`).

   ## Required files

   1. `classify_detections.py` — the classification script (this template)
   2. A `variables.json` in `models/cls/<model-name>/` — model configuration

   ## variables.json fields

   Create `AddaxAI_files/AddaxAI/models/cls/<model-name>/variables.json`:

   ```json
   {
     "model_type": "<model-name>",
     "framework": "pytorch|tensorflow|onnx",
     "cls_model_fname": "<checkpoint-filename>",
     "info_url": "https://...",
     "developer": "Your Name",
     "description": "Short model description",
     "all_classes": ["species1", "species2", "..."]
   }
   ```

   ## Testing your adapter

   1. Place model checkpoint in `AddaxAI_files/AddaxAI/models/cls/<model-name>/`
   2. Create `variables.json` as above
   3. Run AddaxAI, select your model, process test images
   4. Verify output JSON has `classifications` arrays on detections
   ```

3. Create `tests/test_model_protocol.py`:
   - Test that `InferenceBackend` protocol can be instantiated with a mock.
   - Test that `DetectionResult` holds data correctly.
   - Test that the `_template/classify_detections.py` is valid Python (import and
     compile, do not execute).
   - Target: 5–6 tests.

4. Run full test suite. Verify all pass.

5. Commit: `feat: add InferenceBackend protocol and model adapter template`

6. Open PR `phase6/model-protocol` → `main`. Merge when CI passes.

---

### Step 11: Event bus (6.6)

**Branch:** Create `phase6/event-bus` off `main`.

This is the most complex step. Take it in sub-commits.

**Sub-step 11a: Create the event bus module.**

1. Create `addaxai/core/events.py`:

```python
"""Lightweight publish/subscribe event bus.

Usage:
    from addaxai.core.events import event_bus

    # Subscribe
    def on_progress(pct: float, message: str) -> None:
        print(f"{pct}%: {message}")

    event_bus.on("deploy.progress", on_progress)

    # Publish
    event_bus.emit("deploy.progress", pct=50.0, message="Processing image 5/10")

    # Unsubscribe
    event_bus.off("deploy.progress", on_progress)

    # Unsubscribe all listeners for an event
    event_bus.clear("deploy.progress")
"""

import logging
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


class EventBus:
    """Simple synchronous event bus. Not thread-safe — designed for
    single-threaded tkinter event loop usage."""

    def __init__(self) -> None:
        self._listeners: Dict[str, List[Callable[..., Any]]] = {}

    def on(self, event: str, callback: Callable[..., Any]) -> None:
        """Register a callback for an event."""
        if event not in self._listeners:
            self._listeners[event] = []
        if callback not in self._listeners[event]:
            self._listeners[event].append(callback)

    def off(self, event: str, callback: Callable[..., Any]) -> None:
        """Remove a callback for an event."""
        if event in self._listeners:
            self._listeners[event] = [
                cb for cb in self._listeners[event] if cb is not callback
            ]

    def emit(self, event: str, **kwargs: Any) -> None:
        """Emit an event, calling all registered callbacks with kwargs."""
        for callback in self._listeners.get(event, []):
            try:
                callback(**kwargs)
            except Exception:
                logger.error(
                    "Error in event handler for '%s'", event, exc_info=True
                )

    def clear(self, event: str) -> None:
        """Remove all listeners for an event."""
        self._listeners.pop(event, None)

    def clear_all(self) -> None:
        """Remove all listeners for all events."""
        self._listeners.clear()


# Module-level singleton — import this in all modules.
event_bus = EventBus()
```

2. Create `tests/test_event_bus.py`:
   - Test `on()` + `emit()` calls the callback with correct kwargs.
   - Test `off()` removes a specific callback.
   - Test `clear()` removes all callbacks for one event.
   - Test `clear_all()` removes everything.
   - Test that a failing callback does not prevent other callbacks from running.
   - Test that duplicate `on()` calls do not register twice.
   - Test emitting an event with no listeners does not raise.
   - Target: 10–12 tests.

3. Run tests. All pass.

4. Commit: `feat: add lightweight event bus (addaxai/core/events.py)`

**Sub-step 11b: Define the standard event names.**

1. Create `addaxai/core/event_types.py`:

```python
"""Standard event names emitted by AddaxAI.

These are string constants — not an enum — so they can be used as dict keys
and in f-strings without .value access. Grouped by feature area.
"""

# ── Deployment ────────────────────────────────────────────────────────
DEPLOY_STARTED = "deploy.started"
DEPLOY_PROGRESS = "deploy.progress"           # pct: float, message: str
DEPLOY_IMAGE_COMPLETE = "deploy.image_complete"  # image_path: str, index: int, total: int
DEPLOY_ERROR = "deploy.error"                 # message: str, exc: Optional[Exception]
DEPLOY_CANCELLED = "deploy.cancelled"
DEPLOY_FINISHED = "deploy.finished"           # results_path: str

# ── Classification ────────────────────────────────────────────────────
CLASSIFY_STARTED = "classify.started"
CLASSIFY_PROGRESS = "classify.progress"       # pct: float, message: str
CLASSIFY_ERROR = "classify.error"             # message: str
CLASSIFY_FINISHED = "classify.finished"       # results_path: str

# ── Postprocessing ────────────────────────────────────────────────────
POSTPROCESS_STARTED = "postprocess.started"
POSTPROCESS_PROGRESS = "postprocess.progress"  # pct: float, message: str
POSTPROCESS_ERROR = "postprocess.error"        # message: str
POSTPROCESS_FINISHED = "postprocess.finished"

# ── Model management ─────────────────────────────────────────────────
MODEL_DOWNLOAD_STARTED = "model.download_started"    # model_name: str
MODEL_DOWNLOAD_PROGRESS = "model.download_progress"  # pct: float
MODEL_DOWNLOAD_FINISHED = "model.download_finished"  # model_name: str
MODEL_DOWNLOAD_ERROR = "model.download_error"        # model_name: str, message: str
```

2. Add tests in `tests/test_event_bus.py` that verify all event name constants are
   unique strings.

3. Commit: `feat: add standard event type constants`

**Sub-step 11c: Wire one orchestrator to the event bus (proof of concept).**

This sub-step modifies `AddaxAI_GUI.py`. Be careful — this file is large and fragile.

1. In `AddaxAI_GUI.py`, find the `deploy_model()` function. Identify every place it
   calls `progress_window.update_values()` or similar progress update. These are the
   places to add `event_bus.emit()` calls.

2. Add `from addaxai.core.events import event_bus` and
   `from addaxai.core.event_types import DEPLOY_STARTED, DEPLOY_PROGRESS, DEPLOY_FINISHED, DEPLOY_ERROR, DEPLOY_CANCELLED`
   to the imports in `AddaxAI_GUI.py`.

3. At the start of `deploy_model()`, add:
   `event_bus.emit(DEPLOY_STARTED)`

4. At each progress update point, add an `event_bus.emit()` call **alongside** the
   existing `progress_window` call (do not remove the existing call yet — dual-write):
   `event_bus.emit(DEPLOY_PROGRESS, pct=percentage, message=status_text)`

5. At the end of `deploy_model()`, add:
   `event_bus.emit(DEPLOY_FINISHED, results_path=json_path)`

6. On error, add:
   `event_bus.emit(DEPLOY_ERROR, message=error_msg)`

7. On cancel, add:
   `event_bus.emit(DEPLOY_CANCELLED)`

8. Run the GUI smoke test to verify nothing broke:
   `C:\Users\Topam\AddaxAI_files\envs\env-base\python.exe -m pytest tests/test_gui_smoke.py -v`

9. Run unit tests to verify no import errors:
   `.venv/Scripts/python -m pytest tests/ -v`

10. Commit: `feat: wire deploy_model() to event bus (dual-write, no behavior change)`

11. Repeat sub-step 11c for `classify_detections()` and `start_postprocess()` in
    separate commits.

12. Open PR `phase6/event-bus` → `main`. Merge when CI passes.

---

### Step 12: View protocols (part of 6.6b)

**Branch:** Create `phase6/view-protocols` off `main`.

1. Create `addaxai/ui/protocols.py`:

```python
"""View protocols — contracts between orchestration logic and the UI.

Each protocol defines what a UI component must be able to do, without
specifying how (no tkinter/Qt/web types in signatures). Orchestration
code and the event bus talk to these protocols. Concrete implementations
live in the ui/ subpackages.
"""

from typing import Any, Callable, Dict, List, Optional, Protocol


class DeployView(Protocol):
    """UI contract for the deployment workflow."""

    def show_progress(self, pct: float, message: str) -> None: ...
    def show_error(self, message: str) -> None: ...
    def show_completion(self, results_path: str) -> None: ...
    def set_model_list(self, models: List[str]) -> None: ...
    def on_cancel(self, callback: Callable[[], None]) -> None: ...
    def reset(self) -> None: ...


class PostprocessView(Protocol):
    """UI contract for postprocessing."""

    def show_progress(self, pct: float, message: str) -> None: ...
    def show_error(self, message: str) -> None: ...
    def show_completion(self, summary: Dict[str, Any]) -> None: ...
    def reset(self) -> None: ...


class HITLView(Protocol):
    """UI contract for human-in-the-loop annotation."""

    def load_annotations(self, data: Dict[str, Any]) -> None: ...
    def show_image(self, path: str, boxes: List[Dict[str, Any]]) -> None: ...
    def on_save(self, callback: Callable[[Dict[str, Any]], None]) -> None: ...
    def reset(self) -> None: ...


class ResultsView(Protocol):
    """UI contract for results display."""

    def display(self, recognition_json: Dict[str, Any]) -> None: ...
    def set_filters(
        self,
        species: Optional[List[str]],
        confidence: Optional[float],
    ) -> None: ...
    def reset(self) -> None: ...
```

2. Create `tests/test_view_protocols.py`:
   - For each protocol, create a minimal mock class that implements it.
   - Verify the mock satisfies `isinstance` checks via `runtime_checkable` (add
     `@runtime_checkable` decorator to each protocol).
   - Verify that a class missing a method does NOT satisfy the protocol.
   - Target: 8–10 tests.

3. Run full test suite. All pass.

4. Commit: `feat: add view protocols for UI decoupling (DeployView, PostprocessView, HITLView, ResultsView)`

5. Open PR `phase6/view-protocols` → `main`. Merge when CI passes.

---

### Step 13: Rename and break apart the GUI file (6.6b)

**Branch:** Create `phase6/gui-restructure` off `main`.

This is the highest-risk step. Take it in sub-commits, run the GUI smoke test after each.

**Sub-step 13a: Rename `AddaxAI_GUI.py` → `addaxai/app.py`.**

1. `git mv AddaxAI_GUI.py addaxai/app.py`

2. In `addaxai/app.py`, update the `AddaxAI_files` path resolution line (line 114).
   The old logic derives the path from `__file__`:
   ```python
   AddaxAI_files = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
   ```
   After the move, the file is one level deeper (`addaxai/app.py` instead of root).
   Change to:
   ```python
   AddaxAI_files = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
   ```
   (Three `dirname` calls instead of two, because `addaxai/app.py` is two levels below
   the repo root, which is one level below `AddaxAI_files`.)

3. Update `main.py` line 36 — change:
   ```python
   GUI_script = os.path.join(AddaxAI_files, "AddaxAI", "AddaxAI_GUI.py")
   ```
   to:
   ```python
   GUI_script = os.path.join(AddaxAI_files, "AddaxAI", "addaxai", "app.py")
   ```

4. Update `dev_launch.py` line 23 — change:
   ```python
   gui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AddaxAI_GUI.py")
   ```
   to:
   ```python
   gui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "addaxai", "app.py")
   ```

5. Update `dev_launch.py` lines 29–31 — the string replacement target must match the
   NEW path resolution line (three `dirname` calls):
   ```python
   source = source.replace(
       "AddaxAI_files = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))",
       f"AddaxAI_files = r'{ADDAXAI_FILES}'"
   )
   ```

6. Update `tests/gui_test_runner.py` line 36 — change:
   ```python
   gui_path = os.path.join(REPO_ROOT, "AddaxAI_GUI.py")
   ```
   to:
   ```python
   gui_path = os.path.join(REPO_ROOT, "addaxai", "app.py")
   ```

7. Update `tests/gui_test_runner.py` lines 42–44 — same string replacement fix as
   `dev_launch.py` (three `dirname` calls in the target string).

8. Search the entire repo for any remaining references to `AddaxAI_GUI`:
   ```bash
   grep -r "AddaxAI_GUI" --include="*.py" --include="*.yml" --include="*.yaml" --include="*.md"
   ```
   Update every hit. Common locations: CLAUDE.md, CONTRIBUTING.md, README.md, comments
   in test files, docstrings.

9. Run the GUI smoke test:
   ```bash
   C:\Users\Topam\AddaxAI_files\envs\env-base\python.exe -m pytest tests/test_gui_smoke.py -v
   ```

10. Run unit tests:
    ```bash
    .venv/Scripts/python -m pytest tests/ -v
    ```

11. Run GUI integration tests:
    ```bash
    C:\Users\Topam\AddaxAI_files\envs\env-base\python.exe -m pytest tests/test_gui_integration.py -v
    ```

12. All three must pass. If any fail, the path resolution is wrong — debug before continuing.

13. Commit: `refactor: rename AddaxAI_GUI.py to addaxai/app.py`

**Sub-step 13b: Extract deployment UI.**

Wait until step 11 (event bus) is merged and the dual-write emit calls are in
`addaxai/app.py`. Then:

1. Identify all functions in `addaxai/app.py` related to deployment UI:
   - The deployment progress section of `start_deploy()`
   - Widget construction for the deploy tab
   - Event handlers for deploy buttons

2. Create `addaxai/ui/deploy_tab.py`. Move the deployment widget construction and
   event handler code into a class that implements `DeployView` (from step 12).

3. In `addaxai/app.py`, replace the moved code with an instantiation of the new class
   and wire it to the event bus.

4. Run all three test suites (unit, smoke, integration). All must pass.

5. Commit: `refactor: extract deployment UI to addaxai/ui/deploy_tab.py`

**Sub-step 13c–e:** Repeat for postprocessing UI, HITL UI, and results viewer.
Each is a separate commit. Run all tests after each.

6. Open PR `phase6/gui-restructure` → `main`. Merge when CI passes.

---

### Step 14: REST API layer (6.7)

**Branch:** Create `phase6/rest-api` off `main`.

This depends on step 11 (event bus) being merged.

1. Add `fastapi` and `uvicorn` to `pyproject.toml` under a new optional dependency
   group `[project.optional-dependencies] api = ["fastapi", "uvicorn"]`.

2. Create `addaxai/api/__init__.py` (empty).

3. Create `addaxai/api/server.py` — a FastAPI app with read-only endpoints first:

```python
"""Local REST API server for AddaxAI.

Start with: uvicorn addaxai.api.server:app --port 6189
"""

import json
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException

app = FastAPI(
    title="AddaxAI API",
    description="Local API for camera trap image classification",
    version="0.1.0",
)


def _get_base_path() -> str:
    """Resolve AddaxAI_files path."""
    # Check environment variable first, then default locations
    base = os.environ.get("ADDAXAI_FILES")
    if base and os.path.isdir(base):
        return base
    raise HTTPException(
        status_code=500,
        detail="ADDAXAI_FILES environment variable not set",
    )


@app.get("/models")
def list_models() -> Dict[str, List[str]]:
    """List available detection and classification models."""
    from addaxai.models.registry import fetch_known_models
    from addaxai.core.paths import get_model_dir

    base = _get_base_path()
    det_dir = get_model_dir(base, "det")
    cls_dir = get_model_dir(base, "cls")
    return {
        "detection": fetch_known_models(det_dir),
        "classification": fetch_known_models(cls_dir),
    }


@app.get("/results/{folder_name}")
def get_results(folder_name: str) -> Dict[str, Any]:
    """Get recognition results for a processed folder."""
    base = _get_base_path()
    json_path = os.path.join(base, folder_name, "image_recognition_file.json")
    if not os.path.isfile(json_path):
        raise HTTPException(status_code=404, detail="Results not found")
    with open(json_path, "r") as f:
        return json.load(f)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}
```

4. Create `tests/test_api.py`:
   - Use FastAPI's `TestClient` (from `fastapi.testclient`).
   - Test `/health` returns 200.
   - Test `/models` with a mocked `ADDAXAI_FILES` temp directory containing model dirs.
   - Test `/results/{folder}` with a fixture JSON.
   - Test `/results/{missing}` returns 404.
   - Target: 6–8 tests.

5. Mark API tests to skip if `fastapi` is not installed:
   ```python
   pytest.importorskip("fastapi")
   ```

6. Run unit tests. All pass (API tests skip in envs without fastapi).

7. Commit: `feat: add read-only REST API layer (GET /models, /results, /health)`

8. Open PR `phase6/rest-api` → `main`. Merge when CI passes.

---

### Verification checklist

After all steps are complete, verify the following:

- [ ] `pyproject.toml` exists with all metadata, deps, and tool config
- [ ] CI runs 3 jobs: test (with coverage), lint, typecheck
- [ ] `.github/ISSUE_TEMPLATE/` has bug report and feature request forms
- [ ] `.github/pull_request_template.md` exists
- [ ] `.editorconfig`, `Makefile`, `.pre-commit-config.yaml` exist
- [ ] `addaxai/py.typed` marker exists
- [ ] `addaxai/schemas/` has 3 JSON schemas and a `validate.py` module
- [ ] `tests/fixtures/` has test images, golden output, valid/invalid configs
- [ ] `addaxai/models/backend.py` defines `InferenceBackend` protocol
- [ ] `classification_utils/model_types/_template/` has annotated adapter template
- [ ] `addaxai/core/events.py` has `EventBus` class and module-level `event_bus`
- [ ] `addaxai/core/event_types.py` has all standard event constants
- [ ] `addaxai/ui/protocols.py` has 4 view protocols
- [ ] `AddaxAI_GUI.py` no longer exists — it is now `addaxai/app.py`
- [ ] `addaxai/app.py` is under 2,000 lines
- [ ] Feature UI modules exist in `addaxai/ui/` (deploy, postprocess, HITL, results)
- [ ] `addaxai/api/server.py` has working read-only endpoints
- [ ] All unit tests pass (target: 400+)
- [ ] GUI smoke test passes
- [ ] GUI integration tests pass
- [ ] No references to `AddaxAI_GUI` remain anywhere in the repo
