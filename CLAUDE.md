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
