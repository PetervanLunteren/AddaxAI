# AddaxAI - Refactoring Plan

## What This App Is

AddaxAI helps ecologists classify camera trap images using local computer vision (MegaDetector + various classification models). It supports multiple languages (English, Spanish, French), runs fully offline, and is packaged via PyInstaller for non-technical users.

## Why We're Refactoring

The application originally lived in a single ~11,200-line file (`AddaxAI_GUI.py`) with no separation between business logic, UI, model deployment, data processing, and localization. All state was managed via `global` variables. Now at ~8,670 lines with ~3,500 lines extracted to `addaxai/` modules. This makes the codebase:

- Impossible to test in isolation
- Extremely difficult to modify without introducing regressions
- Unable to support future features like cloud-based inference
- Painful for any new contributor to understand

## Architectural Decisions

### UI Framework: Stay on customtkinter (for now)
- Fast startup, zero extra dependencies, works offline, low resource footprint
- Good fit for field ecologists on modest hardware
- Phases 1-4 fully separate business logic from UI, making a future framework swap (PySide6, Dear PyGui) a contained effort limited to the `ui/` directory

### Model Adapters: Leave largely alone
- The `classification_utils/model_types/` adapter system already follows a reasonable plugin pattern
- Each adapter runs as a subprocess (often in its own conda env with different ML frameworks)
- `inference_lib.py` provides the generic orchestration via `crop_function`/`inference_function` callbacks
- The boilerplate duplication is a conscious tradeoff for subprocess isolation

### Inference Backend: Design for future cloud support
- When extracting `models/deploy.py`, keep a clean backend interface so a `CloudBackend` can slot in later
- Likely approach: local detect (MegaDetector is fast) + cloud classify (upload crops only)
- Potential hosting: HuggingFace Inference Endpoints or Replicate (users pay for own compute)

## Target Architecture

```
addaxai/
├── app.py                      # Entry point
├── core/
│   ├── config.py               # Settings load/save (replaces global state)
│   ├── paths.py                # Path resolution
│   └── platform.py             # OS detection, DPI, Python interpreter lookup
├── models/
│   ├── registry.py             # Model discovery, download checks, model info
│   ├── deploy.py               # Deployment orchestration
│   ├── backend.py              # InferenceBackend interface (local + future cloud)
│   └── speciesnet.py           # SpeciesNet-specific logic
├── processing/
│   ├── postprocess.py          # File separation, cropping
│   ├── annotations.py          # Pascal VOC, COCO, XML/JSON conversion
│   └── export.py               # CSV / XLSX / JSON export
├── analysis/
│   ├── plots.py                # Charts (matplotlib, plotly, seaborn)
│   └── maps.py                 # Folium heatmaps, GPS extraction
├── i18n/
│   ├── __init__.py             # t("key") translation function
│   ├── en.json                 # English strings
│   ├── es.json                 # Spanish strings
│   └── fr.json                 # French strings
├── ui/
│   ├── app_window.py           # Root window, mode switching
│   ├── advanced/               # Advanced mode tabs
│   ├── simple/                 # Simple mode window
│   ├── dialogs/                # Progress, donation, model info, results
│   └── widgets/                # Reusable widget classes
├── hitl/
│   ├── session.py              # Human-in-the-loop session management
│   └── exchange.py             # LabelImg exchange, XML conversion
└── utils/
    ├── files.py                # File browsing, size formatting
    ├── images.py               # Corruption check, EXIF, blur
    ├── json_ops.py             # JSON path manipulation, checkpoints
    └── sorting.py              # Natural sort, checkpoint ordering
```

## Execution Plan

### Phase 0: Foundation (DONE)
- [x] Create feature branch `refactor/modularize`
- [x] Create this CLAUDE.md
- [x] Create `addaxai/` package directory structure (13 subpackages)
- [x] Add pytest + smoke tests (13 passing)
- [x] Set up `.venv/` with pytest
- [x] Fork repo to `TeodoroTopa/AddaxAI`
- [x] Configure remotes: `origin` = fork, `upstream` = original
- [x] Add `.venv/` and `.pytest_cache/` to `.gitignore`
- [x] Initial commit and push

### Phase 1: Extract Pure Backend + Wire Into GUI (DONE)
Extracted 58 functions (~1,378 lines) into 12 modules, then wired them back in:
- [x] 1.1–1.12: Extract all modules (see git history for individual commits)
- [x] 1.13: Wire Batch 1 — 28 drop-in functions (identical signatures)
- [x] 1.14: Wire Batch 2 — 10 functions needing `base_path=AddaxAI_files`
- [x] 1.15: Wire Batch 3 — 7 functions needing `base_folder=var_choose_folder.get()`
- [x] 1.16: Wire Batch 4 — 2 special cases (`cancel_subprocess` UI wrapper, `csv_to_coco` version arg)

### Phase 2: Extract Localization (DONE)
- [x] 2.1: Build i18n JSON files and `t()` function
- [x] 2.2: Replace module-level `_txt` variables with `t()` calls (2.2a–2.2e)
- [x] 2.3: Replace `dpd_options_*` translation arrays with `t()` calls
- [x] 2.4: Replace inline anonymous `["En","Es","Fr"][lang_idx]` arrays with `t()` calls
- [x] 2.5: Update `set_language()` to call `i18n_set_language()` so `t()` works
- [x] 2.6: Remove `lang_idx` global — all language state goes through `addaxai.i18n`

### Phase 3: Restructure UI (DONE)
- [x] 3.1: Extract widget classes to `addaxai/ui/widgets/`
- [x] 3.2: Extract dialog classes to `addaxai/ui/dialogs/`
- [x] 3.3: Extract ProgressWindow to `addaxai/ui/dialogs/progress.py`
- [x] 3.4: Extract help tab content to `addaxai/ui/advanced/help_tab.py`
- [x] 3.5: Extract about tab content to `addaxai/ui/advanced/about_tab.py`
- [x] 3.6: Extract simple mode window to `addaxai/ui/simple/simple_window.py`

### Phase 4: Kill Global State (DONE)
- [x] 4.1: Create `AppState` dataclass in `addaxai/core/state.py`
- [x] 4.2: Instantiate `AppState` after root window creation, migrate tkinter variables
- [x] 4.3: Migrate cancel flags and deployment state globals → `AppState`
- [x] 4.4: Migrate HITL state globals → `AppState`
- [x] 4.5: Migrate simple mode widget refs and dropdown options → `AppState`
- [x] 4.6: Migrate remaining globals (init flags, caches, timelapse) → `AppState`
- [x] 4.7: Extract `SpeciesNetOutputWindow` → `addaxai/ui/dialogs/speciesnet_output.py`
- [x] 4.8: Final audit — grep confirms zero `global` declarations remain

### Phase 5: Polish
- [ ] 5.1: Type hints on extracted modules
- [ ] 5.2: Replace print() with proper logging
- [ ] 5.3: CI setup (lint + smoke tests)

## Dev Setup

```bash
# Clone your fork
git clone https://github.com/TeodoroTopa/AddaxAI.git
cd AddaxAI
git checkout refactor/modularize

# Create venv and install dev dependencies
python -m venv .venv
.venv/Scripts/pip install pytest    # Windows
# .venv/bin/pip install pytest      # Mac/Linux

# Run tests
.venv/Scripts/python -m pytest tests/ -v    # Windows
# .venv/bin/python -m pytest tests/ -v      # Mac/Linux

# Remotes
# origin   = TeodoroTopa/AddaxAI (your fork — push here)
# upstream = PetervanLunteren/AddaxAI (original — pull updates from here)
```

## Development Methodology

### Test-Driven Development (TDD)
Every extraction step follows this workflow:
1. **Write tests first** in `tests/test_<module>.py` with top-level imports
2. **Implement the module** to make all tests pass
3. **Run the full test suite** (`python -m pytest tests/ -v`) to verify no regressions
4. **Commit with a conventional commit message** (feat/fix/refactor prefix)
5. **Push immediately** so work is recoverable

### Conventions
- Each extraction step = one commit, immediately pushable
- When extracting a function: parameterize globals (e.g. `var_choose_folder.get()` → `base_folder` parameter)
- Run full test suite after each extraction before committing
- Do not change behavior during extraction — pure mechanical moves only
- Keep `AddaxAI_GUI.py` working at every commit (it's the only entry point until Phase 3)
- Test imports go at top of file, not inside each test function
- Optional heavy dependencies (cv2, matplotlib) use `pytest.mark.skipif` so tests degrade gracefully
- Fix bugs discovered in original code during extraction (document in commit message)

## Phase 1 Behavioral Changes Log

During extraction, the goal was pure mechanical moves (parameterize globals, lift nested
functions, rename to avoid shadowing builtins). Most functions are exact copies. The
changes below are the exceptions — anything that could cause different runtime behavior
when the extracted modules are wired into `AddaxAI_GUI.py`.

### Bug fixes (verify during live testing that the fix is correct)

**1. `utils/files.py` — `sort_checkpoint_files`** (original line 5822)
- Original: `file.split('_')[1].split('.')[0]` to extract timestamp
- Extracted: `file.split('_')[2].split('.')[0]`
- Why: For filenames like `md_checkpoint_20230101120000.json`, index `[1]` gives `"checkpoint"`, not the timestamp. Index `[2]` gives the actual timestamp digits. Original would crash or sort incorrectly.
- Test: Sort a folder with multiple checkpoint files, verify ordering is chronological.

**2. `utils/json_ops.py` — `get_hitl_var_in_json`** (original line 5605)
- Original: `data['info'].get("addaxai_metadata") or data['info'].get("ecoassist_metadata")` then directly accesses `["hitl_status"]`. Crashes with `TypeError` if neither metadata key exists.
- Extracted: Uses explicit `is None` checks, returns `"never-started"` if no metadata found.
- Test: Open a JSON file that has no `addaxai_metadata` or `ecoassist_metadata` key in the HITL workflow. Original would crash; extracted should return gracefully.

**3. `processing/export.py` — `csv_to_coco`** (original line 854)
- Original: `type(val) == float` to detect NA dates — treats ANY float as NA, not just NaN.
- Extracted: `isinstance(val, float) and math.isnan(val)` — only NaN is treated as NA. Also added try/except around datetime parsing for malformed date strings.
- Test: Export a CSV with valid float-like date values (unlikely but possible). Check that datetime fields in the COCO output are correct.

### UI code stripped (caller must handle separately when wiring)

**4. `models/deploy.py` — `cancel_subprocess`** (original line 2961)
- Original also: sets `cancel_deploy_model_pressed = True`, re-enables `btn_start_deploy` and `sim_run_btn` (UI buttons), calls `progress_window.close()`.
- Extracted: Only kills the subprocess (TASKKILL on Windows, SIGTERM on Unix).
- When wiring: The caller in `AddaxAI_GUI.py` must handle button re-enabling, cancel flag, and progress window closure after calling the extracted function.

### Return type changed

**5. `models/registry.py` — `environment_needs_downloading`** (original line 3396)
- Original: Returns `[bool, str]` (list).
- Extracted: Returns `(bool, str)` (tuple).
- Risk: Low. Indexing and unpacking work identically. Only breaks if something checks `isinstance(result, list)`, which is unlikely.

### Error handling loosened (potential regression)

**6. `models/registry.py` — `set_up_unknown_model`** (original line 6292)
- Original: Catches `requests.exceptions.RequestException`, prints informative error/success messages for taxonomy CSV download.
- Extracted: Catches bare `Exception`, silently passes. No logging at all.
- Risk: If a taxonomy CSV download fails, there's no indication of why. Original at least printed the error. Should be restored to specific exception + logging before wiring.

### Removed debug logging (7 functions)

These functions all had `print(f"EXECUTED : {sys._getframe().f_code.co_name}({locals()})\n")` at the top, which was removed. This is intentional — Phase 5.2 will replace all `print()` with proper logging.

- `csv_to_coco` (export.py)
- `move_files` (postprocess.py)
- `switch_yolov5_version` (deploy.py)
- `imitate_object_detection_for_full_image_classifier` (deploy.py)
- `distribute_individual_model_jsons` (registry.py)
- `set_up_unknown_model` (registry.py) — also lost download success/failure messages
- `merge_jsons` (json_ops.py) — lost `"merged json file saved to..."` message

### Cosmetic-only changes (no behavioral impact)

These are NOT behavioral changes, just listed for completeness:
- All globals parameterized (`AddaxAI_files` → `base_path`, `var_choose_folder.get()` → `base_folder`, etc.)
- `indent` → `indent_xml` (avoid shadowing builtin)
- `get_python_interprator` → `get_python_interpreter` (typo fix)
- `conf_dirs` → `CONF_DIRS`, `dtypes` → `CSV_DTYPES` (constant naming convention)
- `object` → `obj` in `create_pascal_voc_annotation` (avoid shadowing builtin)
- `var_file_placement` → `file_placement`, `var_sep_conf` → `sep_conf` (dropped `var_` prefix)
- `except:` → `except Exception:` in `is_image_corrupted` (won't catch SystemExit/KeyboardInterrupt)
- `except:` → `except KeyError:` in `convert_xml_to_coco` (more specific)
- Lazy `import cv2` inside `blur_box` (was top-level in original)
- `import io` moved to module top-level in `fig2img` (was inside function)

## Dev Launch & Smoke Testing

`dev_launch.py` runs the repo's `AddaxAI_GUI.py` using the installed app's conda env-base Python.
It patches `AddaxAI_files` to point at `C:\Users\Topam\AddaxAI_files` and sets up `sys.path` so
all dependencies (visualise_detection, cameratraps, megadetector) resolve correctly.

```bash
# Launch GUI for manual testing:
C:\Users\Topam\AddaxAI_files\envs\env-base\python.exe dev_launch.py

# Automated smoke test (GUI starts, waits 10s, asserts no crash):
C:\Users\Topam\AddaxAI_files\envs\env-base\python.exe -m pytest tests/test_gui_smoke.py -v
```

## Current Status

**Branch:** `refactor/modularize`
**Unit tests:** 319 passing, 9 skipped (optional deps: cv2, matplotlib, customtkinter) — run with `.venv` Python 3.14
**Integration tests:** 3 passing (language cycling, mode switching, folder selection) — run with env-base Python 3.8
**GUI smoke test:** 1 passing — run with env-base Python 3.8
**Python (tests):** `C:\Users\Topam\AppData\Local\Python\bin\python.exe` (3.14)
**Python (GUI):** `C:\Users\Topam\AddaxAI_files\envs\env-base\python.exe`
**Installed test deps:** pytest, Pillow, numpy, pandas, requests

**Phase 4 fully complete!** Zero `global` declarations remain in `AddaxAI_GUI.py`.
- 58 backend functions extracted into 12 modules (Phase 1)
- i18n system: 662 `[lang_idx]` occurrences → `t()` calls, 3 JSON files (~300 keys each) (Phase 2)
- 7 widget/dialog/tab classes extracted to `addaxai/ui/` (Phase 3)
- `AppState` class owns all mutable state; 35 `global` declarations eliminated (Phase 4)
- `SpeciesNetOutputWindow` extracted to `addaxai/ui/dialogs/speciesnet_output.py` (Phase 4.7)

**Next:** Phase 5 — polish (type hints, logging, CI).

## Testing Strategy

### Test infrastructure

Tests are split by runtime:
- **Unit tests** (`tests/test_*.py` except GUI tests): Run with `.venv` Python 3.14. Fast (~6s).
  Import extracted `addaxai/` modules directly. No tkinter or GUI dependencies.
- **GUI integration tests** (`tests/test_gui_integration.py`): Run with env-base Python 3.8.
  Use `tests/gui_test_runner.py` harness that boots the full GUI, schedules test actions via
  `root.after()`, writes results to a temp JSON file, then exits. Each test ~15s.
- **GUI smoke test** (`tests/test_gui_smoke.py`): Run with env-base Python 3.8.
  Launches GUI as subprocess, waits 10s, asserts process is still alive.

### How the GUI test harness works

`tests/gui_test_runner.py` is invoked as:
```bash
env-base/python.exe tests/gui_test_runner.py <test_name> <results_file>
```

It `exec()`s AddaxAI_GUI.py (with patched `AddaxAI_files` path, like `dev_launch.py`), but
removes the `if __name__ == "__main__"` block so `main()` is not called. Instead it:
1. Manually initializes frame states and calls `switch_mode()` twice (same as `main()`)
2. Schedules the named test function via `root.after(2000, ...)`
3. The test function interacts with widgets, collects results into a dict
4. Writes results as JSON to the results file, then calls `root.quit()`

`tests/test_gui_integration.py` launches the runner as a subprocess and asserts on the JSON.

### Current integration tests (Tier 1)

| Test | What it does | What it catches |
|------|-------------|-----------------|
| `test_language_cycling` | Calls `set_language()` 3x (EN→ES→FR→EN), checks 12 advanced + 5 simple mode widget texts per language | Missing `t()` key, widget not updating on language switch |
| `test_mode_switching` | Toggles advanced↔simple twice, checks `advanced_mode` persisted value and window visibility | Broken mode toggle, window not showing/hiding |
| `test_folder_selection` | Sets `var_choose_folder` to a temp dir, calls `update_frame_states()` | Crash on folder selection, broken frame state wiring |

### Running tests

```bash
# Unit tests only (fast, no GUI):
.venv/Scripts/python -m pytest tests/ -v --ignore=tests/test_gui_smoke.py --ignore=tests/test_gui_integration.py

# GUI integration tests (need env-base):
C:\Users\Topam\AddaxAI_files\envs\env-base\python.exe -m pytest tests/test_gui_integration.py -v

# GUI smoke test:
C:\Users\Topam\AddaxAI_files\envs\env-base\python.exe -m pytest tests/test_gui_smoke.py -v

# All tests (env-base — unit tests also work here):
C:\Users\Topam\AddaxAI_files\envs\env-base\python.exe -m pytest tests/ -v
```

### Verification after each Phase 4 step

After every commit during Phase 4, run:
1. `.venv/Scripts/python -m pytest tests/ -v --ignore=tests/test_gui_smoke.py --ignore=tests/test_gui_integration.py` — unit tests pass
2. `C:\Users\Topam\AddaxAI_files\envs\env-base\python.exe -m pytest tests/test_gui_integration.py -v` — integration tests pass
3. `C:\Users\Topam\AddaxAI_files\envs\env-base\python.exe -m pytest tests/test_gui_smoke.py -v` — smoke test passes

If any integration test fails after a Phase 4 step, `git revert` the commit and investigate.

## Phase 4: Kill Global State — Detailed Implementation Plan

### Overview

`AddaxAI_GUI.py` currently has **35 `global` declarations** across its functions and **~50 tkinter
variables** (StringVar, IntVar, BooleanVar, DoubleVar) declared at module level. All mutable state
is accessed via `global` keywords or bare module-level names, making functions impossible to test
in isolation and creating hidden coupling between every part of the codebase.

Phase 4 creates an `AppState` class that owns all mutable state, then systematically passes it to
every function that currently uses `global`. The goal is **zero `global` declarations** in
`AddaxAI_GUI.py` by the end.

### Current Global Variables (35 total)

**Cancel flags and deployment state (7):**
- `cancel_var` — tkinter BooleanVar for cancellation
- `cancel_deploy_model_pressed` — bool flag
- `cancel_speciesnet_deploy_pressed` — bool flag
- `btn_start_deploy` — Button widget (re-enabled on cancel)
- `sim_run_btn` — Simple mode run button (re-enabled on cancel)
- `subprocess_output` — captured stdout from spawned processes
- `warn_smooth_vid` — one-time warning flag

**Progress and error tracking (5):**
- `progress_window` — ProgressWindow instance
- `postprocessing_error_log` — list of errors during postprocess
- `model_error_log` — list of errors during deployment
- `model_warning_log` — list of warnings during deployment
- `model_special_char_log` — list of special character warnings

**HITL state (6):**
- `selection_dict` — dict of selected species for verification
- `rad_ann_var` — IntVar for annotation radio buttons
- `hitl_ann_selection_frame` — frame widget
- `hitl_settings_canvas` — canvas widget
- `hitl_settings_window` — window widget
- `lbl_n_total_imgs` — label widget showing image count

**Simple mode widget refs (5):**
- `sim_dir_pth` — label widget
- `sim_mdl_dpd` — dropdown widget
- `sim_run_btn` — button widget (also in cancel flags)
- `sim_spp_scr` — species scrollable frame
- `sim_dpd_options_cls_model` — dropdown options list

**Dropdown option lists (3):**
- `dpd_options_cls_model` — classification model dropdown options
- `dpd_options_model` — detection model dropdown options
- `loc_chkpnt_file` — checkpoint file path

**Initialization flags (6):**
- `checkpoint_freq_init` — bool
- `image_size_for_deploy_init` — bool
- `nth_frame_init` — bool
- `shown_abs_paths_warning` — bool
- `check_mark_one_row` — bool
- `check_mark_two_rows` — bool

**Timelapse and caches (3):**
- `timelapse_mode` — bool
- `timelapse_path` — string
- `_ALL_SUPPORTED_MODEL_CLASSES_CACHE` — cached model class list
- `temp_frame_folder` — temporary folder during video processing

### AppState Class Design

```python
# addaxai/core/state.py
import tkinter as tk


class AppState:
    """Holds all mutable application state that was previously managed via globals.

    Instantiated once after the root window is created (tkinter variables need
    an active Tk instance). Passed to functions that previously used `global`.
    """

    def __init__(self):
        # ── Tkinter variables (user-facing settings) ──────────────────
        # Folder selection
        self.var_choose_folder = tk.StringVar()
        self.var_choose_folder_short = tk.StringVar()

        # Detection model
        self.var_det_model = tk.StringVar()
        self.var_det_model_short = tk.StringVar()
        self.var_det_model_path = tk.StringVar()

        # Classification model
        self.var_cls_model = tk.StringVar()

        # Thresholds
        self.var_cls_detec_thresh = tk.DoubleVar(value=0.6)
        self.var_cls_class_thresh = tk.DoubleVar(value=0.6)
        self.var_thresh = tk.DoubleVar(value=0.6)

        # Deploy options
        self.var_use_custom_img_size_for_deploy = tk.BooleanVar(value=False)
        self.var_image_size_for_deploy = tk.StringVar(value="1280")
        self.var_disable_GPU = tk.BooleanVar(value=False)
        self.var_process_img = tk.BooleanVar(value=True)
        self.var_use_checkpnts = tk.BooleanVar(value=False)
        self.var_cont_checkpnt = tk.BooleanVar(value=False)
        self.var_checkpoint_freq = tk.StringVar(value="500")
        self.var_process_vid = tk.BooleanVar(value=False)
        self.var_not_all_frames = tk.BooleanVar(value=False)
        self.var_nth_frame = tk.StringVar(value="10")

        # Postprocessing options
        self.var_separate_files = tk.BooleanVar(value=False)
        self.var_file_placement = tk.IntVar(value=2)
        self.var_sep_conf = tk.BooleanVar(value=False)
        self.var_vis_files = tk.BooleanVar(value=False)
        self.var_vis_size = tk.StringVar()
        self.var_vis_bbox = tk.BooleanVar(value=True)
        self.var_vis_blur = tk.BooleanVar(value=False)
        self.var_crp_files = tk.BooleanVar(value=False)
        self.var_exp = tk.BooleanVar(value=False)
        self.var_exp_format = tk.StringVar()
        self.var_plt = tk.BooleanVar(value=False)
        self.var_abs_paths = tk.BooleanVar(value=False)

        # Output directory
        self.var_output_dir = tk.StringVar()
        self.var_output_dir_short = tk.StringVar()

        # Classification extras
        self.var_smooth_cls_animal = tk.BooleanVar(value=False)
        self.var_keep_series_seconds = tk.DoubleVar(value=30.0)
        self.var_tax_fallback = tk.BooleanVar(value=True)
        self.var_exclude_subs = tk.BooleanVar(value=False)
        self.var_tax_levels = tk.StringVar()
        self.var_sppnet_location = tk.StringVar()

        # HITL
        self.var_hitl_file_order = tk.IntVar(value=1)

        # ── Non-widget mutable state (previously `global`) ───────────
        # Cancel/deploy
        self.cancel_var = tk.BooleanVar(value=False)
        self.cancel_deploy_model_pressed = False
        self.cancel_speciesnet_deploy_pressed = False
        self.subprocess_output = ""
        self.warn_smooth_vid = False
        self.temp_frame_folder = ""

        # Progress and error tracking
        self.progress_window = None
        self.postprocessing_error_log = []
        self.model_error_log = []
        self.model_warning_log = []
        self.model_special_char_log = []

        # HITL state
        self.selection_dict = {}

        # Dropdown option lists (rebuilt on language change / model refresh)
        self.dpd_options_cls_model = []
        self.dpd_options_model = []
        self.sim_dpd_options_cls_model = []
        self.loc_chkpnt_file = ""

        # Init flags
        self.checkpoint_freq_init = True
        self.image_size_for_deploy_init = True
        self.nth_frame_init = True
        self.shown_abs_paths_warning = False
        self.check_mark_one_row = False
        self.check_mark_two_rows = False

        # Timelapse integration
        self.timelapse_mode = False
        self.timelapse_path = ""

        # Caches
        self._all_supported_model_classes_cache = None

        # ── Widget references (set after UI construction) ─────────────
        self.btn_start_deploy = None
        self.sim_run_btn = None
        self.sim_dir_pth = None
        self.sim_mdl_dpd = None
        self.sim_spp_scr = None
        self.rad_ann_var = None  # tk.IntVar, set during HITL window build
        self.hitl_ann_selection_frame = None
        self.hitl_settings_canvas = None
        self.hitl_settings_window = None
        self.lbl_n_total_imgs = None
```

### Step-by-Step Execution

---

#### Step 4.1: Create `AppState` class (1 commit)

**File:** `addaxai/core/state.py`

1. Create the `AppState` class exactly as shown above.
2. Write tests in `tests/test_core_state.py`:
   - `AppState` is importable without tkinter (the class definition itself doesn't call `tk.*`)
   - Verify all expected attributes exist using `hasattr` checks against a known list
   - With a `tk.Tk()` root: instantiate `AppState()`, verify all tkinter vars are created
   - Verify default values: `state.cancel_deploy_model_pressed == False`,
     `state.timelapse_mode == False`, `state.model_error_log == []`, etc.

**Important:** Do NOT change `AddaxAI_GUI.py` in this step. Only create the new file and tests.

**Commit:** `feat: create AppState class in addaxai/core/state.py (Phase 4.1)`

---

#### Step 4.2: Instantiate AppState and migrate tkinter variables (1 commit)

This is the largest single step. It moves all ~50 tkinter variable declarations from module-level
code in `AddaxAI_GUI.py` into `AppState`.

**Process:**

1. Add import at top of `AddaxAI_GUI.py`:
   ```python
   from addaxai.core.state import AppState
   ```

2. Find where `root = customtkinter.CTk()` is called (around line 7700). Immediately after it, add:
   ```python
   state = AppState()
   ```

3. For each tkinter variable currently declared at module level (search for `= tk.StringVar`,
   `= tk.BooleanVar`, `= tk.IntVar`, `= tk.DoubleVar`):
   a. Delete the module-level declaration
   b. The variable is now accessed as `state.var_name` instead of bare `var_name`
   c. Find every reference to that variable and prefix with `state.`

4. This step requires updating MANY lines. Do it in sub-batches by variable group:
   - **4.2a:** Folder selection vars (`var_choose_folder`, `var_choose_folder_short`)
   - **4.2b:** Detection model vars (`var_det_model`, `var_det_model_short`, `var_det_model_path`)
   - **4.2c:** Classification model vars (`var_cls_model`)
   - **4.2d:** Threshold vars (`var_cls_detec_thresh`, `var_cls_class_thresh`, `var_thresh`)
   - **4.2e:** Deploy option vars (checkpoints, GPU, image processing toggles)
   - **4.2f:** Postprocessing option vars (separate, vis, crop, export, plots)
   - **4.2g:** Remaining vars (output dir, classification extras, HITL)

**CRITICAL WARNING:** Many tkinter vars are passed to widget constructors via `textvariable=` or
`variable=`. These MUST now reference `state.var_name`:
```python
# Before:
entry_choose_folder = customtkinter.CTkEntry(master=..., textvariable=var_choose_folder)
# After:
entry_choose_folder = customtkinter.CTkEntry(master=..., textvariable=state.var_choose_folder)
```

**Also update:** Functions that read/write these vars via `.get()` and `.set()`:
```python
# Before:
chosen_folder = var_choose_folder.get()
# After:
chosen_folder = state.var_choose_folder.get()
```

**Also update:** `build_simple_mode()` call — it receives several tkinter vars as params. These
must now come from `state`:
```python
# Before:
_sim = build_simple_mode(..., var_choose_folder=var_choose_folder, ...)
# After:
_sim = build_simple_mode(..., var_choose_folder=state.var_choose_folder, ...)
```

**Verification:** After each sub-batch:
- `python -m pytest tests/ -v` — all tests pass
- GUI smoke test — GUI starts without crash
- Manual check: can still select a folder, change model dropdown, toggle checkboxes

**Commit:** `refactor: migrate tkinter variables to AppState (Phase 4.2)`

---

#### Step 4.3: Migrate cancel flags and deployment state (1 commit)

Move these globals into `AppState`:
- `cancel_var`, `cancel_deploy_model_pressed`, `cancel_speciesnet_deploy_pressed`
- `subprocess_output`, `warn_smooth_vid`, `temp_frame_folder`
- `btn_start_deploy` (widget ref)

**Process:**

1. After `state = AppState()` and after `btn_start_deploy` is created in the UI construction,
   assign: `state.btn_start_deploy = btn_start_deploy`

2. For each function that declares `global cancel_deploy_model_pressed` (etc.):
   a. Remove the `global` declaration
   b. Add `state` as a parameter (or use module-level `state` — see note below)
   c. Replace bare `cancel_deploy_model_pressed` with `state.cancel_deploy_model_pressed`

**Module-level `state` vs parameter passing:**
Since `state` is a module-level singleton in `AddaxAI_GUI.py`, functions defined in the same file
can access it directly without needing it passed as a parameter. This is acceptable for Phase 4
because the goal is eliminating `global` declarations, not achieving full dependency injection.
Full DI would be Phase 5+ work.

So the pattern is:
```python
# Before:
def cancel_deployment(process):
    global cancel_deploy_model_pressed
    global btn_start_deploy
    global sim_run_btn
    cancel_deploy_model_pressed = True
    btn_start_deploy.configure(state=NORMAL)
    sim_run_btn.configure(state=NORMAL)

# After:
def cancel_deployment(process):
    state.cancel_deploy_model_pressed = True
    state.btn_start_deploy.configure(state=NORMAL)
    state.sim_run_btn.configure(state=NORMAL)
```

3. Functions affected (search for `global cancel_deploy_model_pressed`, `global cancel_var`, etc.):
   - `cancel_deployment()` — lines 2781-2792
   - `deploy_model()` — lines 2793-3109
   - `classify_detections()` — lines 2587-2780
   - `start_deploy()` — lines 3172-3914
   - `start_postprocess()` — lines 1217-1353
   - `cancel()` — lines 7289-7296
   - `SpeciesNetOutputWindow.cancel()` — lines 4714-4725

**Commit:** `refactor: migrate cancel/deployment globals to AppState (Phase 4.3)`

---

#### Step 4.4: Migrate HITL state globals (1 commit)

Move these globals into `AppState`:
- `selection_dict`, `rad_ann_var`
- `hitl_ann_selection_frame`, `hitl_settings_canvas`, `hitl_settings_window`
- `lbl_n_total_imgs`

**Process:**

1. For `selection_dict`: Replace `global selection_dict` with `state.selection_dict` in:
   - `open_hitl_settings_window()` — line 4261
   - `open_species_selection()` — line 6308
   - `enable_selection_widgets()` — line 6995

2. For `rad_ann_var` and the HITL widgets: These are created inside `open_hitl_settings_window()`.
   After creation, assign to `state`:
   ```python
   # Inside open_hitl_settings_window():
   state.rad_ann_var = tk.IntVar(value=1)
   state.hitl_ann_selection_frame = ...
   state.hitl_settings_canvas = ...
   state.hitl_settings_window = ...
   state.lbl_n_total_imgs = ...
   ```
   Then remove all `global` declarations for these variables.

3. Functions affected:
   - `open_hitl_settings_window()` — 6 globals declared
   - `toggle_hitl_ann_selection_frame()` — reads `hitl_ann_selection_frame`
   - `toggle_hitl_ann_selection()` — reads `rad_ann_var`, `hitl_ann_selection_frame`
   - `select_detections()` — reads `selection_dict`
   - `resize_canvas_to_content()` — reads `hitl_settings_canvas`

**Commit:** `refactor: migrate HITL state globals to AppState (Phase 4.4)`

---

#### Step 4.5: Migrate simple mode widget refs and dropdown options (1 commit)

Move these globals into `AppState`:
- `sim_dir_pth`, `sim_mdl_dpd`, `sim_run_btn`, `sim_spp_scr`
- `sim_dpd_options_cls_model`
- `dpd_options_cls_model`, `dpd_options_model`

**Process:**

1. After `build_simple_mode()` returns `_sim` dict, assign widget refs to `state`:
   ```python
   _sim = build_simple_mode(...)
   simple_mode_win = _sim['window']
   state.sim_dir_pth = _sim['dir_pth']
   state.sim_mdl_dpd = _sim['mdl_dpd']
   state.sim_run_btn = _sim['run_btn']
   state.sim_spp_scr = _sim['spp_scr']
   # ... etc
   ```

2. Remove all `global sim_dir_pth`, `global sim_mdl_dpd`, etc. declarations.

3. Replace bare `sim_dir_pth` references with `state.sim_dir_pth` in:
   - `browse_dir()` — line 5191
   - `update_frame_states()` — line 7199
   - `main()` — line 8611

4. For `dpd_options_cls_model` and `dpd_options_model`: These are rebuilt dynamically in
   `update_model_dropdowns()` (line 6795). Replace:
   ```python
   # Before:
   global dpd_options_cls_model
   dpd_options_cls_model = ...
   # After:
   state.dpd_options_cls_model = ...
   ```

**Commit:** `refactor: migrate simple mode and dropdown globals to AppState (Phase 4.5)`

---

#### Step 4.6: Migrate remaining globals (1 commit)

Move the init flags, timelapse state, and caches:
- `checkpoint_freq_init`, `image_size_for_deploy_init`, `nth_frame_init`
- `shown_abs_paths_warning`, `check_mark_one_row`, `check_mark_two_rows`
- `timelapse_mode`, `timelapse_path`
- `_ALL_SUPPORTED_MODEL_CLASSES_CACHE`, `loc_chkpnt_file`

**Process:**

1. For each init flag, find the function that declares it `global` and replace:
   ```python
   # Before (image_size_for_deploy_focus_in):
   global image_size_for_deploy_init
   if image_size_for_deploy_init:
       image_size_for_deploy_init = False
   # After:
   if state.image_size_for_deploy_init:
       state.image_size_for_deploy_init = False
   ```

2. For `timelapse_mode` and `timelapse_path`: Only set in `main()` from argparse:
   ```python
   # Before:
   global timelapse_mode
   global timelapse_path
   timelapse_mode = args.timelapse is not None
   # After:
   state.timelapse_mode = args.timelapse is not None
   state.timelapse_path = args.timelapse or ""
   ```

3. For `_ALL_SUPPORTED_MODEL_CLASSES_CACHE`: Used in `get_all_supported_model_classes()`:
   ```python
   # Before:
   global _ALL_SUPPORTED_MODEL_CLASSES_CACHE
   if _ALL_SUPPORTED_MODEL_CLASSES_CACHE is not None and not force_refresh:
       return _ALL_SUPPORTED_MODEL_CLASSES_CACHE
   # After:
   if state._all_supported_model_classes_cache is not None and not force_refresh:
       return state._all_supported_model_classes_cache
   ```

**After this step:** Run `grep -n "global " AddaxAI_GUI.py` — should return **zero** results.
If any remain, investigate and migrate them.

**Commit:** `refactor: migrate remaining globals to AppState (Phase 4.6)`

---

#### Step 4.7: Extract SpeciesNetOutputWindow (1 commit)

Now that globals are accessed via `state`, `SpeciesNetOutputWindow` can be extracted.

**File:** `addaxai/ui/dialogs/speciesnet_output.py`

**Current globals used by SpeciesNetOutputWindow:**
- `root` → pass as `master` parameter
- `cancel_speciesnet_deploy_pressed` → now `state.cancel_speciesnet_deploy_pressed`
- `btn_start_deploy` → now `state.btn_start_deploy`
- `sim_run_btn` → now `state.sim_run_btn`
- `bring_window_to_top_but_not_for_ever()` → pass as callback
- `remove_ansi_escape_sequences()` → import from `addaxai.utils.files`

**Extraction:**
```python
# addaxai/ui/dialogs/speciesnet_output.py
import os
import signal
import tkinter as tk
import customtkinter
from subprocess import Popen
from addaxai.utils.files import remove_ansi_escape_sequences


class SpeciesNetOutputWindow:
    def __init__(self, master=None, bring_to_top_func=None, on_cancel=None):
        """
        Args:
            master: parent tkinter window
            bring_to_top_func: callable to bring window to front
            on_cancel: callable(process) invoked when user clicks Cancel,
                       responsible for setting cancel flags and re-enabling buttons
        """
        self.on_cancel = on_cancel
        self.sppnet_output_window_root = customtkinter.CTkToplevel(master)
        self.sppnet_output_window_root.title("SpeciesNet output")
        self.text_area = tk.Text(self.sppnet_output_window_root, wrap=tk.WORD,
                                 height=7, width=85)
        self.text_area.pack(padx=10, pady=10)
        self.close_button = tk.Button(self.sppnet_output_window_root,
                                      text="Cancel", command=self.cancel)
        self.close_button.pack(pady=5)
        self.sppnet_output_window_root.protocol("WM_DELETE_WINDOW", self.close)
        if bring_to_top_func:
            bring_to_top_func(self.sppnet_output_window_root)

    def add_string(self, text, process=None):
        # ... move existing add_string method as-is ...
        # Replace remove_ansi_escape_sequences with imported version
        pass

    def close(self):
        self.sppnet_output_window_root.destroy()

    def cancel(self):
        if os.name == 'nt':
            Popen(f"TASKKILL /F /PID {self.process.pid} /T")
        else:
            import os as _os
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        if self.on_cancel:
            self.on_cancel(self.process)
        self.sppnet_output_window_root.destroy()
```

**Wiring in `AddaxAI_GUI.py`:**
```python
from addaxai.ui.dialogs.speciesnet_output import SpeciesNetOutputWindow

# Where SpeciesNetOutputWindow is instantiated (inside deploy_speciesnet or start_deploy):
def _on_speciesnet_cancel(process):
    state.btn_start_deploy.configure(state=NORMAL)
    state.sim_run_btn.configure(state=NORMAL)
    state.cancel_speciesnet_deploy_pressed = True

sppnet_output_window = SpeciesNetOutputWindow(
    master=root,
    bring_to_top_func=bring_window_to_top_but_not_for_ever,
    on_cancel=_on_speciesnet_cancel,
)
```

**Tests** (`tests/test_ui_speciesnet_output.py`):
- Import `SpeciesNetOutputWindow` without error
- Verify `__init__` accepts `master`, `bring_to_top_func`, `on_cancel` keywords
- Verify class has `add_string`, `close`, `cancel` methods

**Commit:** `refactor: extract SpeciesNetOutputWindow to addaxai/ui/dialogs/ (Phase 4.7)`

---

#### Step 4.8: Final audit (1 commit)

1. Run: `grep -n "global " AddaxAI_GUI.py`
   - Expected result: zero matches
   - If any remain, migrate them to `state`

2. Run: `grep -rn "global " addaxai/`
   - Only expected matches: `addaxai/i18n/__init__.py` (the `global _current, _strings` in
     `init()` and `set_language()` — these are acceptable module-level singletons)

3. Run full test suite: `.venv/Scripts/python -m pytest tests/ -v`

4. Run GUI smoke test: `C:\Users\Topam\AddaxAI_files\envs\env-base\python.exe -m pytest tests/test_gui_smoke.py -v`

5. Manual verification:
   - Launch via `dev_launch.py`
   - Select a folder, choose a model, toggle checkboxes
   - Switch languages (EN→ES→FR→EN)
   - Switch modes (advanced ↔ simple)
   - Open HITL settings window
   - Start a deploy (if test images available), verify progress window works
   - Cancel a deploy, verify buttons re-enable

**Commit:** `refactor: final audit — zero global declarations remain (Phase 4.8)`

---

### Verification After Each Step

1. `.venv/Scripts/python -m pytest tests/ -v` — all existing tests pass
2. `C:\Users\Topam\AddaxAI_files\envs\env-base\python.exe -m pytest tests/test_gui_smoke.py -v` — GUI starts
3. Manual: launch via `dev_launch.py`, verify basic interactions still work

### Risk Assessment

| Step | Risk | Reason |
|------|------|--------|
| 4.1 (create AppState) | **Low** | New file only, no changes to existing code |
| 4.2 (tkinter vars) | **High** | Touches ~200+ lines, every widget constructor and `.get()`/`.set()` call |
| 4.3 (cancel/deploy) | **Medium** | 7 functions affected, but pattern is mechanical (remove `global`, add `state.`) |
| 4.4 (HITL) | **Medium** | 6 functions affected, HITL window construction is complex |
| 4.5 (simple mode + dropdowns) | **Medium** | Wiring between `build_simple_mode()` return dict and `state` |
| 4.6 (remaining) | **Low** | Small number of isolated globals |
| 4.7 (SpeciesNetOutputWindow) | **Medium** | Cancel callback pattern is new, but class is self-contained |
| 4.8 (audit) | **Low** | Verification only |

**Mitigation:** Each step is one commit. If GUI breaks, `git revert` the last commit.
Step 4.2 is the riskiest — consider splitting into the 7 sub-batches (4.2a–4.2g) listed above.

### Expected Outcome

- **Zero `global` declarations** in `AddaxAI_GUI.py`
- All mutable state owned by a single `AppState` instance
- `SpeciesNetOutputWindow` extracted to `addaxai/ui/dialogs/`
- Every function's dependencies are explicit (reads `state.x` instead of invisible `global x`)
- Foundation for future testability: functions can be tested by constructing `AppState` with
  mock values
