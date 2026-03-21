# AddaxAI - Refactoring Plan

## What This App Is

AddaxAI helps ecologists classify camera trap images using local computer vision (MegaDetector + various classification models). It supports multiple languages (English, Spanish, French), runs fully offline, and is packaged via PyInstaller for non-technical users.

## Why We're Refactoring

The application originally lived in a single ~11,200-line file (`AddaxAI_GUI.py`) with no separation between business logic, UI, model deployment, data processing, and localization. All state was managed via `global` variables. Now at ~8,570 lines with ~3,750 lines extracted to 39 `addaxai/` modules. This makes the codebase:

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

### Phase 5: Polish (see detailed plan below)
- [x] 5.1a: Type hints — core/ modules
- [x] 5.1b: Type hints — utils/ modules
- [x] 5.1c: Type hints — processing/ modules
- [x] 5.1d: Type hints — models/ modules
- [x] 5.1e: Type hints — analysis/, i18n/, hitl/ modules
- [x] 5.1f: Type hints — ui/ modules
- [x] 5.2a: Logging infrastructure (`addaxai/core/logging.py`)
- [x] 5.2b: Replace prints in addaxai/ modules
- [x] 5.2c: Replace debug trace prints in AddaxAI_GUI.py
- [x] 5.2d: Replace remaining prints in AddaxAI_GUI.py
- [x] 5.2e: Verify zero print() calls remain
- [x] 5.3a: GitHub Actions workflow for unit tests
- [x] 5.3b: Add ruff linting to CI
- [x] 5.3c: Add mypy type checking to CI

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
**Unit tests:** 325 passing, 9 skipped (optional deps: cv2, matplotlib, customtkinter) — run with `.venv` Python 3.14
**Integration tests:** 8 passing (Tier 1 + Tier 2) — run with env-base Python 3.8
**GUI smoke test:** 1 passing — run with env-base Python 3.8
**Python (tests):** `C:\Users\Topam\AppData\Local\Python\bin\python.exe` (3.14)
**Python (GUI):** `C:\Users\Topam\AddaxAI_files\envs\env-base\python.exe`
**Installed test deps:** pytest, Pillow, numpy, pandas, requests

**Phases 0–4 fully complete.** Zero `global` declarations remain in `AddaxAI_GUI.py`.
- 58 backend functions extracted into 12 modules (Phase 1)
- i18n system: 662 `[lang_idx]` occurrences → `t()` calls, 3 JSON files (~300 keys each) (Phase 2)
- 7 widget/dialog/tab classes extracted to `addaxai/ui/` (Phase 3)
- `AppState` class owns all mutable state; 35 `global` declarations eliminated (Phase 4)
- `SpeciesNetOutputWindow` extracted to `addaxai/ui/dialogs/speciesnet_output.py` (Phase 4.7)

**Phase 5.1 complete** — all 39 `addaxai/` modules fully annotated with type hints (Python 3.8 compatible, `typing` module).
**Phase 5.2 complete** — zero `print()` calls remain; all output goes through `logging`; `addaxai.log` written to `AddaxAI_files/` during GUI runs.
**Phase 5.3 complete** — GitHub Actions CI runs unit tests (Python 3.9 + 3.11), ruff lint, and mypy type check on every push to `refactor/modularize` and PRs to `main`.

**Phase 5 fully complete.** The refactoring is done.

## Why AddaxAI_GUI.py Is Still ~8,570 Lines

After Phases 1–4, the main file is still large because what remains is **inherently GUI code** —
the kind of code that is tightly coupled to tkinter/customtkinter widget construction, layout,
and event wiring. Specifically:

| Category | ~Lines | Why it stays |
|----------|--------|-------------|
| Widget construction & layout | ~3,000 | Frame/label/button/dropdown creation with `.grid()` calls. Each widget is 3–8 lines of constructor + layout + binding. This is unavoidable boilerplate for any tkinter app. |
| Deployment orchestrator (`start_deploy`, `deploy_model`, `classify_detections`) | ~900 | Coordinates subprocess spawning, progress updates, error dialogs, cancel handling. The *pure logic* (subprocess kill, model lookup) was extracted; what remains is the UI-facing orchestration that reads tkinter vars, updates progress windows, and shows messageboxes. |
| HITL workflow | ~700 | `open_hitl_settings_window()` alone is ~400 lines of window construction + widget binding. The data logic was extracted; the UI construction must stay near the root window. |
| Postprocessing orchestrator | ~400 | Similar to deploy: reads UI state, calls extracted functions, updates progress. |
| Callbacks & event handlers | ~600 | ~25 toggle/focus/browse callbacks, each 5–30 lines. These read/write tkinter vars and show/hide frames — pure UI plumbing. |
| Model download/info dialogs | ~500 | Window construction for model info, download progress, release notes. |
| Module-level UI setup (after `state = AppState()`) | ~1,500 | The ~1,500 lines from `state = AppState()` through `main()` that build the root window, tabs, all frames, all widgets, and wire everything together. |
| `main()` + settings persistence | ~300 | Entry point, argparse, settings load/save, `mainloop()`. |

**The extractable surface is mostly exhausted.** Further reduction requires either:
- A future Phase 6 that moves orchestrators (`start_deploy`, `start_postprocess`, HITL) into controller classes — but these are so interleaved with messagebox calls and progress window updates that extraction would require introducing a callback/event system first.
- Moving widget construction into declarative builders (like the `build_simple_mode()` pattern used in Phase 3) — possible but diminishing returns since each builder still needs the same number of lines.

**Bottom line:** ~8,500 lines for a ~40-dialog, 3-language, multi-workflow desktop app with no UI framework abstraction layer is normal. The important metric is that every *testable* function has been extracted, globals are eliminated, and the file's complexity is now linear (read top-to-bottom) rather than tangled.

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

### Current integration tests (8 total)

**Tier 1 (basic UI state):**

| Test | What it does | What it catches |
|------|-------------|-----------------|
| `test_language_cycling` | Calls `set_language()` 3x (EN→ES→FR→EN), checks 12 advanced + 5 simple mode widget texts per language | Missing `t()` key, widget not updating on language switch |
| `test_mode_switching` | Toggles advanced↔simple twice, checks `advanced_mode` persisted value and window visibility | Broken mode toggle, window not showing/hiding |
| `test_folder_selection` | Sets `var_choose_folder` to a temp dir, calls `update_frame_states()` | Crash on folder selection, broken frame state wiring |

**Tier 2 (Phase 4 wiring validation):**

| Test | What it does | What it catches |
|------|-------------|-----------------|
| `test_model_dropdown_population` | Calls `update_model_dropdowns()`, asserts `state.dpd_options_model` and `state.dpd_options_cls_model` are non-empty 3-element lists | Phase 4.5 dropdown globals→state wiring |
| `test_toggle_frames` | Toggles `var_separate_files` and `var_vis_files` on/off, calls toggle callbacks | Broken `state.` references in toggle callbacks |
| `test_reset_values` | Sets 5 vars to non-defaults, calls `reset_values()`, asserts all revert | Missing `state.` prefix in reset logic |
| `test_deploy_validation` | Sets folder to empty temp dir, calls `start_deploy()` with patched `mb.showerror` | Deploy validation crashes after state migration |
| `test_state_attributes` | Checks 15 non-tkinter attrs, 7 tkinter vars, and 2 widget refs on `state` | AppState init defaults wrong, widget refs not assigned |

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

### Verification after each step

After every commit, run:
1. `.venv/Scripts/python -m pytest tests/ -v --ignore=tests/test_gui_smoke.py --ignore=tests/test_gui_integration.py` — unit tests pass
2. `C:\Users\Topam\AddaxAI_files\envs\env-base\python.exe -m pytest tests/test_gui_integration.py -v` — integration tests pass
3. `C:\Users\Topam\AddaxAI_files\envs\env-base\python.exe -m pytest tests/test_gui_smoke.py -v` — smoke test passes

If any test fails after a step, `git revert` the commit and investigate.

## Phase 5: Polish — Detailed Implementation Plan

### Overview

Phase 5 has three independent workstreams: type hints (5.1), logging (5.2), and CI (5.3).
They can be done in any order. Each step is one commit.

**Current state going in:**
- 39 modules in `addaxai/` (~3,750 lines) — only `i18n/__init__.py` has type hints
- 73 `print()` calls in `AddaxAI_GUI.py`, 11 in `addaxai/` modules
- No test-running CI (only PyInstaller release builds in `.github/workflows/`)
- 337 tests (319 unit + 8 integration + 1 smoke + 9 skipped)

**CRITICAL: Update CLAUDE.md after every commit** — update the Current Status section and mark
completed steps with `[x]`. This keeps future sessions oriented.

### Step 5.1: Type Hints on Extracted Modules

Add type annotations to all function signatures in `addaxai/` modules. Do NOT add type hints
to `AddaxAI_GUI.py` — it is not worth the effort for a file that is mostly UI construction code.

**Scope:** Only the 39 files under `addaxai/`. Estimated ~200 function signatures to annotate.

**Rules:**
- Add parameter types and return types to every `def` — no exceptions
- Use `typing` imports only when needed (`Optional`, `List`, `Dict`, `Tuple`, `Union`)
- For Python 3.8 compatibility: use `typing.List` not `list[str]`, `typing.Optional` not `X | None`
- For tkinter widget params, use `Any` — do not import widget types
- Do NOT add type hints to test files
- Do NOT add docstrings, comments, or make any behavioral changes — type hints only
- Run `python -m pytest tests/ -v` after each file to verify no regressions

**Execution order** (one commit per batch, roughly by dependency):

#### Step 5.1a: Core modules (1 commit)

Files: `core/config.py`, `core/paths.py`, `core/platform.py`, `core/state.py`

These are the foundation — other modules depend on them.

Example for `core/config.py`:
```python
# Before:
def load_global_vars(base_path):
    ...
    return global_vars

# After:
def load_global_vars(base_path: str) -> Dict[str, Any]:
    ...
    return global_vars
```

Example for `core/state.py`:
```python
# Before:
class AppState:
    def __init__(self):
        self.var_choose_folder = tk.StringVar()
        ...

# After:
class AppState:
    def __init__(self) -> None:
        self.var_choose_folder: tk.StringVar = tk.StringVar()
        ...
```

**Commit:** `refactor: add type hints to core/ modules (Phase 5.1a)`

#### Step 5.1b: Utility modules (1 commit)

Files: `utils/files.py`, `utils/images.py`, `utils/json_ops.py`, `utils/sorting.py`

These are pure functions with clear input/output types.

Example for `utils/files.py`:
```python
# Before:
def is_valid_float(value):

# After:
def is_valid_float(value: str) -> bool:
```

Example for `utils/images.py`:
```python
# Before:
def is_image_corrupted(fpath):

# After:
def is_image_corrupted(fpath: str) -> bool:
```

**Commit:** `refactor: add type hints to utils/ modules (Phase 5.1b)`

#### Step 5.1c: Processing modules (1 commit)

Files: `processing/annotations.py`, `processing/export.py`, `processing/postprocess.py`

Example for `processing/annotations.py`:
```python
# Before:
def create_pascal_voc_annotation(folder, filename, path, width, height, depth, objs):

# After:
def create_pascal_voc_annotation(
    folder: str, filename: str, path: str,
    width: int, height: int, depth: int,
    objs: List[Dict[str, Any]]
) -> str:
```

**Commit:** `refactor: add type hints to processing/ modules (Phase 5.1c)`

#### Step 5.1d: Models modules (1 commit)

Files: `models/registry.py`, `models/deploy.py`

Example for `models/registry.py`:
```python
# Before:
def fetch_known_models(model_dir):

# After:
def fetch_known_models(model_dir: str) -> List[str]:
```

Example for `models/deploy.py`:
```python
# Before:
def cancel_subprocess(process):

# After:
def cancel_subprocess(process: subprocess.Popen) -> None:
```

**Commit:** `refactor: add type hints to models/ modules (Phase 5.1d)`

#### Step 5.1e: Analysis, i18n, and HITL modules (1 commit)

Files: `analysis/plots.py`, `analysis/maps.py`, `i18n/__init__.py` (already partially typed),
`hitl/session.py`, `hitl/exchange.py`

Note: `i18n/__init__.py` already has hints — just verify and improve if needed.

**Commit:** `refactor: add type hints to analysis/, i18n/, hitl/ modules (Phase 5.1e)`

#### Step 5.1f: UI modules (1 commit)

Files: All files under `ui/widgets/`, `ui/dialogs/`, `ui/advanced/`, `ui/simple/`

For UI modules, many parameters are tkinter widgets — use `Any`:
```python
# Before:
def build_simple_mode(master, var_choose_folder, ...):

# After:
def build_simple_mode(master: Any, var_choose_folder: Any, ...) -> Dict[str, Any]:
```

**Commit:** `refactor: add type hints to ui/ modules (Phase 5.1f)`

---

### Step 5.2: Replace print() With Proper Logging

Replace all `print()` calls with Python's `logging` module. This affects `AddaxAI_GUI.py` (73
calls) and `addaxai/` modules (11 calls).

**Rules:**
- Use `logging.getLogger(__name__)` at the top of each file that logs
- Do NOT add logging to files that don't currently have `print()` calls
- Map print categories to log levels:
  - `print(f"EXECUTED: ...")` → `logger.debug(...)` (function entry tracing)
  - `print(f"ERROR: ...")` → `logger.error(...)` (error messages)
  - `print(command_args)` → `logger.debug(...)` (subprocess commands)
  - `print(line, end='')` → `logger.info(line.rstrip())` (subprocess output)
  - `print('This is an Apple Silicon system.')` → `logger.info(...)` (platform info)
  - `print(sys.path)` → `logger.debug(...)` (debug info)
- For `ui/dialogs/progress.py`: the 8 `lambda: print("")` are NOOP button command placeholders —
  replace with `lambda: None`
- Do NOT change any other behavior — same messages, just through logging
- Run all tests after each commit

**Execution order:**

#### Step 5.2a: Set up logging infrastructure (1 commit)

Create `addaxai/core/logging.py`:
```python
"""Logging setup for AddaxAI."""
import logging
import os
import sys


def setup_logging(log_dir: str = "", level: int = logging.INFO) -> None:
    """Configure root logger with console + optional file handler.

    Args:
        log_dir: Directory for log file. If empty, file logging is disabled.
        level: Logging level (default INFO).
    """
    root_logger = logging.getLogger("addaxai")
    root_logger.setLevel(level)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                            datefmt="%H:%M:%S")
    console.setFormatter(fmt)
    root_logger.addHandler(console)

    # File handler (optional)
    if log_dir and os.path.isdir(log_dir):
        fh = logging.FileHandler(os.path.join(log_dir, "addaxai.log"),
                                 encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        root_logger.addHandler(fh)
```

Write tests in `tests/test_core_logging.py`:
- `setup_logging()` is importable
- After calling `setup_logging()`, `logging.getLogger("addaxai")` has at least one handler
- Log messages at INFO level appear in captured output

**Commit:** `feat: add logging infrastructure in addaxai/core/logging.py (Phase 5.2a)`

#### Step 5.2b: Replace prints in addaxai/ modules (1 commit)

Only 11 `print()` calls across 4 files. Quick mechanical replacement:

- `ui/dialogs/progress.py`: Replace 8 `lambda: print("")` with `lambda: None`
- `ui/dialogs/speciesnet_output.py`: Add `logger = logging.getLogger(__name__)`, replace 1 print
- `core/config.py`: Add logger, replace 1 print
- `utils/images.py`: Add logger, replace 1 print

**Commit:** `refactor: replace print() with logging in addaxai/ modules (Phase 5.2b)`

#### Step 5.2c: Replace prints in AddaxAI_GUI.py — debug/trace category (1 commit)

Target: All `print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")` calls.
There are approximately 20+ of these.

At top of file, add:
```python
import logging
from addaxai.core.logging import setup_logging
logger = logging.getLogger("addaxai.gui")
```

In `main()`, before `root.mainloop()`, add:
```python
setup_logging(log_dir=AddaxAI_files)
```

Replace each `EXECUTED` print:
```python
# Before:
print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")

# After:
logger.debug("EXECUTED: %s", sys._getframe().f_code.co_name)
```

Note: Remove `({locals()})` from the log message — it dumps all local variables including large
data structures, which is too noisy even for debug. Just log the function name.

**Commit:** `refactor: replace debug trace prints with logging in AddaxAI_GUI.py (Phase 5.2c)`

#### Step 5.2d: Replace prints in AddaxAI_GUI.py — errors and info (1 commit)

Target: All remaining `print()` calls (~50). Categories:

- Error prints: `print("ERROR:\n" + str(error) ...)` → `logger.error("...", exc_info=True)`
- Subprocess output: `print(line, end='')` → `logger.info(line.rstrip())`
- Platform info: `print('This is an Apple Silicon system.')` → `logger.info(...)`
- Command args: `print(command_args)` → `logger.debug("Command: %s", command_args)`
- Debug info: `print(sys.path)` → `logger.debug("sys.path: %s", sys.path)`

**Commit:** `refactor: replace remaining prints with logging in AddaxAI_GUI.py (Phase 5.2d)`

#### Step 5.2e: Verify zero print() calls remain (1 commit)

Run: `grep -n "print(" AddaxAI_GUI.py addaxai/**/*.py`

Expected: zero matches (or only inside string literals / comments).

If any remain, replace them. Update CLAUDE.md Current Status.

**Commit:** `refactor: final print() audit — zero remaining (Phase 5.2e)`

---

### Step 5.3: CI Setup (Lint + Tests in GitHub Actions)

Create a GitHub Actions workflow that runs on every push to `refactor/modularize` and on PRs
to `main`. The CI uses `.venv` Python (latest stable) for unit tests only — GUI/integration tests
require env-base and a display, so they cannot run in CI.

#### Step 5.3a: Create CI workflow (1 commit)

Create `.github/workflows/test.yml`:
```yaml
name: Tests

on:
  push:
    branches: [refactor/modularize, main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.11"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pytest numpy pandas requests Pillow

      - name: Run unit tests
        run: |
          python -m pytest tests/ -v \
            --ignore=tests/test_gui_smoke.py \
            --ignore=tests/test_gui_integration.py \
            --ignore=tests/gui_test_runner.py
```

Note: Tests that require `cv2`, `matplotlib`, or `customtkinter` will be automatically skipped
via the existing `pytest.mark.skipif` guards.

Write no new tests — just verify the workflow runs.

**Commit:** `ci: add GitHub Actions workflow for unit tests (Phase 5.3a)`

#### Step 5.3b: Add linting to CI (1 commit)

Add `ruff` linting to the workflow. Ruff is fast, opinionated, and replaces flake8+isort+pyflakes.

Add to `.github/workflows/test.yml`:
```yaml
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install ruff
      - name: Lint addaxai/
        run: ruff check addaxai/
```

Create `ruff.toml` in repo root:
```toml
target-version = "py38"
line-length = 120

[lint]
select = ["E", "F", "W"]  # basic errors, pyflakes, warnings
ignore = ["E501"]  # line length — not enforced yet

[lint.per-file-ignores]
"addaxai/ui/*" = ["E402"]  # late imports OK in UI modules
```

Fix any lint errors that `ruff check addaxai/` flags before committing. Common fixes:
- Unused imports → remove them
- Undefined names → add missing imports
- Bare `except:` → `except Exception:`

Do NOT lint `AddaxAI_GUI.py` — it would produce hundreds of warnings. Scope is `addaxai/` only.

**Commit:** `ci: add ruff linting for addaxai/ modules (Phase 5.3b)`

#### Step 5.3c: Add type checking to CI (1 commit)

Add `mypy` to the workflow in permissive mode. This catches obvious type errors without requiring
full strictness.

Add to `.github/workflows/test.yml`:
```yaml
  typecheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install mypy
      - name: Type check addaxai/
        run: mypy addaxai/ --ignore-missing-imports --no-strict-optional
```

Fix any type errors mypy flags. Common fixes:
- Missing return type on `__init__` → add `-> None`
- Incompatible types in assignment → fix or add `# type: ignore` with explanation

**Commit:** `ci: add mypy type checking for addaxai/ modules (Phase 5.3c)`

---

### Verification After Each Step

1. `.venv/Scripts/python -m pytest tests/ -v --ignore=tests/test_gui_smoke.py --ignore=tests/test_gui_integration.py` — unit tests pass
2. `C:\Users\Topam\AddaxAI_files\envs\env-base\python.exe -m pytest tests/test_gui_integration.py -v` — integration tests pass
3. `C:\Users\Topam\AddaxAI_files\envs\env-base\python.exe -m pytest tests/test_gui_smoke.py -v` — smoke test passes
4. **Update CLAUDE.md** — mark completed steps with `[x]`, update Current Status section

If any test fails, `git revert` the commit and investigate.

### Risk Assessment

| Step | Risk | Reason |
|------|------|--------|
| 5.1a–f (type hints) | **Low** | Annotations don't change behavior; tests catch import errors |
| 5.2a (logging infra) | **Low** | New file only, no existing code changed |
| 5.2b (module prints) | **Low** | Only 11 prints in 4 files |
| 5.2c (GUI debug prints) | **Medium** | Touching ~20 lines across deployment/HITL functions; easy to miss one |
| 5.2d (GUI remaining prints) | **Medium** | ~50 replacements; subprocess output logging must preserve line-by-line behavior |
| 5.2e (print audit) | **Low** | Verification only |
| 5.3a (CI tests) | **Low** | New file; may need to fix import paths for CI environment |
| 5.3b (ruff lint) | **Medium** | May flag issues that need fixing before lint passes |
| 5.3c (mypy) | **Medium** | May find type inconsistencies that need `# type: ignore` |

### Expected Outcome

- All `addaxai/` functions have type annotations (parameter + return types)
- Zero `print()` calls remain — all output goes through `logging`
- Log file written to `AddaxAI_files/addaxai.log` during GUI runs
- GitHub Actions runs unit tests + lint + type check on every push
- CI blocks merges if tests fail or lint errors are introduced
