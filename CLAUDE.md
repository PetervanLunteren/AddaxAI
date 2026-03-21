# AddaxAI - Refactoring Plan

## What This App Is

AddaxAI helps ecologists classify camera trap images using local computer vision (MegaDetector + various classification models). It supports multiple languages (English, Spanish, French), runs fully offline, and is packaged via PyInstaller for non-technical users.

## Why We're Refactoring

The application (~10,400 lines, down from ~11,200) lives in a single file (`AddaxAI_GUI.py`) with no separation between business logic, UI, model deployment, data processing, and localization. All state is managed via `global` variables. This makes the codebase:

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

### Phase 4: Kill Global State
- [ ] 4.1: Create AppState class holding all tkinter variables
- [ ] 4.2: Pass AppState to components, remove all `global` declarations

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
**Tests:** 205 passing, 3 skipped (optional deps: cv2, matplotlib)
**Python (tests):** `C:\Users\Topam\AppData\Local\Python\bin\python.exe` (3.14)
**Python (GUI):** `C:\Users\Topam\AddaxAI_files\envs\env-base\python.exe`
**Installed test deps:** pytest, Pillow, numpy, pandas, requests

**Phase 3 fully complete!** AddaxAI_GUI.py reduced from 10,323 → 8,624 lines (~1,700 lines removed).
- 7 widget/dialog/tab classes extracted to `addaxai/ui/widgets/`, `addaxai/ui/dialogs/`, `addaxai/ui/advanced/`, `addaxai/ui/simple/`
- All extracted classes importable and testable without customtkinter (stub base class pattern)
- `SpeciesNetOutputWindow` (325 lines, deeply entangled) remains in `AddaxAI_GUI.py` — deferred to Phase 4
- GUI smoke test passes at every step; language switching and mode switching work

**Next:** Phase 4 (kill global state) — create AppState class, pass to components, remove `global` declarations.
See "Phase 4: Kill Global State" above for steps.

## Phase 2: Localization — Detailed Implementation Plan

### Overview

AddaxAI supports 3 languages (English, Spanish, French). Currently, translations are scattered
throughout `AddaxAI_GUI.py` as inline arrays indexed by a global `lang_idx` (0=EN, 1=ES, 2=FR).

**There are 662 occurrences of `[lang_idx]`** in the file, broken into 3 categories:
1. **~80 module-level `_txt` variables** (lines 219–235, 8992–9623): Named arrays like `browse_txt = ['Browse', 'Examinar', 'Parcourir']`, used as `browse_txt[lang_idx]`
2. **~5 `dpd_options_*` variables** (dropdown option lists): Arrays of arrays like `dpd_options_vis_size = [["Extra small", ...], ["Muy pequeño", ...], ["Très petit", ...]]`, used as `dpd_options_vis_size[lang_idx]`
3. **~136 inline anonymous arrays**: Patterns like `["Warning", "Advertencia", "Avertissement"][lang_idx]` scattered throughout function bodies

### How Language Switching Currently Works

1. On startup: `lang_idx = global_vars["lang_idx"]` (line 218) reads saved preference
2. `set_language()` (line 8107) cycles through languages: increments `lang_idx`, saves via `write_global_vars()`, then manually updates every UI widget's text by re-indexing the `_txt` arrays
3. All translation arrays stay in memory — switching just changes which index is read

### Target Design

```python
# addaxai/i18n/__init__.py
import json, os

_strings = {}     # {"en": {...}, "es": {...}, "fr": {...}}
_current = "en"   # current language code
_LANG_CODES = ["en", "es", "fr"]

def init(lang_idx=0):
    """Load all language JSON files. Call once at startup."""
    global _strings, _current
    base = os.path.dirname(__file__)
    for code in _LANG_CODES:
        with open(os.path.join(base, f"{code}.json"), encoding="utf-8") as f:
            _strings[code] = json.load(f)
    _current = _LANG_CODES[lang_idx]

def set_language(lang_idx):
    """Switch current language."""
    global _current
    _current = _LANG_CODES[lang_idx]

def t(key):
    """Look up a translation string by key."""
    return _strings[_current][key]

def lang_idx():
    """Return current language index (for backward compatibility)."""
    return _LANG_CODES.index(_current)
```

### JSON Structure

```json
// en.json
{
  "step": "Step",
  "browse": "Browse",
  "cancel": "Cancel",
  "warning": "Warning",
  "error": "Error",

  "lbl_choose_folder": "Source folder",
  "lbl_model": "Model to detect animals, vehicles, and persons",
  "btn_start_deploy": "Start processing",

  "dpd_vis_size": ["Extra small", "Small", "Medium", "Large", "Extra large"],
  "dpd_exp_format": ["XLSX", "CSV", "COCO", "Sensing Clues (TSV)"],

  "msg_no_model_output": "No model output file present. Make sure you run step 2 before post-processing the files.",
  "msg_verification_in_progress_title": "Verification session in progress",
  "msg_verification_in_progress_body": "Your verification session is not yet done..."
}
```

Keys follow this naming convention:
- `step`, `browse`, `cancel` — short reusable words (from the module-level `_txt` vars, drop `_txt` suffix)
- `lbl_*` — label texts (keep the existing variable name prefix)
- `btn_*` — button texts
- `dpd_*` — dropdown option lists (these are arrays, not strings)
- `msg_*` — message box texts (titles and bodies)
- `adv_*` — advanced mode specific texts
- `sim_*` — simple mode specific texts

### Step-by-Step Execution

#### Step 2.1: Build JSON files and `t()` function (1 commit)

1. **Extract all translations**: Grep `AddaxAI_GUI.py` for every `_txt = [` and `dpd_options_` definition. For each, create a key name and add entries to en.json, es.json, fr.json.

2. **Extract inline arrays**: Grep for `["...", "...", "..."][lang_idx]`. For each unique inline array, create a key name and add to the JSON files. Many inline arrays are duplicated (same text appears multiple times) — use a single key for all occurrences.

3. **Create `addaxai/i18n/__init__.py`** with `init()`, `set_language()`, `t()`, and `lang_idx()` functions as shown above.

4. **Write tests** in `tests/test_i18n.py`:
   - All 3 JSON files load without error
   - All 3 JSON files have identical key sets
   - `t("browse")` returns "Browse" when lang is EN, "Examinar" for ES, "Parcourir" for FR
   - `set_language()` switches correctly
   - `lang_idx()` returns correct index
   - No key in en.json has an empty string value
   - dpd_* keys return lists, not strings

5. **Commit**: `feat: create i18n system with translation JSON files (Phase 2.1)`

#### Step 2.2: Replace module-level `_txt` variables (1 commit)

These are the ~80 variables defined at module level like `browse_txt = ['Browse', 'Examinar', 'Parcourir']`.

**Before:**
```python
browse_txt = ['Browse', 'Examinar', 'Parcourir']
# ... later in code ...
btn_choose_folder.configure(text=browse_txt[lang_idx])
```

**After:**
```python
# browse_txt definition DELETED
# ... later in code ...
btn_choose_folder.configure(text=t("browse"))
```

**Process:**
1. Add `from addaxai.i18n import init as i18n_init, t, set_language as i18n_set_language` to imports in AddaxAI_GUI.py
2. Add `i18n_init(lang_idx)` right after `lang_idx = global_vars["lang_idx"]` (line 218)
3. For each `_txt` variable:
   a. Note the key name (e.g., `browse_txt` → key `"browse"`)
   b. Delete the variable definition line
   c. Find every usage like `browse_txt[lang_idx]` and replace with `t("browse")`
   d. Some are used without `[lang_idx]` (e.g., passed as whole arrays to `update_dpd_options`) — handle these separately in Step 2.3
4. Run full test suite + smoke test after each sub-batch

**IMPORTANT gotcha**: Some `_txt` variables are used in TWO ways:
- `fst_step_txt[lang_idx]` — needs `t("fst_step")` replacement
- `lbl_hitl_main_txt[lang_idx]` inside f-strings in inline arrays (line 272-276) — these are in messages that reference label text. The inline array itself will be handled in Step 2.4, but the `_txt[lang_idx]` reference inside it also needs to become `t("lbl_hitl_main")`.

**Sub-batches** (commit after each, run tests):
- 2.2a: The 17 small reusable words (lines 219–235): `step_txt`, `browse_txt`, `cancel_txt`, `change_folder_txt`, `view_results_txt`, `custom_model_txt`, `again_txt`, `eg_txt`, `show_txt`, `new_project_txt`, `warning_txt`, `information_txt`, `error_txt`, `select_txt`, `invalid_value_txt`, `none_txt`, `of_txt`, `suffixes_for_sim_none`
- 2.2b: Step/section labels (lines 8992–9052): `adv_btn_switch_mode_txt`, `adv_btn_sponsor_txt`, `adv_btn_reset_values_txt`, `adv_abo_lbl_txt`, `fst_step_txt`, `lbl_choose_folder_txt`
- 2.2c: Detection/classification labels (lines 9064–9242)
- 2.2d: Postprocessing labels (lines 9332–9623)
- 2.2e: Any remaining `_txt` variables

#### Step 2.3: Replace `dpd_options_*` arrays (1 commit)

These are dropdown option arrays: `dpd_options_model`, `dpd_options_cls_model`, `dpd_options_vis_size`, `dpd_options_exp_format`, `dpd_options_sppnet_location`, `dpd_options_tax_levels`.

**Special complexity**: These are arrays of arrays (one sub-array per language). Some contain dynamic content (model lists) mixed with translated labels.

**Before:**
```python
dpd_options_vis_size = [["Extra small", "Small", "Medium", "Large", "Extra large"],
                        ["Muy pequeño", "Pequeño", "Mediano", "Grande", "Muy grande"],
                        ["Très petit", "Petit", "Moyen", "Grand", "Très grand"]]
# used as:
dpd_options_vis_size[lang_idx]  # returns the current language's list
```

**After:**
```python
# dpd_options_vis_size definition DELETED
# used as:
t("dpd_vis_size")  # returns the current language's list from JSON
```

**Dynamic dropdowns** (`dpd_options_model`, `dpd_options_cls_model`): These mix model names (not translated) with a translated suffix like "Custom model"/"Otro modelo". Handle by storing only the translated suffix in JSON, and constructing the full list at runtime:
```python
# Before:
dpd_options_model = [det_models + ["Custom model"], det_models + ["Otro modelo"], det_models + ["Modèle personnalisé"]]
# After:
dpd_options_model = det_models + [t("custom_model")]  # rebuilt when language changes
```

#### Step 2.4: Replace inline anonymous arrays (3-4 commits)

136 occurrences of `["English", "Spanish", "French"][lang_idx]`. Group by function/section.

**Before:**
```python
mb.showerror(["Error", "Error", "Erreur"][lang_idx],
             ["No model output file present...", "Ningún archivo...", "Aucun fichier..."][lang_idx])
```

**After:**
```python
mb.showerror(t("error"), t("msg_no_model_output"))
```

**Process:**
1. For each inline array, check if it duplicates an existing key (many `["Error","Error","Erreur"]` arrays are the same as `error_txt` which became `t("error")` in Step 2.2)
2. If unique, add a new key to all 3 JSON files
3. Replace the inline array with `t("key_name")`

**Sub-batches by function** (commit after each):
- 2.4a: `postprocess()` function (~lines 244–860) — ~20 inline arrays
- 2.4b: `deploy_model()` and `classify_detections()` — ~25 inline arrays
- 2.4c: HITL functions — ~15 inline arrays
- 2.4d: Remaining scattered inline arrays — ~76 inline arrays

**Multiline inline arrays**: Some messages span multiple lines with f-strings. These need careful extraction:
```python
# Before (lines 271-277):
mb.askyesno(["Verification session in progress", "Sesión de verificación en curso",
             "Session de vérification en cours"][lang_idx],
            [f"Your verification session is not yet done. You can finish the session by clicking 'Continue' at '{lbl_hitl_main_txt[lang_idx]}', "
             "or just continue post-processing with the current results.\n\nDo you wish to continue post-processing?",
             f"La sesión de verificación aún no ha finalizado...",
             f"Votre session de vérification n'est pas encore terminée..."][lang_idx])
# After:
mb.askyesno(t("msg_verification_in_progress_title"),
            t("msg_verification_in_progress_body").format(label=t("lbl_hitl_main")))
```

For messages with embedded variables, use Python `.format()` placeholders in the JSON:
```json
{
  "msg_verification_in_progress_body": "Your verification session is not yet done. You can finish the session by clicking 'Continue' at '{label}', or just continue post-processing with the current results.\n\nDo you wish to continue post-processing?"
}
```

#### Step 2.5: Update `set_language()` (1 commit)

The current `set_language()` function (line 8107) manually re-indexes every `_txt` variable. After Steps 2.2–2.4, it should just call `i18n_set_language(new_idx)` and re-configure widgets using `t()`.

**Before (80+ lines):**
```python
def set_language():
    global lang_idx
    ...
    lang_idx = to_lang_idx
    write_global_vars(AddaxAI_files, {"lang_idx": lang_idx})
    fst_step.configure(text=" " + fst_step_txt[lang_idx] + " ")
    lbl_choose_folder.configure(text=lbl_choose_folder_txt[lang_idx])
    # ... 80 more widget updates ...
```

**After:**
```python
def set_language():
    global lang_idx
    ...
    lang_idx = to_lang_idx
    i18n_set_language(lang_idx)
    write_global_vars(AddaxAI_files, {"lang_idx": lang_idx})
    fst_step.configure(text=" " + t("fst_step") + " ")
    lbl_choose_folder.configure(text=t("lbl_choose_folder"))
    # ... same widget updates but with t() ...
```

This step should be mostly mechanical: every `var_txt[lang_idx]` → `t("var")`.

#### Step 2.6: Remove `lang_idx` global (1 commit)

After all `[lang_idx]` usages are replaced with `t()` calls:
1. Remove `global lang_idx` declarations
2. Replace remaining `lang_idx` reads with `from addaxai.i18n import lang_idx as get_lang_idx` and call `get_lang_idx()`
3. The only place that *sets* `lang_idx` should be `set_language()`, which calls `i18n_set_language()`
4. Verify: grep for `lang_idx` — should only appear in `set_language()` and `i18n_init()` call

### Verification After Each Step

1. `.venv/Scripts/python -m pytest tests/ -v` — all tests pass
2. `C:\Users\Topam\AddaxAI_files\envs\env-base\python.exe -m pytest tests/test_gui_smoke.py -v` — GUI starts
3. Manual: launch via `dev_launch.py`, click the language button 3 times to cycle EN→ES→FR→EN, verify all labels change correctly

### Risk Assessment

- **Low risk**: Steps 2.1 (new files only), 2.2a (small reusable words)
- **Medium risk**: Steps 2.2b-e, 2.3 (lots of search-and-replace, easy to miss one)
- **Higher risk**: Step 2.4 (multiline f-string arrays with embedded variables need careful `.format()` conversion), Step 2.6 (removing the global)
- **Mitigation**: Each sub-step is one commit. If the GUI breaks, `git revert` the last commit.

### Expected Outcome

- `AddaxAI_GUI.py` loses ~200 lines of translation variable definitions
- 662 `[lang_idx]` occurrences → 0
- All translations live in 3 JSON files (~300 keys each)
- Adding a 4th language = adding one JSON file + one entry in `_LANG_CODES`
- `lang_idx` global eliminated

## Phase 3: Restructure UI — Detailed Implementation Plan

### Overview

Phase 3 extracts UI classes from the 10,323-line `AddaxAI_GUI.py` into the `addaxai/ui/` package.
The goal is **mechanical extraction only** — move existing classes/code to separate files, parameterize
any globals they reference, then import them back. No behavioral changes.

**Current state of `addaxai/ui/`:**
```
ui/
├── __init__.py          (empty)
├── advanced/
│   └── __init__.py      (empty)
├── dialogs/
│   └── __init__.py      (empty)
├── simple/
│   └── __init__.py      (empty)
└── widgets/
    └── __init__.py      (empty)
```

### Key Globals Referenced by UI Classes

All classes currently access these module-level variables from `AddaxAI_GUI.py`. When extracting,
each must be passed as a parameter (typically to `__init__`) or imported:

| Variable | What it is | Used by |
|----------|-----------|---------|
| `root` | Main `customtkinter.CTk()` window | All dialog classes (parent window) |
| `scale_factor` | DPI scaling multiplier (float) | `MyMainFrame`, `EnvDownloadProgressWindow`, `ModelDownloadProgressWindow`, `ProgressWindow`, `model_info_frame`, `donation_popup_frame` |
| `PADX`, `PADY` | Padding constants (ints) | All widget/dialog classes |
| `green_primary` | Color hex string | `EnvDownloadProgressWindow`, `ModelDownloadProgressWindow`, `ProgressWindow` |
| `yellow_primary`, `yellow_secondary`, `yellow_tertiary` | Color hex strings | `GreyTopButton` |
| `GREY_BUTTON_BORDER_WIDTH` | Border width int | `GreyTopButton` |
| `i18n_lang_idx()`, `t()` | Localization functions | `EnvDownloadProgressWindow`, `ModelDownloadProgressWindow`, `ProgressWindow`, `PatienceDialog`, `SpeciesSelectionFrame` |
| `CancelButton` | Button class | `ProgressWindow` (uses it internally) |
| `bring_window_to_top_but_not_for_ever()` | Utility function | `TextButtonWindow`, `show_result_info()` |

### Strategy: Pass Globals as Parameters

**For constants** (`PADX`, `PADY`, `scale_factor`, colors): Create a `UIConstants` namedtuple or
simple class and pass it to constructors. This avoids 6+ individual parameters.

```python
# addaxai/ui/widgets/constants.py
from collections import namedtuple

UIConstants = namedtuple('UIConstants', [
    'PADX', 'PADY', 'scale_factor',
    'green_primary', 'yellow_primary', 'yellow_secondary', 'yellow_tertiary',
    'GREY_BUTTON_BORDER_WIDTH',
])
```

**For `root`**: Pass as the `master` / first positional argument (already done for most classes).

**For `t()` and `i18n_lang_idx()`**: Import directly from `addaxai.i18n` — these are already a module.

### Step-by-Step Execution

---

#### Step 3.1: Extract Widget Classes → `addaxai/ui/widgets/` (1 commit)

**What to extract** (all from `AddaxAI_GUI.py`):

| Class | Lines | Description |
|-------|-------|-------------|
| `MyMainFrame` | 6492–6500 | CTkFrame with 2-column layout, respects `scale_factor` |
| `MySubFrame` | 6502–6506 | CTkFrame with 2×250px columns |
| `MySubSubFrame` | 6508–6510 | Plain CTkFrame subclass |
| `InfoButton` | 6512–6519 | Styled CTkButton (grey, no hover, small) |
| `CancelButton` | 7309–7316 | Styled CTkButton for cancel actions |
| `GreyTopButton` | 8730–8738 | Styled CTkButton for top toolbar |
| `SpeciesSelectionFrame` | 6572–6599 | Scrollable checkbox list for species |

**Target files:**

1. `addaxai/ui/widgets/constants.py` — `UIConstants` namedtuple
2. `addaxai/ui/widgets/frames.py` — `MyMainFrame`, `MySubFrame`, `MySubSubFrame`
3. `addaxai/ui/widgets/buttons.py` — `InfoButton`, `CancelButton`, `GreyTopButton`
4. `addaxai/ui/widgets/species_selection.py` — `SpeciesSelectionFrame`

**Detailed instructions for each class:**

**`MyMainFrame`** (lines 6492–6500):
```python
# BEFORE in AddaxAI_GUI.py:
class MyMainFrame(customtkinter.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        if scale_factor != 1.0:  # <-- references global scale_factor
            self.columnconfigure(0, weight=1, minsize=70 * round(scale_factor * 1.35, 2))
            self.columnconfigure(1, weight=1, minsize=350 * round(scale_factor * 1.35, 2))
        else:
            self.columnconfigure(0, weight=1, minsize=70)
            self.columnconfigure(1, weight=1, minsize=350)

# AFTER in addaxai/ui/widgets/frames.py:
import customtkinter

class MyMainFrame(customtkinter.CTkFrame):
    def __init__(self, master, scale_factor=1.0, **kwargs):
        super().__init__(master, **kwargs)
        if scale_factor != 1.0:
            self.columnconfigure(0, weight=1, minsize=70 * round(scale_factor * 1.35, 2))
            self.columnconfigure(1, weight=1, minsize=350 * round(scale_factor * 1.35, 2))
        else:
            self.columnconfigure(0, weight=1, minsize=70)
            self.columnconfigure(1, weight=1, minsize=350)
```

**`MySubFrame`** (lines 6502–6506): No globals — move as-is.

**`MySubSubFrame`** (lines 6508–6510): No globals — move as-is.

**`InfoButton`** (lines 6512–6519): No globals — move as-is.

**`CancelButton`** (lines 7309–7316): No globals — move as-is.

**`GreyTopButton`** (lines 8730–8738):
```python
# BEFORE: references yellow_secondary, yellow_tertiary, GREY_BUTTON_BORDER_WIDTH globals
class GreyTopButton(customtkinter.CTkButton):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color = (yellow_secondary, "#333333"),
                       hover_color = (yellow_tertiary, "#2B2B2B"),
                       text_color = ("black", "white"),
                       height = 10, width = 140,
                       border_width=GREY_BUTTON_BORDER_WIDTH)

# AFTER: add color params with defaults
class GreyTopButton(customtkinter.CTkButton):
    def __init__(self, master, yellow_secondary="#F5E6A3", yellow_tertiary="#EBD98A",
                 border_width=0, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color = (yellow_secondary, "#333333"),
                       hover_color = (yellow_tertiary, "#2B2B2B"),
                       text_color = ("black", "white"),
                       height = 10, width = 140,
                       border_width=border_width)
```
NOTE: Look up the actual color hex values for `yellow_secondary` and `yellow_tertiary` in
`AddaxAI_GUI.py` (search for `yellow_secondary =` near the top). Use those as defaults.

**`SpeciesSelectionFrame`** (lines 6572–6599): References `i18n_lang_idx()` and `PADY`:
```python
# AFTER in addaxai/ui/widgets/species_selection.py:
import customtkinter
from addaxai.i18n import lang_idx as i18n_lang_idx

class SpeciesSelectionFrame(customtkinter.CTkScrollableFrame):
    def __init__(self, master, all_classes=[], selected_classes=[], command=None,
                 dummy_spp=False, pady=2, **kwargs):
        super().__init__(master, **kwargs)
        self.dummy_spp = dummy_spp
        if dummy_spp:
            all_classes = [f"{['Species', 'Especies', 'Espèces'][i18n_lang_idx()]} {i + 1}" for i in range(10)]
        self.command = command
        self.checkbox_list = []
        self.selected_classes = selected_classes
        for item in all_classes:
            self.add_item(item, pady)
    # ... rest of methods, pass pady through
```

**Wiring back in `AddaxAI_GUI.py`:**
```python
# At top of file, add:
from addaxai.ui.widgets.frames import MyMainFrame, MySubFrame, MySubSubFrame
from addaxai.ui.widgets.buttons import InfoButton, CancelButton, GreyTopButton
from addaxai.ui.widgets.species_selection import SpeciesSelectionFrame

# Delete the original class definitions (lines 6492–6519, 6572–6599, 7309–7316, 8730–8738)

# Update MyMainFrame calls to pass scale_factor:
#   MyMainFrame(master=simple_main_frame) → MyMainFrame(master=simple_main_frame, scale_factor=scale_factor)
# Update GreyTopButton calls to pass colors:
#   GreyTopButton(master=...) → GreyTopButton(master=..., yellow_secondary=yellow_secondary, yellow_tertiary=yellow_tertiary, border_width=GREY_BUTTON_BORDER_WIDTH)
```

**Tests** (`tests/test_ui_widgets.py`):
- Import all 7 classes without error (no tkinter needed — just verify import)
- Verify `MyMainFrame.__init__` accepts `scale_factor` keyword
- Verify `GreyTopButton.__init__` accepts color keywords
- Verify `SpeciesSelectionFrame.__init__` accepts `dummy_spp` keyword

**Commit**: `refactor: extract widget classes to addaxai/ui/widgets/ (Phase 3.1)`

---

#### Step 3.2: Extract Dialog Classes → `addaxai/ui/dialogs/` (1 commit)

**What to extract:**

| Class | Lines | Globals Referenced |
|-------|-------|--------------------|
| `TextButtonWindow` | 7121–7155 | `root`, `bring_window_to_top_but_not_for_ever()` |
| `PatienceDialog` | 7158–7187 | `root`, `t()` |
| `CustomWindow` | 7190–7208 | `root` |
| `EnvDownloadProgressWindow` | 6605–6677 | `root`, `scale_factor`, `PADX`, `PADY`, `i18n_lang_idx()`, `green_primary`, `CancelButton`, `open_nosleep_page()` |
| `ModelDownloadProgressWindow` | 6680–6721 | Same as above minus nosleep |
| `model_info_frame` | 6980–6984 | `scale_factor` |
| `donation_popup_frame` | 6987–6990 | `scale_factor` |
| `SpeciesNetOutputWindow` | 4632–4957 | Complex — many globals. **SKIP for now.** |

**SKIP `SpeciesNetOutputWindow`** — it's 325 lines with deep entanglement (references
`deploy_speciesnet`, subprocess management, `progress_window`, multiple tkinter vars). Extract
it in Phase 4 when global state is being killed.

**Target files:**

1. `addaxai/ui/dialogs/text_button.py` — `TextButtonWindow`
2. `addaxai/ui/dialogs/patience.py` — `PatienceDialog`
3. `addaxai/ui/dialogs/custom_window.py` — `CustomWindow`
4. `addaxai/ui/dialogs/download_progress.py` — `EnvDownloadProgressWindow`, `ModelDownloadProgressWindow`
5. `addaxai/ui/dialogs/info_frames.py` — `model_info_frame`, `donation_popup_frame`

**Detailed instructions:**

**`TextButtonWindow`** (lines 7121–7155): References `root` and `bring_window_to_top_but_not_for_ever`.
```python
# AFTER in addaxai/ui/dialogs/text_button.py:
import customtkinter
import tkinter as tk

class TextButtonWindow:
    def __init__(self, title, text, buttons, master=None, bring_to_top_func=None):
        self.root = customtkinter.CTkToplevel(master)
        self.root.title(title)
        self.root.geometry("+10+10")
        if bring_to_top_func:
            bring_to_top_func(self.root)
        # ... rest identical ...
```

**`PatienceDialog`** (lines 7158–7187): References `root` and `t()`.
```python
# AFTER in addaxai/ui/dialogs/patience.py:
import customtkinter
import tkinter as tk
from tkinter import ttk
import math
from addaxai.i18n import t

class PatienceDialog:
    def __init__(self, total, text, master=None):
        self.root = customtkinter.CTkToplevel(master)
        self.root.title(t('be_patient'))
        # ... rest identical ...
```

**`CustomWindow`** (lines 7190–7208): References `root`.
```python
# AFTER: add master parameter
class CustomWindow:
    def __init__(self, title="", text="", master=None):
        self.title = title
        self.text = text
        self._master = master
        self.root = None

    def open(self):
        self.root = customtkinter.CTkToplevel(self._master)
        # ... rest identical ...
```

**`EnvDownloadProgressWindow`** (lines 6605–6677) and **`ModelDownloadProgressWindow`** (lines 6680–6721):
These two are similar. Both reference `root`, `scale_factor`, `PADX`, `PADY`, `i18n_lang_idx()`,
`green_primary`, and `CancelButton`.

```python
# AFTER in addaxai/ui/dialogs/download_progress.py:
import customtkinter
from addaxai.i18n import lang_idx as i18n_lang_idx
from addaxai.ui.widgets.buttons import CancelButton

class EnvDownloadProgressWindow:
    def __init__(self, env_title, total_size_str, master=None, scale_factor=1.0,
                 padx=5, pady=2, green_primary="#00A86B", open_nosleep_func=None):
        self.dm_root = customtkinter.CTkToplevel(master)
        # Replace PADX/PADY with padx/pady params
        # Replace scale_factor global with param
        # Replace green_primary global with param
        # Replace open_nosleep_page with open_nosleep_func param
        # ... rest of __init__ identical but using params ...
```

Apply the same pattern for `ModelDownloadProgressWindow`.

**`model_info_frame`** and **`donation_popup_frame`** (lines 6980–6990):
```python
# AFTER in addaxai/ui/dialogs/info_frames.py:
import customtkinter

class ModelInfoFrame(customtkinter.CTkFrame):  # Renamed to PascalCase
    def __init__(self, master, scale_factor=1.0, **kwargs):
        super().__init__(master, **kwargs)
        self.columnconfigure(0, weight=1, minsize=120 * scale_factor)
        self.columnconfigure(1, weight=1, minsize=500 * scale_factor)

class DonationPopupFrame(customtkinter.CTkFrame):  # Renamed to PascalCase
    def __init__(self, master, scale_factor=1.0, **kwargs):
        super().__init__(master, **kwargs)
        self.columnconfigure(0, weight=1, minsize=500 * scale_factor)
```

**Wiring back in `AddaxAI_GUI.py`:**
```python
# Add imports at top:
from addaxai.ui.dialogs.text_button import TextButtonWindow
from addaxai.ui.dialogs.patience import PatienceDialog
from addaxai.ui.dialogs.custom_window import CustomWindow
from addaxai.ui.dialogs.download_progress import EnvDownloadProgressWindow, ModelDownloadProgressWindow
from addaxai.ui.dialogs.info_frames import ModelInfoFrame as model_info_frame, DonationPopupFrame as donation_popup_frame

# Delete original class definitions
# Update all constructor calls to pass required params:
#   TextButtonWindow(title, text, buttons) → TextButtonWindow(title, text, buttons, master=root, bring_to_top_func=bring_window_to_top_but_not_for_ever)
#   PatienceDialog(total, text) → PatienceDialog(total, text, master=root)
#   CustomWindow(title, text) → CustomWindow(title, text, master=root)
#   EnvDownloadProgressWindow(env_title, size) → EnvDownloadProgressWindow(env_title, size, master=root, scale_factor=scale_factor, padx=PADX, pady=PADY, green_primary=green_primary, open_nosleep_func=open_nosleep_page)
#   ModelDownloadProgressWindow(model_title, size) → ModelDownloadProgressWindow(model_title, size, master=root, scale_factor=scale_factor, padx=PADX, pady=PADY, green_primary=green_primary)
#   model_info_frame(master) → model_info_frame(master, scale_factor=scale_factor)
#   donation_popup_frame(master) → donation_popup_frame(master, scale_factor=scale_factor)
```

**Tests** (`tests/test_ui_dialogs.py`):
- Import all classes without error
- Verify each `__init__` signature accepts `master` keyword
- Verify `EnvDownloadProgressWindow.__init__` accepts `scale_factor`, `padx`, `pady`, `green_primary` keywords
- Verify `ModelInfoFrame` and `DonationPopupFrame` accept `scale_factor` keyword

**Commit**: `refactor: extract dialog classes to addaxai/ui/dialogs/ (Phase 3.2)`

---

#### Step 3.3: Extract ProgressWindow → `addaxai/ui/dialogs/progress.py` (1 commit)

This is separated from 3.2 because `ProgressWindow` (lines 7319–8037) is ~718 lines — the single
largest class in the file. It creates up to 7 process panels (img_det, img_cls, vid_det, vid_cls,
img_pst, vid_pst, plt) each with progress bars, labels, and cancel buttons.

**Globals referenced:**
- `root` (implicitly — `CTkToplevel()` with no parent, add `master` param)
- `scale_factor`, `PADX`, `PADY`, `green_primary` (layout/colors)
- `i18n_lang_idx()`, `t()` (translations)
- `CancelButton` class (now imported from `addaxai.ui.widgets.buttons`)

**Target file**: `addaxai/ui/dialogs/progress.py`

**Extraction approach:**
```python
# addaxai/ui/dialogs/progress.py
import customtkinter
from addaxai.i18n import t, lang_idx as i18n_lang_idx
from addaxai.ui.widgets.buttons import CancelButton

class ProgressWindow:
    def __init__(self, processes, master=None, scale_factor=1.0, padx=5, pady=2,
                 green_primary="#00A86B"):
        self.progress_top_level_window = customtkinter.CTkToplevel(master)
        # Replace all bare PADX/PADY with self.padx/self.pady (already stored as self.padx_progress_window etc.)
        # Replace scale_factor global with param
        # Replace green_primary global with param
        # ... rest of __init__ is identical but parameterized ...

    def update_values(self, ...):
        # This method is ~350 lines. Move as-is.
        # It only references i18n_lang_idx() and self.* attributes — no additional globals.
        pass

    def update_progress(self, ...):
        # Move as-is — references green_primary for color changes
        pass

    def close(self):
        self.progress_top_level_window.destroy()
```

**IMPORTANT**: The `ProgressWindow.__init__` has local translation arrays like
`in_queue_txt = ['In queue', 'En cola', ...]` that use `[i18n_lang_idx()]`. These are NOT yet
migrated to `t()` (Phase 2 left complex local arrays alone). Move them as-is for now — they work
via `i18n_lang_idx()` import.

Similarly, `update_values()` has ~12 local translation arrays. Move them as-is.

**Wiring:**
```python
# In AddaxAI_GUI.py, replace the class with:
from addaxai.ui.dialogs.progress import ProgressWindow

# Update the 2 places ProgressWindow is instantiated:
#   ProgressWindow(processes) → ProgressWindow(processes, master=root, scale_factor=scale_factor, padx=PADX, pady=PADY, green_primary=green_primary)
# Search for "ProgressWindow(" to find all call sites.
```

**Tests** (`tests/test_ui_progress.py`):
- Import `ProgressWindow` without error
- Verify `__init__` signature accepts `master`, `scale_factor`, `padx`, `pady`, `green_primary`
- Verify class has `update_values`, `update_progress`, `close` methods

**Commit**: `refactor: extract ProgressWindow to addaxai/ui/dialogs/progress.py (Phase 3.3)`

---

#### Step 3.4: Extract Help Tab → `addaxai/ui/advanced/help_tab.py` (1 commit)

**What to extract:** The `write_help_tab()` function (lines 9601–10088) which populates the help
tab with formatted text and hyperlinks. Also the `HyperlinkManager` class (lines 8302–8330).

**Globals referenced by `write_help_tab()`:**
- `help_text` (the `tk.Text` widget — created at ~line 9003, passed via global)
- `help_tab` (the tab frame)
- `HyperlinkManager` class
- `i18n_lang_idx()` — used extensively for help text language selection
- `text_font` — font name string
- `callback()` (line 6530) — opens URLs in browser

**Globals referenced by `HyperlinkManager`:** None — it's self-contained.

**Target file**: `addaxai/ui/advanced/help_tab.py`

```python
# addaxai/ui/advanced/help_tab.py
import tkinter as tk
import webbrowser
from addaxai.i18n import lang_idx as i18n_lang_idx

class HyperlinkManager:
    # Move lines 8302-8330 as-is — no globals needed
    pass

def write_help_tab(help_text_widget, text_font="TkDefaultFont"):
    """Populate the help tab text widget with formatted content.

    Args:
        help_text_widget: tk.Text widget to write into
        text_font: Font family name string
    """
    # Move the body of write_help_tab() (lines 9602-10088)
    # Replace bare `help_text` with `help_text_widget` parameter
    # Replace `callback` with inline `webbrowser.open_new`
    # i18n_lang_idx() is imported from addaxai.i18n
    pass
```

**Wiring:**
```python
# In AddaxAI_GUI.py:
from addaxai.ui.advanced.help_tab import write_help_tab, HyperlinkManager

# Delete the HyperlinkManager class (lines 8302-8330)
# Delete write_help_tab function definition (lines 9601-10088)
# The call at line ~9601 becomes:
#   write_help_tab(help_text, text_font=text_font)
# (The call site already exists — just update the function reference)
```

**Tests** (`tests/test_ui_help_tab.py`):
- Import `HyperlinkManager` and `write_help_tab` without error
- Verify `write_help_tab` accepts `help_text_widget` and `text_font` params
- Verify `HyperlinkManager.__init__` accepts a `text` param

**Commit**: `refactor: extract help tab to addaxai/ui/advanced/help_tab.py (Phase 3.4)`

---

#### Step 3.5: Extract About Tab → `addaxai/ui/advanced/about_tab.py` (1 commit)

**What to extract:** The `write_about_tab()` function (lines 10101–10152).

**Globals referenced:**
- `about_text` (tk.Text widget)
- `i18n_lang_idx()`
- `text_font`, `current_AA_version`
- `callback()` — opens URLs

**Target file**: `addaxai/ui/advanced/about_tab.py`

```python
# addaxai/ui/advanced/about_tab.py
import webbrowser
from addaxai.i18n import lang_idx as i18n_lang_idx

def write_about_tab(about_text_widget, version="", text_font="TkDefaultFont"):
    """Populate the about tab text widget with formatted content."""
    # Move body of write_about_tab() (lines 10102-10152)
    # Replace bare `about_text` with `about_text_widget` parameter
    # Replace `current_AA_version` with `version` parameter
    pass
```

**Wiring:**
```python
from addaxai.ui.advanced.about_tab import write_about_tab

# Delete write_about_tab function definition
# Update call: write_about_tab() → write_about_tab(about_text, version=current_AA_version, text_font=text_font)
```

**Tests** (`tests/test_ui_about_tab.py`):
- Import `write_about_tab` without error
- Verify signature accepts `about_text_widget`, `version`, `text_font`

**Commit**: `refactor: extract about tab to addaxai/ui/advanced/about_tab.py (Phase 3.5)`

---

#### Step 3.6: Extract Simple Mode → `addaxai/ui/simple/simple_window.py` (1 commit)

**What to extract:** The simple mode window construction code (lines 10154–10258) and the
3 info-button callback functions that are simple-mode-specific:
- `sim_dir_show_info()` (line 6521)
- `sim_spp_show_info()` (line 6533)
- `sim_mdl_show_info()` (line 6552)

**This is the highest-risk step** because the simple mode widget construction is module-level code
(not inside a function/class), and the resulting widgets are referenced by other functions
(`set_language`, `switch_mode`, `update_frame_states`, `start_deploy`, `sim_mdl_dpd_callback`).

**Strategy:** Wrap in a `build_simple_mode(master, ...)` function that constructs all widgets and
returns them as a namespace/dict. The caller stores the result and updates the global references.

```python
# addaxai/ui/simple/simple_window.py
import os
import customtkinter
import tkinter as tk
from tkinter.font import Font
from addaxai.i18n import t, lang_idx as i18n_lang_idx
from addaxai.ui.widgets.frames import MyMainFrame, MySubFrame, MySubSubFrame
from addaxai.ui.widgets.buttons import InfoButton, GreyTopButton
from addaxai.ui.widgets.species_selection import SpeciesSelectionFrame

def sim_dir_show_info():
    # Move lines 6521-6528
    pass

def sim_spp_show_info():
    # Move lines 6533-6545
    pass

def sim_mdl_show_info():
    # Move lines 6552-6567
    pass

def build_simple_mode(root, version, addaxai_files, scale_factor, padx, pady,
                      yellow_primary, green_primary, icon_size, logo_width, logo_height,
                      sim_window_width, sim_window_height, addax_txt_size,
                      pil_sidebar, pil_logo_incl_text, pil_dir_image, pil_mdl_image,
                      pil_spp_image, pil_run_image,
                      on_toplevel_close, switch_mode, set_language, sponsor_project,
                      reset_values, browse_dir_func, update_frame_states,
                      start_deploy_func, sim_mdl_dpd_callback,
                      var_choose_folder, var_choose_folder_short, dsp_choose_folder,
                      row_choose_folder, dpd_options_cls_model, suffixes_for_sim_none,
                      global_vars):
    """Build the simple mode window and all its widgets.

    Returns a dict of all widget references that other code needs.
    """
    # Move lines 10154-10258, parameterizing all globals
    # Return dict with keys for every widget referenced elsewhere:
    # 'window', 'main_frame', 'btn_switch_mode', 'btn_switch_lang', 'btn_sponsor',
    # 'btn_reset_values', 'dir_lbl', 'dir_btn', 'dir_pth', 'mdl_lbl', 'mdl_dpd',
    # 'spp_lbl', 'spp_scr', 'run_btn', 'abo_lbl'
    pass
```

**IMPORTANT — why this is risky:** The simple mode widgets are referenced in at least these places:
- `set_language()` (~line 8088) — reconfigures `sim_btn_switch_mode`, `sim_dir_lbl`, `sim_mdl_lbl`, etc.
- `switch_mode()` (line 8706) — calls `simple_mode_win.deiconify()` / `.withdraw()`
- `update_frame_states()` (line 8213) — updates `sim_dir_pth`, `sim_spp_lbl`, `sim_spp_scr`
- `start_deploy()` — references `sim_run_btn`
- `sim_mdl_dpd_callback()` — references `sim_spp_scr`, `sim_spp_lbl`
- `update_sim_mdl_dpd()` — references `sim_mdl_dpd`
- `main()` — references `sim_dir_pth`

After `build_simple_mode()` returns the widget dict, assign each to the existing global names:
```python
# In AddaxAI_GUI.py, replace lines 10154-10258 with:
_sim = build_simple_mode(root=root, version=current_AA_version, ...)
simple_mode_win = _sim['window']
sim_btn_switch_mode = _sim['btn_switch_mode']
sim_dir_lbl = _sim['dir_lbl']
# ... etc for every widget referenced elsewhere ...
```

This preserves all existing global references so `set_language()`, `switch_mode()`, etc. continue
working unchanged.

**Tests** (`tests/test_ui_simple.py`):
- Import `build_simple_mode`, `sim_dir_show_info`, `sim_spp_show_info`, `sim_mdl_show_info` without error
- Verify `build_simple_mode` signature has all required params (use `inspect.signature`)
- Verify `sim_dir_show_info` is callable

**Commit**: `refactor: extract simple mode to addaxai/ui/simple/simple_window.py (Phase 3.6)`

---

### Verification After Each Step

1. `.venv/Scripts/python -m pytest tests/ -v` — all tests pass (Python 3.14)
2. `C:\Users\Topam\AddaxAI_files\envs\env-base\python.exe -m pytest tests/test_gui_smoke.py -v` — GUI starts without crash
3. Manual: launch via `dev_launch.py`, verify:
   - Advanced mode renders correctly (all 3 tabs)
   - Simple mode renders correctly
   - Language switching works (cycle EN→ES→FR→EN)
   - Mode switching works (advanced ↔ simple)
   - Progress window appears when running a deploy (if possible to test)

### Risk Assessment

| Step | Risk | Reason |
|------|------|--------|
| 3.1 (widgets) | **Low** | Self-contained classes, minimal globals, clear boundaries |
| 3.2 (dialogs) | **Low** | Self-contained classes, `master` param is standard pattern |
| 3.3 (ProgressWindow) | **Medium** | Large class (718 lines), many translation arrays, but well-encapsulated |
| 3.4 (help tab) | **Low** | Single function, clear inputs/outputs |
| 3.5 (about tab) | **Low** | Single function, very small |
| 3.6 (simple mode) | **High** | Module-level code, widgets referenced by 7+ other functions, complex wiring |

**Mitigation:** Each step is one commit. If the GUI breaks, `git revert` the last commit.
Do steps 3.1–3.5 first (low risk) to build confidence before tackling 3.6 (high risk).

### Expected Outcome

- `AddaxAI_GUI.py` loses ~1,200–1,500 lines of class/function definitions
- 18 classes → 11 remain in `AddaxAI_GUI.py` (the 7 extracted + `SpeciesNetOutputWindow` stays)
- `addaxai/ui/` goes from 5 empty `__init__.py` files to 12+ populated modules
- All UI components are importable and testable independently
- No behavioral changes — the app works identically

### What Phase 3 Does NOT Do

- Does NOT extract the deploy tab widget construction (~1,100 lines, lines 9014–9574). This is
  deeply entangled with dozens of globals (tkinter vars, callback functions, model lists). It will
  be handled in Phase 4 alongside global state elimination.
- Does NOT extract `set_language()`, `switch_mode()`, `update_frame_states()`, or other functions
  that orchestrate across multiple windows. These stay in `AddaxAI_GUI.py` until Phase 4.
- Does NOT rename `model_info_frame`/`donation_popup_frame` to PascalCase in `AddaxAI_GUI.py` —
  uses `import ... as` aliases to preserve existing call sites.
