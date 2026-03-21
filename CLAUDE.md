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

### Phase 3: Restructure UI
- [ ] 3.1: Extract dialog classes (ProgressWindow, etc.)
- [ ] 3.2: Extract reusable widgets
- [ ] 3.3: Extract advanced mode tabs
- [ ] 3.4: Extract simple mode window
- [ ] 3.5: Create AppWindow class wrapping module-level UI code

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
**Tests:** 152 passing, 3 skipped (optional deps: cv2, matplotlib)
**Python (tests):** `C:\Users\Topam\AppData\Local\Python\bin\python.exe` (3.14)
**Python (GUI):** `C:\Users\Topam\AddaxAI_files\envs\env-base\python.exe`
**Installed test deps:** pytest, Pillow, numpy, pandas, requests

**Phase 2 fully complete!** AddaxAI_GUI.py reduced from 10,414 → 10,323 lines.
- `lang_idx` global eliminated — all language state in `addaxai.i18n`
- 110 JSON keys in `en.json` / `es.json` / `fr.json`
- All `_txt` arrays, `dpd_options_*` arrays, and major inline arrays replaced with `t()`
- Remaining `[i18n_lang_idx()]` patterns: complex help-text paragraphs, download dialog,
  deploy-progress local arrays — these use the index but are dynamically constructed

**Next:** Phase 3 (restructure UI) — extract dialog classes, tabs, widgets.

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
