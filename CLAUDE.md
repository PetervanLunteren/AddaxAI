# AddaxAI - Refactoring Plan

## What This App Is

AddaxAI helps ecologists classify camera trap images using local computer vision (MegaDetector + various classification models). It supports multiple languages (English, Spanish, French), runs fully offline, and is packaged via PyInstaller for non-technical users.

## Why We're Refactoring

The entire application (~11,000 lines) lives in a single file (`AddaxAI_GUI.py`) with no separation between business logic, UI, model deployment, data processing, and localization. All state is managed via `global` variables. This makes the codebase:

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

### Phase 1: Extract Pure Backend (~3,000 lines, zero UI risk)
Move functions with no tkinter dependency into proper modules:
- [ ] 1.1: `core/config.py` — settings load/save
- [ ] 1.2: `core/paths.py` — path resolution
- [ ] 1.3: `core/platform.py` — OS detection, DPI, interpreter lookup
- [ ] 1.4: `utils/files.py` — file utilities
- [ ] 1.5: `utils/json_ops.py` — JSON manipulation
- [ ] 1.6: `utils/images.py` — image corruption, blur
- [ ] 1.7: `processing/annotations.py` — Pascal VOC, COCO, XML conversion
- [ ] 1.8: `processing/export.py` — CSV/XLSX generation
- [ ] 1.9: `analysis/plots.py` — charts and maps
- [ ] 1.10: `processing/postprocess.py` — file separation, postprocessing
- [ ] 1.11: `models/deploy.py` — model deployment orchestration
- [ ] 1.12: `models/registry.py` — model discovery and download

### Phase 2: Extract Localization
- [ ] 2.1: Create i18n JSON files from inline translation arrays
- [ ] 2.2: Create `t(key)` function
- [ ] 2.3: Replace all inline arrays with `t()` calls

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

## Conventions

- Each extraction step = one commit, immediately pushable
- When extracting a function: move it to new module, add import in `AddaxAI_GUI.py` at the old location so nothing breaks
- Run smoke tests after each extraction before committing
- Do not change behavior during extraction — pure mechanical moves only
- Keep `AddaxAI_GUI.py` working at every commit (it's the only entry point until Phase 3)
