# AddaxAI-WebUI
A temporary repository to build a new AddaxAI version with backend / frontend / API / webUI features. Completely separate from its original repo https://github.com/PetervanLunteren/addaxai so that we can mess around and dont have to be gentle.

# TODOwiat
- [ ] https://github.com/agentmorris/MegaDetector/blob/main/megadetector/postprocessing/classification_postprocessing.py

Please investigate this method: https://github.com/agentmorris/MegaDetector/blob/main/megadetector/postprocessing/classification_postprocessing.py (its already included in the repo in the megadetector package). Investigate how that would work if I want it implemented. Can we do that on the fly? As in, make this a toggle in the settings where users can switch to see how it affects the historic results, or can it only do future deployments. Investigate thoroughly. I'm not in a rush.  

OK, a few things to discuss. 
- the 7-token taxonomy strings (GUID;class;order;family;genus;species;common_name) can be computed from the taxonomy.csv files that are shipped with each model (except speciesnet, which uses a different technique - how does that currenty work?). I guess we just need to add a column to each observation with this taxonomy information, right? 
- Is storing this at tow locations redundant? Detection.classification_all_probs (JSON blob) & Classification categories in the JSON files. It feels redundant. Perhaps just remove it in the database and keep the JSONs for the raw predictions which we can use for things like this, where it just reprocesses all results again and writes to DB again. Makes sense?
- I prefer approcah A. Keep it simple.  

SAVED INVESTIGATION

  Investigation: MegaDetector Classification Postprocessing

  What the module does

  classification_postprocessing.py (~2,200 lines) is a pure post-processing module — no ML inference, just logical transformations of existing labels. It
   operates on MegaDetector-format JSON and applies a 4-step smoothing pipeline:

  1. "Other" smoothing — If the dominant species is real (e.g., "cow") and there are "unknown"/"other"/"blank" labels in the same image, flip them to the
   dominant species
  2. Dominant category smoothing — 700 cows + 1 deer in one image? All become cows (configurable thresholds)
  3. Taxonomic rollup — If an image has 1 "carnivore" and 3 "fox", replace the vague "carnivore" with "fox" (walks down the taxonomy tree to be more
  specific)
  4. Within-family smoothing — 5 gray foxes + 1 red fox (same family)? The red fox becomes gray fox

  It works at two levels:
  - Image-level: smooths across detections within a single image
  - Sequence-level: smooths across all detections in a temporal sequence of images (requires COCO Camera Traps sequence info)

  What data it needs

  The module only needs the top-1 classification per detection (it actually strips all others). It also needs:
  - classification_categories: ID-to-name mapping
  - classification_category_descriptions: 7-token taxonomy strings (GUID;class;order;family;genus;species;common_name) — required for taxonomic rollup
  - Detection confidence and classification confidence thresholds

  What AddaxAI currently stores

  Good news — the raw data is already preserved:

  ┌────────────────────────────┬─────────┬───────────────────────────────────────────────────────────┐
  │            Data            │ Stored? │                           Where                           │
  ├────────────────────────────┼─────────┼───────────────────────────────────────────────────────────┤
  │ Top species + confidence   │ Yes     │ Detection.species, Detection.species_confidence           │
  ├────────────────────────────┼─────────┼───────────────────────────────────────────────────────────┤
  │ All class probabilities    │ Yes     │ Detection.classification_all_probs (JSON blob)            │
  ├────────────────────────────┼─────────┼───────────────────────────────────────────────────────────┤
  │ Raw JSON results           │ Yes     │ .addaxai/results_with_classifications.json per deployment │
  ├────────────────────────────┼─────────┼───────────────────────────────────────────────────────────┤
  │ Temporal data (timestamps) │ Yes     │ File.timestamp, Event records                             │
  ├────────────────────────────┼─────────┼───────────────────────────────────────────────────────────┤
  │ Classification categories  │ Yes     │ In the JSON files                                         │
  └────────────────────────────┴─────────┴───────────────────────────────────────────────────────────┘

  Also relevant: your Project model already has taxonomic_rollup, taxonomic_rollup_threshold, event_smoothing, and excluded_classes settings exposed in
  the Settings UI — but none of them are actually wired up yet. The Settings page even says "Changes apply to all analyses (past and future)" which is
  aspirational at this point.

  Can it be retroactive? YES

  Because:
  1. The .addaxai/ JSON files are saved on disk after every analysis
  2. The module is pure post-processing (no re-inference needed)
  3. classification_all_probs preserves the full probability distribution in the DB
  4. The module accepts in-memory dicts, not just files

  Practical architecture for a "toggle"

  There are two viable approaches:

  Approach A: JSON-based reprocessing (recommended)
  - When user changes smoothing/rollup settings, re-run the postprocessing on the saved .addaxai/results_with_classifications.json files
  - Reload the smoothed results to the database (updating species and species_confidence on Detection records)
  - This is a lightweight batch job — no ML, just logic
  - Works for both historic and future deployments
  - Future deployments get postprocessing applied automatically after classification

  Approach B: Dual-column storage
  - Keep species/species_confidence as the raw ML output (never changed)
  - Add species_smoothed/species_smoothed_confidence columns
  - Frontend reads the smoothed columns when smoothing is enabled
  - More complex schema but preserves the distinction between raw and smoothed

  The catch: taxonomy descriptions

  The taxonomic rollup feature specifically requires classification_category_descriptions in the 7-token format
  (GUID;class;order;family;genus;species;common_name). Whether this is available depends on the classification model:

  - SpeciesNet: Likely includes taxonomy data (it's a Google/taxonomy-aware model)
  - Custom models (EUR-DF, NAM-ADS): May not include the full 7-token descriptions

  Without taxonomy descriptions, you still get "other" smoothing and dominant-category smoothing — just not the taxonomy-aware parts. That's still
  valuable.

  Bottom line

  ┌─────────────────────────────────────────┬───────────────────────────────────────────────────────────────────┐
  │                Question                 │                              Answer                               │
  ├─────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ Can it work retroactively?              │ Yes — all needed data is preserved                                │
  ├─────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ Can it be a settings toggle?            │ Yes — reprocess JSON on settings change                           │
  ├─────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ Does it need re-inference?              │ No — pure post-processing                                         │
  ├─────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ Does it work on future deployments too? │ Yes — add as pipeline step after classification                   │
  ├─────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ How fast would reprocessing be?         │ Very fast — no ML, just label logic on JSON. Seconds, not minutes │
  ├─────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ What's the main gap?                    │ Taxonomy descriptions may not be available for all models         │
  └─────────────────────────────────────────┴───────────────────────────────────────────────────────────────────┘






- [ ] https://github.com/agentmorris/MegaDetector/blob/main/megadetector/postprocessing/postprocess_batch_results.py

## Architecture

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the comprehensive technical architecture, technology stack, and implementation roadmap.

### Logging System

The application includes a comprehensive logging system for debugging and diagnostics:
- **Backend logs**: Python `logging` with rotating file handlers (`~/AddaxAI/logs/backend.log`)
- **Frontend logs**: Batched logging forwarded to backend (`~/AddaxAI/logs/frontend.log`)
- **Electron logs**: Winston logger for main process events (`~/AddaxAI/logs/electron.log`)
- **Log retention**: 7 days, max 100MB total (33MB per log file, 3 backups each)
- **Export**: One-click ZIP export with all logs + system info via Settings page 

### Start app

#### 1. Start backend
    ```cmd
    cd backend
    source venv/bin/activate
    uvicorn app.main:app --reload
    ```
#### 2. Start frontend
    ```cmd
    cd frontend
    nvm use 20
    npm run dev
    ```
#### 3. Watch logs in real-time
    ```cmd
    tail -f ~/AddaxAI/logs/backend.log
    ```


## Fresh installation

### Prerequisites

- **Python 3.11-3.13** (check with `python3 --version`) - **Python 3.14 is NOT supported yet** due to pydantic-core compatibility
- **Node.js 20+** and npm (check with `node --version`)
- **Git**

### 1. Clone repository

```bash
git clone https://github.com/PetervanLunteren/AddaxAI-WebUI.git
cd AddaxAI-WebUI
```

### 2. Clean up any old data (if reinstalling)

```bash
# Remove old user data and database
rm -rf ~/AddaxAI

# Remove old virtual environments
rm -rf backend/venv
rm -rf frontend/node_modules
```

### 3. Set up backend

```bash
cd backend

# Create Python virtual environment with Python 3.13 (or 3.12/3.11)
python3.13 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or: .\venv\Scripts\activate  # On Windows

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Set up database
# Apply all database migrations
PYTHONPATH=. alembic upgrade head

# Deactivate venv (optional)
deactivate
```

### 4. Set up frontend

```bash
cd ../frontend

# Use Node.js 20
nvm install 20 && nvm use 20

# Install dependencies
npm install
```

### 5. Verify installation

After setup, you should have:
- `~/AddaxAI/addaxai.db` - SQLite database with schema initialized
- `backend/venv/` - Python virtual environment
- `frontend/node_modules/` - Node dependencies

## Running the app (development mode)

### Start backend (Terminal 1)

```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload
```

Backend will be available at http://localhost:8000

### Start frontend (Terminal 2)

```bash
cd frontend
nvm use 20
npm run dev
```

Frontend will be available at http://localhost:5173

### Watch logs (Terminal 3 - optional)

```bash
tail -f ~/AddaxAI/logs/backend.log
```

## Architecture

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for comprehensive technical architecture, technology stack, and implementation roadmap.

## Key directories

- `~/AddaxAI/` - User data directory (created automatically)
  - `addaxai.db` - SQLite database
  - `logs/` - Application logs
  - `models/` - ML model weights and environments
  - `envs/` - Isolated Python environments for ML models
- `backend/` - FastAPI Python backend
- `frontend/` - React TypeScript frontend
- `electron/` - Electron desktop shell

## Troubleshooting

### Python 3.14 compatibility error

If you see an error about Python 3.14 not being supported by PyO3/pydantic-core:

```bash
# Remove the venv created with Python 3.14
rm -rf backend/venv

# Check which Python versions you have installed
python3.13 --version || python3.12 --version || python3.11 --version

# Create venv with Python 3.13 (or 3.12/3.11)
cd backend
python3.13 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Database initialization failed

If you get "no such table" errors or "Target database is not up to date" errors:

```bash
cd backend
source venv/bin/activate

# Delete the corrupted database
rm ~/AddaxAI/addaxai.db

# If you have old incremental migrations that don't include an initial schema,
# delete them and regenerate:
rm alembic/versions/*.py  # BE CAREFUL: This deletes all migrations

# Generate fresh initial migration
PYTHONPATH=. alembic revision --autogenerate -m "initial schema"

# Apply it
PYTHONPATH=. alembic upgrade head
```

### Port already in use

```bash
# Kill existing backend process
lsof -ti:8000 | xargs kill -9

# Kill existing frontend process
lsof -ti:5173 | xargs kill -9
```

### Missing Python modules

```bash
cd backend
source venv/bin/activate
pip install -r requirements.txt
```

### Frontend build errors

```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```