# AddaxAI-WebUI
A temporary repository to build a new AddaxAI version with backend / frontend / API / webUI features. Completely separate from its original repo https://github.com/PetervanLunteren/addaxai so that we can mess around and dont have to be gentle.













  1. Vector search stack: Are we allowed to add a native vector index dependency such as faiss-cpu or hnswlib, or should we stay pure-SQLite/NumPy? This choice affects the entire backend/
     indexing design.
You're allowed to add dependencies. 

  2. Scope of similarity search: Should nearest-neighbour queries run only within the currently filtered subset (e.g., same project + species filter) or can they span the whole project
     regardless of filters?
run only within the currently filtered subset. But it should also work if no filters are set. It will always be inside a project, so it should never show results of another project. 

  3. Verification actions: Do you expect these embedding-driven views to write back to detections immediately (label change, verification flags) the same way as existing verification
     workflows, or just queue suggested edits elsewhere?
write back immidiately the same way as existing verification
     workflows

  4. Dataset scale targets: Roughly how many detections per project should we optimize for (tens of thousands vs millions) so we can justify background indexing jobs vs on-the-fly
     computation?
tens of thousands. on the fly computation with spinner is fine. If the dataset is large users will understand it takes some time. Keep it simple. 

  5. Frontend expectations: Should the clustered/similarity views live under the existing “Verify” page or become a dedicated new route/tab?
live under the existing “Verify” page. I'm envirioning a tab for event based verification, and similarity based verification. 




You are working inside the AddaxAI codebase.

Goal: design and plan a new feature that lets users visually review and verify animal detections using embeddings, by grouping and sorting detections so visually similar items sit next to each other. This should support:

“show me all detections classified as wolves, arranged into similarity clusters so I can quickly spot mislabels like foxes”

“similarity search from a chosen crop to find visually similar detections, including cases that are not part of the model’s class set”

Context: every detection in the database now has an embedding stored after analysis.

Your tasks:

Read deep-research-report.md and extract the landscape of relevant tools and patterns that offer similar experiences (visual similarity review, clustering, nearest-neighbour search, dataset exploration, verification workflows). Summarise only what matters for AddaxAI’s feature decisions.

Inspect the repository to understand:

project structure

frontend patterns and components

backend architecture (framework, routing, services)

database schema and migrations

where detections are queried, filtered, paginated, and rendered

how embeddings are stored and accessed

Propose the best UX and UI for AddaxAI to support:

“clustered view” within a filtered set (example: only predicted wolves)

“similarity search” from one selected detection crop

fast verification actions (confirm label, change label, bulk operations)

performance for large result sets (pagination, virtualisation, background jobs where needed)

transparency controls (distance metric display, threshold sliders, cluster size control, label distribution hints)

Produce a detailed but concise implementation plan that includes:

database considerations (indexes, vector search strategy, storage type, migrations if needed)

backend design (services, background jobs if needed, caching, failure modes)

required api endpoints (routes, request and response shapes, pagination, filters)

frontend implementation (views, components, state management, loading states)

clustering approach (what algorithm, where it runs, how to tune it)

similarity search approach (top-k, threshold, hybrid filtering by species, date range, camera, etc.)

security and permissions (who can access what)

observability (logging and metrics that matter)

Write the output to a new markdown file:

path: EMB_UX_UI_PLAN_CODEX.md

Output format for EMB_UX_UI_PLAN_CODEX.md:

Title and one-paragraph overview

“what already exists in the repo” (short, factual)

“tool landscape takeaways” (short, decision-relevant)

UX and UI proposal

primary user flows (step by step)

screens and layouts (describe placement of buttons, overlays, controls)

interaction details (selection, hover, zoom, compare, bulk verify, keyboard shortcuts if appropriate)

Backend and api design

endpoints list with method, route, purpose

json request and response examples for each endpoint

pagination strategy

filtering options and how they compose with similarity

Data and algorithms

embedding format assumptions

distance metric

nearest neighbour approach (db native vs external index)

clustering approach and where it runs

performance notes and constraints

Stepwise todo list

ordered checklist from schema changes through ui polish

each item small, actionable, and testable

Testing plan

unit tests, integration tests, ui tests

test datasets and edge cases

Open questions and risks

Important constraints:

Prioritise simplicity and clarity over perfection.

Keep code and architecture clean and easy for collaborators.

If any critical detail is unclear, stop and ask before proceeding further.

Do not assume hidden infrastructure. Only propose what is reasonable given the repo.

Be thorough, but avoid fluff.

Before you start writing the plan, do this:

List the key unknowns you need to confirm.

If any unknown blocks a correct plan, stop and ask me those questions.




















## TODO crop verification
- [ ] add UI interface to cluster similarity. 

## TODO priority 1
- [ ] that websocket stuff is very subjective huh? some seconds here, some seconds there.... it feels very hacky tacky.                                                          
                                                                                                                     
⏺ Yeah, it is. The whole buffer-and-replay approach is a workaround for a race condition: the frontend opens the WebSocket after the backend already started sending         
  progress. The "right" fix would be to guarantee ordering — don't start processing until the WebSocket is connected, or have the frontend poll for current state instead of
  relying on replayed messages. Then you wouldn't need a buffer at all.                                                                                                      
                                                                                                                                                                             
- [ ] run on ukraine dep 2 and check the rep image for the lynx. The rep is animal 91%, while there are others with 93%. Maybe we should revisit the formula for rep definition. First combined animal conf, then bbox surface, then sharpness. 



- [ ] Apparently "the classification worker avoids reloading the model for each deployment but is more complex to implement and manage". Should we make it simple and just batch process it every time again for each deployment and task (img / vid)? It adds a bit of model loading, but I would like to keep it as siomple as possible. Investigate what is currently happening and report the options to me. What is possible to make it more simple and what would be the benefit? Is it a major refactor? 

- [ ] Investiagte whether we can make the event smoothing more aggresive. And whether if wouold be translatable to a slider of some kind, or a dropdown with a few categories like mild, normal, aggresive, very aggresive, or simething like that. 
- [ ] build a proper test infrascturture where we can keep adding tests. Add some basic ones to fill the test suite. 
- [ ] is there a way that you can read the console.log yourself without me having to copy paste it every time?
- [ ] do not use instances of "blank", "false detection", "vide", "no cv result", (case insensitive) into the smoothing function and do not load into the DB. They should remain in the JSON as raw data, but should otherwise be ignored. 

## TODO priority 2
- [ ] make the pbars for processing more compact. Take the information that is now below it and put that in the pbar description with icons. SO it would be something like "Running... (file-icon) 18 of 46 - (elapsed time icon) 00:06 - etc. Then make the modal wider so it feels less cramped. 

## TODO priority 3
- [ ] merge all alembic/versions/ into one. We do not have any users yet, so we can make it just the start DB. 
- [ ] make the backgrounds of all modals not only overlay dark, bot also vague. 
- [ ] make sure on app istall it installs the default models and their environments (MDv5A and DINOv2-B). 

## TODO after beta version
- [ ] add a feature that allows datetime offset. This should happen at the "new deployment" options. Perhaps something that says "your data spans X days/weeks, etc. " Click here to see the burned in pixel dates (show a few images / frames) and show the extracted datetime next to it. Then users can add an offset to all data in the deployment. Add fast options to switch from AM to PM etc. +12:00 and -12:00. 

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