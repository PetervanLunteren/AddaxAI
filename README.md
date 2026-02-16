# AddaxAI-WebUI
A temporary repository to build a new AddaxAI version with backend / frontend / API / webUI features. Completely separate from its original repo https://github.com/PetervanLunteren/addaxai so that we can mess around and dont have to be gentle.

# TODO
- [ ] NO PRIORITY https://github.com/agentmorris/MegaDetector/blob/main/megadetector/postprocessing/postprocess_batch_results.py

## verification task

I want to add a method for verification. This is my plan. 

Remove the page /review, and leave /images for what it is. We will work on a new page, called "browse and verify". Here we will have cards per event which will show a few images of the event (with a certain max and then something like "+ X"). Then, if we click the event card, it opens a near full screen modal which shows the reproesentative file (investigate, i think we defined the "best" frame for videos somewhere - lets use the same centralised logic), and the other files below it for context. See /Users/peter/Desktop/home-annotate-9864e9322bae026ca866230bc055b925.png as inspiration. If we click a file in the bar below, it opens in the big view. This is the "normal" modal view. There should be a "Verify mode" toggle, where it opens verification options, like edit box, draw box, edit label of box, etc. It should be a full fledged annotation service. 

Please read VERIFICATION_PLAN_v1.md and VERIFICATION_PLAN_v2.md fro context. These are documents written before by me and a friend. Pay more attention to v2, as it is more recent. 

Take your time with this task. It is a considerable task, so dont rush. I have all the time in the world. Please make it thorrough. 

You're in plan mode. 

Instructions:
* If something is unclear at any point, stop and ask before continuing.
* Prioritize simplicity and clarity over perfection. The code must be clean, easy to read, and understandable for collaborators. Avoid unnecessary complexity.
* I'm not in a rush. Please be precise and do the task thoroughly. 
* Please ask me any question for clarification. I would rather that you ask too many questions than assume certain details. 

Workflow:
* Ask at least 3 clarifying questions before beginning.
* Based on my answers, suggest a few general approaches. These should range from simple solutions to more sophisticated alternatives, with clear trade-offs for each. For every approach, explain:
   - Complexity (difficulty, dependencies, maintainability)
   - Readability (clarity for collaborators)
   - Effect (impact on performance, usability, flexibility)
* Give your recommendation regarding the alternatives discribed earlier, with a short reasoning. 
* After I select an approach, draft a detailed plan for implementation.
* Only start working if I agree with the proposed plan.






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