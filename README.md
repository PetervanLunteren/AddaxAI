# AddaxAI-WebUI
A temporary repository to build a new AddaxAI version with backend / frontend / API / webUI features. Completely separate from its original repo https://github.com/PetervanLunteren/addaxai so that we can mess around and dont have to be gentle.

# WHEREWASI
- Implementing the actual ML stuff.

# TODO
- [ ] Why has the backend env PyTorch? Isnt that just for FastAPI?
- [ ] Make the FPS configurable in the project settings.
- [ ] Why do we have these files? Isnt everything the model developer needs to supply taxonomy.csv, inference.py and the wetghts? Which are all inside the HF repo, so why are there model specific scripts inside the app itself? A model should be a self contained package so that model developers do not have to dig around in the actual app code. 
     backend/app/
     ├── ml/
     │   ├── inference/
     │   │   ├── megadetector.py         # Image detection (existing)
     │   │   ├── video_detector.py       # NEW: Video detection wrapper
     │   │   ├── yolov8_classifier.py    # Classification (existing)
     │   │   └── speciesnet_model.py     # SpeciesNet (existing)

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

# IMPORTANT: Use Python 3.13 or earlier (NOT 3.14)
# Check your Python version first
python3.13 --version || python3.12 --version || python3.11 --version

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
nvm use 20  # Or: nvm install 20 && nvm use 20

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