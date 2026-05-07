# AddaxAI-WebUI
A temporary repository to build a new AddaxAI version with backend / frontend / API / webUI features. Completely separate from its original repo https://github.com/PetervanLunteren/addaxai so that we can mess around and dont have to be gentle.

### Logging System

The application includes a comprehensive logging system for debugging and diagnostics:
- **Backend logs**: Python `logging` with rotating file handlers (`~/AddaxAI/logs/backend.log`)
- **Frontend logs**: Batched logging forwarded to backend; window errors and unhandled promise rejections are also captured
- **Electron logs**: Stdio captured to backend log; native renderer crashes go to `~/AddaxAI/crash-dumps/`
- **Log retention**: ~7 days, max 100MB total (33MB per log file, 3 backups)
- **Crash detection**: a sentinel file (`~/AddaxAI/.last-shutdown-clean`) is written on graceful exit; missing on next launch triggers a banner

### When things go wrong

**If AddaxAI runs**: Settings page → **Diagnostics** → click **Export diagnostic report**. This builds a ZIP containing logs, system info, installed models, env state, and recent jobs. Save it to Downloads, then email to [peter@addaxdatascience.com](mailto:peter@addaxdatascience.com).

**If AddaxAI won't open**: zip the logs folder manually and email it.

| OS | Logs folder |
|----|-------------|
| macOS / Linux | `~/AddaxAI/logs/` |
| Windows | `%USERPROFILE%\AddaxAI\logs\` |

On macOS: open Finder → press `Cmd+Shift+G` → paste `~/AddaxAI/logs/` → right-click the folder → **Compress**. On Windows: open File Explorer → paste `%USERPROFILE%\AddaxAI\logs\` → right-click → **Send to → Compressed folder**. Native renderer crashes (segfault, OOM) leave minidumps under `~/AddaxAI/crash-dumps/` — include those too if present.

Nothing is uploaded automatically. Sharing logs is always an explicit user action.

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

## Timelapse Analyser integration (Windows only)

AddaxAI exposes a Timelapse integration that writes a `timelapse_recognition_file.json`
next to a chosen folder for import in [Timelapse Analyser](https://saul.cpsc.ucalgary.ca/timelapse/).

Launch options:

- Inside AddaxAI: hamburger menu > Timelapse integration.
- From a shell or Timelapse Analyser: `AddaxAI.exe --timelapse "C:\path\to\folder"`.

The Windows installer drops a compatibility shim at
`%ProgramFiles%\AddaxAI_files\AddaxAI\open.bat`. The legacy invocation
(`open.bat timelapse <folder>`) is forwarded to the new exe transparently,
so no changes are needed on the Timelapse side. After Saul updates
Timelapse to call `AddaxAI.exe --timelapse "<folder>"` directly the shim
becomes redundant; it stays in place as a long-term fallback.

In Timelapse, after a run completes, open `Recognition > Import recognition
data for this image set` and select the generated `timelapse_recognition_file.json`.

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