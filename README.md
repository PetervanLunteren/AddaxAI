# AddaxAI-WebUI
A temporary repository to build a new AddaxAI version with backend / frontend / API / webUI features. Completely separate from its original repo https://github.com/PetervanLunteren/addaxai so that we can mess around and dont have to be gentle.

## TODO verification task
- [ ] fix the grid view, so chips in the images like Connect, etc.
- [ ] if another project has already run a deployment with videos, it saves files to the /.addaxai/ folder. Those frames are then picked up as being images if we process them again. Hoe to avoid? We could of course say that the folder scanner should read the files inside .addaxai, but the real problem would be MegaDetector, as it will just search for everything recusively. Would it make sense to save the frames with an obscure extention like so that MD and the folder scanner doesnt see them as files, and then just edit the extentsion during runtime if we need to read in the frames later?

## TODO crop verification
- [ ] add a crop verifiction page that verifies instances, just like the mothbox tool. That means a DB migration to add verification status to instances, and add an enbedding step in the AI analysis that runs DINOv2 ViT-S/14 (vits14) on all crops (from all images and from the best frames of videos). Investigate its potential and how it would work for this project. This is a major task. 

## TODO priority 1
- [ ] Investiagte whether we can make the event smoothing more aggresive. And whether if wouold be translatable to a slider of some kind, or a dropdown with a few categories like mild, normal, aggresive, very aggresive, or simething like that. 
- [ ] build a proper test infrascturture where we can keep adding tests. Add some basic ones to fill the test suite. 
- [ ] is there a way that you can read the console.log yourself without me having to copy paste it every time?
- [ ] do not use instances of "blank", "false detection", "vide", "no cv result", (case insensitive) into the smoothing function and do not load into the DB. They should remain in the JSON as raw data, but should otherwise be ignored. 

## TODO priority 2
- [ ] make the pbars for processing more compact. Take the information that is now below it and put that in the pbar description with icons. SO it would be something like "Running... (file-icon) 18 of 46 - (elapsed time icon) 00:06 - etc. Then make the modal wider so it feels less cramped. 

## TODO priority 3
- [ ] merge all alembic/versions/ into one. We do not have any users yet, so we can make it just the start DB. 
- [ ] make the backgrounds of all modals not only overlay dark, bot also vague. 

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