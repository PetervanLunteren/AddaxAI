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