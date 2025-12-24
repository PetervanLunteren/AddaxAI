# Camera Trap Wildlife Analysis Platform - Project Plan

## Overview

### Goals
Build a local-first, cross-platform desktop application for camera trap wildlife analysis that:
- Runs entirely on localhost with full offline capability
- Executes multiple ML models with isolated Python environments
- Provides a modern browser-based UI for annotation and review
- Scales to datasets of 100K-1M+ images
- Works on consumer laptops and GPU workstations
- Ships as a single-click installer (initially macOS, then Windows/Linux)

### Scope
**In scope for MVP (3-6 months):**
- Import camera trap media from SD cards or folders
- Run MegaDetector and custom species classifiers
- Store predictions in hierarchical data model (project > site > deployment > event > file > detection)
- Review and annotate detections with bounding boxes
- Export to CSV and Camtrap DP format
- Keyboard-driven annotation workflows
- Basic statistics and filtering

**Out of scope for MVP:**
- Multi-user collaboration
- Cloud sync or remote deployment
- Advanced GIS features
- Real-time model training
- Mobile apps

### Key Constraints
- No Docker dependency (blocked on many institutional networks)
- No admin permissions required
- Python ML code must remain in Python
- Must work offline after initial setup
- Small team (1-2 developers) with Python/ML expertise

---

## 1. Technology Stack & Decisions

### Decision Table

| Subsystem | Technology | Key Rationale | Alternative Considered |
|-----------|-----------|---------------|----------------------|
| **Desktop Shell** | Electron | Cross-platform, embedded browser, mature packaging ecosystem | Tauri (smaller but less mature) |
| **Backend API** | FastAPI (Python) | Team expertise, async support, automatic OpenAPI docs | Flask (simpler but less modern) |
| **Background Jobs** | AsyncIO workers + Redis/LiteQ | Lightweight async task processing, WebSocket progress updates, offline queue fallback | Celery (heavier but more features), RQ (simpler but less feature-rich) |
| **ML Execution** | micromamba + subprocess | Isolated environments, no Docker, proven to work | conda (larger, slower) |
| **Frontend Framework** | React | Massive ecosystem, excellent libraries for our use case | Vue (smaller ecosystem) |
| **State Management** | TanStack Query + Zustand | Server state (Query) + client state (Zustand) separation | Redux (more boilerplate) |
| **UI Components** | shadcn/ui + Radix | Accessible, customizable, modern | MUI (opinionated styling) |
| **Data Tables** | TanStack Table | Virtual scrolling, 100K+ rows, flexible | AG Grid (commercial license) |
| **Annotation Canvas** | Fabric.js or Konva | Canvas manipulation, bbox drawing | Custom canvas (reinventing wheel) |
| **Database** | SQLite | Local-first, zero-config, sufficient performance | DuckDB (less mature ecosystem) |
| **ORM/Query Builder** | SQLAlchemy | Migration path to Postgres, type safety, flexibility | Raw SQL (maintenance burden) |
| **Video Player** | Video.js | Extensible, overlay support, well-maintained | Plyr (less flexible) |
| **Maps** | MapLibre GL JS + Deck.gl | WebGL-accelerated, complex spatial viz (heatmaps, hexbins, coverage analysis), offline vector tiles | Leaflet (simpler but limited for advanced geospatial analysis) |
| **Plots** | Plotly | Interactive, feature-rich, publication-quality | Recharts (lighter but less features) |
| **Packaging** | electron-builder | Industry standard, code signing, auto-update support | electron-forge (less features) |
| **Logging** | Python `logging` + Winston | Built-in rotating file handlers, structured logs, cross-component aggregation | Loki/Grafana (overkill for local app) |

### Detailed Subsystem Recommendations

#### Desktop Shell & Installer
**Recommended: Electron**
- **Why:** Mature cross-platform framework, allows browser-based UI, large ecosystem, well-documented packaging
- **Tradeoffs:** Larger bundle size (~150MB), more memory usage
- **Alternative:** Tauri (Rust-based, smaller binaries, but less mature and team lacks Rust expertise)

#### Backend API
**Recommended: FastAPI (Python 3.11+)**
- **Why:** Team has Python expertise, async/await for non-blocking operations, automatic API docs, excellent typing support, large community
- **Tradeoffs:** Python performance limits (mitigated by async and offloading ML to background jobs)
- **Alternative:** Flask (simpler but synchronous, less modern), Node.js (team would need to learn, splits stack)

#### Background Job Execution
**Recommended: AsyncIO workers + Redis + LiteQ (offline fallback)**
- **Why:** Lightweight async task processing in Python, WebSocket progress streaming for real-time UI updates, simpler than Celery for small team, LiteQ provides SQLite-backed queue for offline scenarios
- **Tradeoffs:** Less mature ecosystem than Celery, fewer built-in features (but sufficient for our needs)
- **Alternative:** Celery (battle-tested but heavyweight), RQ (simpler but fewer features), Dramatiq (less adoption)

#### ML Pipeline & Model Execution
**Recommended: micromamba + subprocess invocation**
- **Why:** Already proven to work, isolates conflicting dependencies, no Docker requirement, fast environment creation
- **Tradeoffs:** Environment management complexity, initial download size for models
- **Alternative:** Docker (blocked in target environments), venv (cannot handle conflicting dependencies or Python versions)

#### Frontend UI
**Recommended: React 18 + TypeScript + Vite**
- **Why:** Largest ecosystem, excellent libraries for data tables/annotation/video, team can find resources easily, TypeScript for safety, Vite for fast dev experience
- **Tradeoffs:** Boilerplate compared to simpler frameworks
- **Alternative:** Vue 3 (smaller ecosystem, less community libraries), Svelte (less mature ecosystem)

#### Database & Migrations
**Recommended: SQLite + SQLAlchemy 2.0 + Alembic**
- **Why:** Zero-config, local-first, file-based, sufficient for 1M+ rows with proper indexing, SQLAlchemy provides migration path to Postgres
- **Tradeoffs:** No concurrent writes (acceptable for single-user), less powerful than Postgres for complex queries
- **Alternative:** DuckDB (columnar, analytical, but less mature for transactional workloads), Postgres (requires server process, overkill for local-first)

#### Search & Indexing
**Recommended: SQLite FTS5 (Full-Text Search)**
- **Why:** Built into SQLite, no additional dependencies, sufficient for text search on species names and labels
- **Tradeoffs:** Less powerful than dedicated search engines
- **Alternative:** Elasticsearch (massive overkill), Meilisearch (additional process to manage)

#### Maps
**Recommended: MapLibre GL JS + Deck.gl + react-map-gl**
- **Why:** WebGL-accelerated rendering handles both simple markers and complex spatial analysis (heatmaps, hexbins, grid counts, sampling effort visualization, coverage distance analysis). Deck.gl provides high-performance geospatial layers. Supports offline vector tiles (MBTiles). Essential for camera trap density analysis and spatial statistics.
- **Tradeoffs:** Larger bundle (~500KB vs Leaflet's 150KB), but necessary for advanced geospatial requirements
- **Alternative:** Leaflet (lighter, simpler, but inadequate for hexbin/heatmap/coverage analysis)

#### Plotting & Statistics
**Recommended: Plotly (plotly.js via react-plotly.js)**
- **Why:** Highly interactive, publication-quality charts, extensive chart types, zoom/pan/hover built-in, excellent for scientific data visualization
- **Tradeoffs:** Larger bundle size (~3MB), but worth it for camera trap analysis workflows
- **Alternative:** Recharts (lighter, simpler but less interactive), D3.js (low-level, requires more code)

#### Video & Image Overlays
**Recommended: Video.js + custom overlay components**
- **Why:** Extensible plugin system, can render React components as overlays for bounding boxes
- **Tradeoffs:** Requires custom integration for bbox rendering
- **Alternative:** Custom HTML5 video + canvas (reinventing the wheel)

---

## 2. High-Level Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                          Electron Shell                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    React Frontend (Browser)                 │ │
│  │  - TanStack Query (API state)                              │ │
│  │  - Zustand (UI state)                                      │ │
│  │  - TanStack Table (large tables with virtualization)       │ │
│  │  - Fabric.js (bbox annotation)                             │ │
│  └─────────────────────┬──────────────────────────────────────┘ │
│                        │ HTTP/REST (localhost:8000)             │
└────────────────────────┼──────────────────────────────────────────┘
                         │
┌────────────────────────▼──────────────────────────────────────────┐
│                    FastAPI Backend (Python)                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  REST API Endpoints                                       │   │
│  │  - Project/Site/Deployment CRUD                          │   │
│  │  - File import & metadata extraction                     │   │
│  │  - Detection queries & filtering                         │   │
│  │  - Annotation save/update                                │   │
│  │  - Job submission & status polling                       │   │
│  │  - Export generation                                     │   │
│  │  - Log forwarding (frontend logs)                       │   │
│  │  - Log export (ZIP all logs + system info)             │   │
│  └────────┬──────────────────────────────┬──────────────────┘   │
│           │                               │                       │
│  ┌────────▼─────────┐          ┌─────────▼──────────────────┐   │
│  │  SQLAlchemy ORM  │          │  AsyncIO Task Queue        │   │
│  │  - Models        │          │  - ML inference tasks      │   │
│  │  - Relationships │          │  - Import tasks            │   │
│  │  - Migrations    │          │  - Export tasks            │   │
│  └────────┬─────────┘          └─────────┬──────────────────┘   │
│           │                               │                       │
│           │                    ┌──────────▼──────────────────┐   │
│           │                    │  WebSocket Manager          │   │
│           │                    │  - Job progress streams     │   │
│           │                    │  - Real-time updates        │   │
│           │                    └──────────┬──────────────────┘   │
└───────────┼───────────────────────────────┼────────────────────┘
            │                                │
┌───────────▼─────────────┐      ┌──────────▼──────────────────────┐
│   SQLite Database       │      │  Redis + LiteQ (fallback)       │
│   - Projects            │      │  - Task queue                   │
│   - Sites               │      │  - Job state                    │
│   - Deployments         │      │  - WebSocket pub/sub            │
│   - Files               │      └──────────┬──────────────────────┘
│   - Events              │                 │
│   - Detections          │      ┌──────────▼──────────────────────┐
│   - Annotations         │      │  AsyncIO Worker Pool            │
│   - Jobs (tracking)     │      │  ┌──────────────────────────┐   │
│   - Audit Log           │      │  │  Job Executor            │   │
└─────────────────────────┘      │  │  - Picks jobs from queue │   │
                                 │  ┌──────────────────────────┐   │
                                 │  │  Environment Manager     │   │
┌─────────────────────────┐      │  │  - Select env by model   │   │
│  File Storage           │      │  │  - micromamba run -n ... │   │
│  ~/AddaxAI/             │◄─────┼──┤  - Pass file paths       │   │
│  - data/                │      │  │  - Collect results       │   │
│    - projects/          │      │  └──────────┬───────────────┘   │
│      - <proj_id>/       │      └─────────────┼───────────────────┘
│        - media/         │                    │
│        - exports/       │      ┌─────────────▼───────────────────┐
│  - models/              │      │  ML Model Execution             │
│    - environments/      │      │  (subprocess in isolated env)   │
│    - weights/           │      │  ┌────────────────────────┐     │
└─────────────────────────┘      │  │ MegaDetector (env1)    │     │
                                 │  │ - PyTorch 2.0          │     │
                                 │  │ - Python 3.10          │     │
                                 │  └────────────────────────┘     │
                                 │  ┌────────────────────────┐     │
                                 │  │ Species Classifier     │     │
                                 │  │ - TensorFlow 2.13      │     │
                                 │  │ - Python 3.9           │     │
                                 │  └────────────────────────┘     │
                                 └─────────────────────────────────┘
```

### Component Interaction Flow

#### 1. Application Startup
1. Electron main process starts
2. Starts Redis server (bundled binary) for task queue and WebSocket pub/sub
3. Starts FastAPI backend (bundled Python) with AsyncIO worker pool
4. Backend initializes WebSocket manager for real-time job updates
5. Opens browser window pointing to `http://localhost:8000` (served by FastAPI static files)

#### 2. Import Workflow
1. User selects folder/SD card in UI
2. Frontend POST `/api/deployments/{id}/import` with file paths
3. Backend creates AsyncIO task for import and inserts job record in DB
4. Returns job ID immediately (non-blocking)
5. Frontend subscribes to WebSocket `/api/jobs/stream` for real-time progress updates
6. AsyncIO worker:
   - Reads file metadata (EXIF, timestamps)
   - Copies/moves files to project media folder
   - Creates File records in database
   - Groups files into Events based on timestamp rules
   - Publishes progress updates via WebSocket (e.g., "Processing file 150/500")
7. Frontend receives real-time updates, shows progress bar
8. On completion, WebSocket sends final status and frontend refreshes file list

#### 3. ML Inference Workflow
1. User selects files/events and clicks "Run MegaDetector"
2. Frontend POST `/api/models/run` with `{model: "megadetector", file_ids: [...]}`
3. Backend creates AsyncIO task and inserts job record
4. Returns job ID for progress tracking
5. Frontend subscribes to WebSocket for job updates
6. AsyncIO worker:
   - Loads model manifest (specifies environment)
   - Checks if environment exists, downloads if needed
   - Invokes `micromamba run -n megadetector python run_detector.py --files <file_list.json> --output <results.json>`
   - Subprocess writes detections to JSON
   - Worker parses JSON and inserts Detection records into database
   - Publishes progress updates via WebSocket (e.g., "Processed 50/200 images")
7. Frontend receives real-time progress, updates progress bar
8. On completion, WebSocket sends final status and frontend refreshes detections

#### 4. Annotation Workflow
1. User navigates to file detail page
2. Frontend fetches GET `/api/files/{id}/detections`
3. Renders image with Fabric.js canvas overlay
4. User draws/edits bounding boxes
5. On save, frontend POST `/api/detections` or PATCH `/api/detections/{id}`
6. Backend updates Detection records with user edits (stores `edited_by=human` flag)

#### 5. Export Workflow
1. User selects filters (date range, species, confidence threshold)
2. Clicks "Export to CSV"
3. Frontend POST `/api/exports/csv` with filter params
4. Backend creates AsyncIO task and inserts job record (export may take time for large datasets)
5. Frontend subscribes to WebSocket for export progress
6. AsyncIO worker queries database, formats CSV, writes to disk, publishes progress updates
7. On completion, WebSocket sends download URL
8. Frontend downloads file

### State Management

#### Backend State
- **Database (SQLite):** All persistent structured data (projects, files, detections, annotations, jobs, audit_log)
- **File System:** Media files (images/videos), model weights, exports
- **Redis + LiteQ:** Task queue (Redis for online, LiteQ SQLite-backed queue for offline fallback), WebSocket pub/sub for real-time updates

#### Frontend State
- **Server State (TanStack Query):** Cached API responses, automatic refetch, optimistic updates
- **Client State (Zustand):** UI state (selected files, active filters, annotation mode, keyboard shortcuts)

---

## 3. Environment & Model Management

### Strategy: micromamba-based Isolation

#### Directory Structure
```
~/AddaxAI/
├── envs/                      # Conda environments (created by micromamba)
│   ├── env-addaxai-base/      # Base environment for utilities
│   ├── env-pytorch/           # PyTorch environment for detection models
│   ├── env-tensorflow-v1/     # TensorFlow 1.x for legacy classifiers
│   └── env-tensorflow-v2/     # TensorFlow 2.x for modern classifiers
├── models/                    # Model weights (self-contained by type)
│   ├── det/                   # Detection models
│   │   ├── MD5A-0-0/          # MegaDetector v5a
│   │   │   ├── manifest.json
│   │   │   └── md_v5a.0.0.pt
│   │   └── MD5B-0-0/          # MegaDetector v5b
│   │       └── manifest.json
│   └── cls/                   # Classification models
│       ├── EUR-DF-v1-3/       # European species classifier
│       │   ├── manifest.json
│       │   └── model.h5
│       └── AFR-BASIC-v1/      # African species classifier
│           ├── manifest.json
│           └── model.pth
├── bin/
│   └── micromamba             # Micromamba binary
├── logs/
│   ├── backend.log
│   ├── frontend.log
│   └── electron.log
└── addaxai.db                 # SQLite database
```

#### Model Manifest Format

Each model is defined by a JSON manifest stored in its model directory:

```json
// ~/AddaxAI/models/det/MD5A-0-0/manifest.json
{
  "model_id": "MD5A-0-0",
  "friendly_name": "MegaDetector 5a",
  "emoji": "🐅",
  "type": "detection",
  "env": "addaxai-base",
  "model_fname": "md_v5a.0.0.pt",
  "hf_repo": "Addax-Data-Science/MD5A-0-0",
  "description": "MegaDetector is an object detection model that identifies animals, people, and vehicles in camera trap images.",
  "developer": "Dan Morris",
  "info_url": "https://github.com/agentmorris/MegaDetector",
  "min_app_version": "0.1.0",
  "confidence_threshold": 0.1
}
```

```json
// ~/AddaxAI/models/cls/EUR-DF-v1-3/manifest.json
{
  "model_id": "EUR-DF-v1-3",
  "friendly_name": "European species classifier v1.3",
  "emoji": "🦌",
  "type": "classification",
  "env": "tensorflow-v1",
  "model_fname": "model.h5",
  "hf_repo": "Addax-Data-Science/EUR-DF-v1-3",
  "description": "Species classifier for European wildlife (deer, fox, etc.)",
  "developer": "AddaxAI Team",
  "info_url": "https://addax.ai/models/eur-df",
  "min_app_version": "0.1.0",
  "confidence_threshold": 0.5
}
```

**Manifest Fields:**
- `model_id`: Unique identifier (used as directory name)
- `friendly_name`: Human-readable name shown in UI
- `emoji`: Icon for UI display
- `type`: "detection" or "classification"
- `env`: Environment name (points to ~/AddaxAI/envs/env-{env}/)
- `model_fname`: Filename of model weights (e.g., .pt, .h5, .pth)
- `hf_repo`: HuggingFace repository (format: org/repo)
- `description`: Model description for UI
- `developer`: Model creator/organization
- `info_url`: Documentation/source URL
- `min_app_version`: Minimum app version required
- `confidence_threshold`: Default confidence threshold for predictions

#### Environment Creation & Management

**Environment Manager Component (Python):**

```python
# backend/app/ml/environment_manager.py

class EnvironmentManager:
    """
    Manages micromamba environments using static YAML files.

    Environments are defined in backend/app/ml/envs/{env_name}/{platform}/environment.yml
    and created in ~/AddaxAI/envs/env-{env_name}/
    """

    def __init__(self, envs_dir: Path | None = None, micromamba_path: Path | None = None):
        user_data_dir = Path.home() / "AddaxAI"
        self.envs_dir = envs_dir or (user_data_dir / "envs")
        self.envs_dir.mkdir(parents=True, exist_ok=True)

        bin_dir = user_data_dir / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        self.micromamba_path = micromamba_path or (bin_dir / "micromamba")

        # Auto-download micromamba if missing
        if not self.micromamba_path.exists():
            self._download_micromamba()

    def get_or_create_env(
        self, manifest: ModelManifest, progress_callback: Callable[[str, float], None] | None = None
    ) -> Path:
        """
        Get existing environment or create new one from YAML.

        Args:
            manifest: Model manifest with env name
            progress_callback: Optional callback(message: str, progress: float)

        Returns:
            Path to environment directory (~/AddaxAI/envs/env-{env_name}/)
        """
        env_name = f"env-{manifest.env}"
        env_path = self.envs_dir / env_name

        # Check if environment exists and is valid
        if env_path.exists() and self._validate_env(env_path):
            logger.info(f"Using existing environment: {env_name}")
            return env_path

        # Get environment YAML path from repo
        # backend/app/ml/envs/{env_name}/{platform}/environment.yml
        yaml_path = self.get_env_yaml_path(manifest.env)

        # Create new environment
        self._create_env(env_name, env_path, yaml_path, progress_callback)
        return env_path

    def _create_env(
        self, env_name: str, env_path: Path, yaml_path: Path,
        progress_callback: Callable[[str, float], None] | None = None
    ) -> None:
        """Create micromamba environment from YAML file."""
        cmd = [
            str(self.micromamba_path),
            "create",
            "-f", str(yaml_path),
            "-p", str(env_path),
            "-y",
            "--no-rc",
        ]

        # Stream output line by line to show progress
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        for line in process.stdout:
            if line.strip() and progress_callback:
                progress_callback(line[:80], 0.5)

        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"micromamba create failed")

    def get_python(self, env_name: str) -> Path:
        """Get path to Python executable in environment."""
        env_path = self.envs_dir / env_name
        return self._get_python_path(env_path)
```

**Model Storage Component:**

```python
# backend/app/ml/model_storage.py

class ModelStorage:
    """
    Manages model weight downloads from HuggingFace.

    All models download to ~/AddaxAI/models/{det|cls}/{model_id}/
    """

    def __init__(self, models_dir: Path | None = None):
        user_data_dir = Path.home() / "AddaxAI"
        self.models_dir = models_dir or (user_data_dir / "models")
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def check_weights_ready(self, manifest: ModelManifest) -> bool:
        """Check if model weights are downloaded."""
        model_type = "det" if manifest.type == "detection" else "cls"
        model_path = self.models_dir / model_type / manifest.model_id
        model_file = model_path / manifest.model_fname
        return model_file.exists()

    def download_weights(
        self, manifest: ModelManifest,
        progress_callback: Callable[[str, float], None] | None = None
    ) -> Path:
        """Download model weights from HuggingFace if not cached."""
        model_type = "det" if manifest.type == "detection" else "cls"
        model_path = self.models_dir / model_type / manifest.model_id

        if self.check_weights_ready(manifest):
            return model_path

        # Download from HuggingFace
        hf_repo = manifest.hf_repo or f"Addax-Data-Science/{manifest.model_id}"
        downloader = HuggingFaceRepoDownloader(max_workers=4)
        downloader.download_repo(
            repo_id=hf_repo,
            local_dir=model_path,
            progress_callback=progress_callback,
            revision="main"
        )

        return model_path

    def get_model_file(self, manifest: ModelManifest) -> Path:
        """Get path to model weight file (.pt, .h5, .pth)."""
        model_path = self.get_model_path(manifest)
        return model_path / manifest.model_fname
```

#### Environment Definition with Static YAML Files

Environments are defined in the repository at `backend/app/ml/envs/{env_name}/{platform}/environment.yml`:

```yaml
# backend/app/ml/envs/addaxai-base/darwin/environment.yml
name: addaxai-base
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pytorch::pytorch=2.3.1
  - pytorch::torchvision=0.18.1
  - pillow=10.3.0
  - numpy=1.26.4
  - pip
  - pip:
    - ultralytics==8.3.38
```

**Platform-specific YAMLs:**
- `darwin/` - macOS (both Intel and Apple Silicon)
- `linux/` - Linux x86_64
- `windows/` - Windows x86_64

**Benefits:**
- Version-controlled, reproducible environments
- Platform-specific package selection
- Explicit dependency management
- No runtime dependency resolution issues

#### GPU vs CPU Handling

**Current implementation:** CPU-only packages in all environments.

**Future:** Add GPU-enabled environments when needed:
- Detect GPU availability at app startup
- Use separate environment YAMLs (e.g., `pytorch-gpu/` vs `pytorch-cpu/`)
- Model runners select appropriate environment based on hardware

#### Offline Reuse

Once environments and weights are downloaded:
- Environments cached in `~/AddaxAI/envs/`
- Weights cached in `~/AddaxAI/models/det/` and `~/AddaxAI/models/cls/`
- Subsequent runs use cached versions (no network required)
- For offline deployment, transfer the entire `~/AddaxAI/` directory to the offline machine

#### On-Demand Installation

- App ships with micromamba binary (auto-downloaded on first run)
- First time user runs a model:
  1. **Download weights:** "Downloading MegaDetector from HuggingFace..." (progress bar with speed)
  2. **Build environment:** "Installing packages..." (streaming micromamba output via WebSocket)
  3. Cached for all future use
- User only downloads what they need (if they never use model X, they never download it)

---

## 4. Database Design

### Schema Overview

#### Tables & Relationships

```sql
-- Projects
CREATE TABLE projects (
    id TEXT PRIMARY KEY,  -- UUID
    name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sites (camera locations)
CREATE TABLE sites (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    latitude REAL,
    longitude REAL,
    elevation_m REAL,
    habitat_type TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(project_id, name)
);

-- Deployments (camera deployment periods)
CREATE TABLE deployments (
    id TEXT PRIMARY KEY,
    site_id TEXT NOT NULL REFERENCES sites(id) ON DELETE CASCADE,
    start_date DATE NOT NULL,
    end_date DATE,
    camera_model TEXT,
    camera_serial TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Files (images and videos)
CREATE TABLE files (
    id TEXT PRIMARY KEY,
    deployment_id TEXT NOT NULL REFERENCES deployments(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL UNIQUE,  -- Relative to project media dir
    file_type TEXT NOT NULL,  -- 'image' or 'video'
    file_format TEXT,  -- 'jpg', 'png', 'mp4', etc.
    size_bytes INTEGER,
    width_px INTEGER,
    height_px INTEGER,
    timestamp TIMESTAMP NOT NULL,  -- From EXIF or filename
    exif_data JSON,  -- Full EXIF as JSON blob
    duration_seconds REAL,  -- For videos
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_files_deployment (deployment_id),
    INDEX idx_files_timestamp (timestamp)
);

-- Events (time-clustered file groups)
CREATE TABLE events (
    id TEXT PRIMARY KEY,
    deployment_id TEXT NOT NULL REFERENCES deployments(id) ON DELETE CASCADE,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    file_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_events_deployment (deployment_id),
    INDEX idx_events_time (start_time, end_time)
);

-- Event-File junction (many-to-many)
CREATE TABLE event_files (
    event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    file_id TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    sequence_number INTEGER,  -- Order within event
    PRIMARY KEY (event_id, file_id),
    INDEX idx_event_files_event (event_id),
    INDEX idx_event_files_file (file_id)
);

-- Detections (ML predictions or human annotations)
CREATE TABLE detections (
    id TEXT PRIMARY KEY,
    file_id TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    bbox_x REAL NOT NULL,  -- Normalized 0-1
    bbox_y REAL NOT NULL,
    bbox_width REAL NOT NULL,
    bbox_height REAL NOT NULL,
    category TEXT NOT NULL,  -- 'animal', 'person', 'vehicle', or species name
    confidence REAL,  -- 0-1, NULL for human annotations
    source TEXT NOT NULL,  -- 'megadetector', 'species_classifier', 'human'
    source_version TEXT,  -- Model version
    created_by TEXT,  -- 'system' or user ID (future)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reviewed BOOLEAN DEFAULT FALSE,
    notes TEXT,
    INDEX idx_detections_file (file_id),
    INDEX idx_detections_category (category),
    INDEX idx_detections_confidence (confidence)
);

-- Model runs (track which models have been applied to which files)
CREATE TABLE model_runs (
    id TEXT PRIMARY KEY,
    deployment_id TEXT REFERENCES deployments(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    run_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    file_count INTEGER,
    detection_count INTEGER,
    parameters JSON,  -- Model-specific params
    status TEXT,  -- 'pending', 'running', 'completed', 'failed'
    error TEXT
);

-- Jobs (background task tracking)
CREATE TABLE jobs (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,  -- 'import', 'ml_inference', 'export', etc.
    status TEXT NOT NULL,  -- 'pending', 'running', 'completed', 'failed', 'cancelled'
    progress_current INTEGER DEFAULT 0,
    progress_total INTEGER,
    payload JSON,  -- Job-specific parameters
    result JSON,  -- Job output/results
    error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    INDEX idx_jobs_status (status),
    INDEX idx_jobs_created (created_at)
);

-- Audit log (track all data changes)
CREATE TABLE audit_log (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,  -- 'project', 'site', 'detection', 'annotation', etc.
    entity_id TEXT NOT NULL,
    action TEXT NOT NULL,  -- 'create', 'update', 'delete'
    user_id TEXT,  -- Future: for multi-user support
    changes JSON,  -- Before/after values
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_audit_entity (entity_type, entity_id),
    INDEX idx_audit_timestamp (timestamp)
);
```

#### Event Derivation

Events are computed from file timestamps during import:

```python
# Pseudo-code for event grouping
def create_events(deployment_id: str, event_gap_seconds: int = 60):
    files = db.query(File).filter(
        File.deployment_id == deployment_id
    ).order_by(File.timestamp).all()

    current_event = None

    for file in files:
        if current_event is None:
            # Start new event
            current_event = Event(
                deployment_id=deployment_id,
                start_time=file.timestamp,
                end_time=file.timestamp
            )
        elif (file.timestamp - current_event.end_time).total_seconds() > event_gap_seconds:
            # Gap too large, close current event and start new one
            db.add(current_event)
            current_event = Event(
                deployment_id=deployment_id,
                start_time=file.timestamp,
                end_time=file.timestamp
            )
        else:
            # Add to current event
            current_event.end_time = file.timestamp

        db.add(EventFile(event_id=current_event.id, file_id=file.id))

    if current_event:
        db.add(current_event)
```

- Event gap threshold configurable per project (default 60 seconds)
- Events recomputed if gap threshold changes
- Events materialized (stored in DB) for fast queries

#### Annotation & Review History

- Detections table stores both ML predictions and human edits
- `source` field distinguishes origin ('megadetector' vs 'human')
- Human edits can modify ML predictions (update in place) or add new detections
- `reviewed` flag indicates human has reviewed the detection
- Future: Separate `detection_history` table for full audit trail

#### Migration to Cloud/Postgres

This SQLite schema maps directly to Postgres:
- Replace `TEXT PRIMARY KEY` with `UUID PRIMARY KEY DEFAULT gen_random_uuid()`
- Replace `TIMESTAMP DEFAULT CURRENT_TIMESTAMP` with `TIMESTAMP DEFAULT NOW()`
- Replace `JSON` columns with `JSONB` for better Postgres performance
- Add full-text indexes (`CREATE INDEX ... USING GIN`)
- SQLAlchemy ORM abstracts differences, minimal code changes needed

#### Indexing Strategy

Performance-critical indexes:
- `files.timestamp` - Fast time-range queries
- `files.deployment_id` - Fast joins
- `detections.file_id` - Fast detection lookups
- `detections.category` - Filter by species
- `detections.confidence` - Filter by confidence threshold
- `event_files.event_id` and `event_files.file_id` - Fast many-to-many joins

#### Scaling Considerations

SQLite can handle 1M+ rows with proper indexing:
- Use `PRAGMA journal_mode=WAL` for concurrent reads during writes
- Use `PRAGMA synchronous=NORMAL` for faster writes (safe with WAL)
- Use `PRAGMA cache_size=-64000` (64MB cache)
- Vacuum periodically to reclaim space
- For 10M+ rows, migration to Postgres is straightforward

---

## 5. Backend API Design

### API Style: REST

**Rationale:**
- Simple, well-understood, good client library support (TanStack Query)
- Sufficient for our use case (mostly CRUD + job submission)
- GraphQL overkill for small team and limited API surface area

### Core Endpoints

#### Projects
```
GET    /api/projects           List all projects
POST   /api/projects           Create project
GET    /api/projects/{id}      Get project details
PATCH  /api/projects/{id}      Update project
DELETE /api/projects/{id}      Delete project (cascade delete all data)
GET    /api/projects/{id}/stats  Summary stats (file count, detection count, etc.)
```

#### Sites
```
GET    /api/sites?project_id={id}   List sites in project
POST   /api/sites                    Create site
GET    /api/sites/{id}               Get site details
PATCH  /api/sites/{id}               Update site (including lat/lon)
DELETE /api/sites/{id}               Delete site
```

#### Deployments
```
GET    /api/deployments?site_id={id}  List deployments
POST   /api/deployments                Create deployment
GET    /api/deployments/{id}           Get deployment
PATCH  /api/deployments/{id}           Update deployment
DELETE /api/deployments/{id}           Delete deployment
POST   /api/deployments/{id}/import    Start import job
GET    /api/deployments/{id}/files     List files (paginated)
```

#### Files
```
GET    /api/files?deployment_id={id}&limit=100&offset=0  List files (paginated)
GET    /api/files?event_id={id}                          Files in event
GET    /api/files/{id}                                   Get file details
GET    /api/files/{id}/detections                        Get all detections for file
GET    /api/files/{id}/media                             Serve image/video file
DELETE /api/files/{id}                                   Delete file
```

#### Events
```
GET    /api/events?deployment_id={id}  List events
GET    /api/events/{id}                Get event with files
POST   /api/events/recompute            Recompute events with new gap threshold
```

#### Detections
```
GET    /api/detections?file_id={id}              List detections for file
GET    /api/detections?category={name}           Filter by species
GET    /api/detections?min_confidence=0.8        Filter by confidence
POST   /api/detections                           Create detection (human annotation)
PATCH  /api/detections/{id}                      Update detection
DELETE /api/detections/{id}                      Delete detection
POST   /api/detections/{id}/review               Mark as reviewed
```

#### Models
```
GET    /api/models                          List available models (from manifests)
GET    /api/models/{name}                   Get model details
POST   /api/models/{name}/install           Install model environment
POST   /api/models/{name}/run               Run model (returns task_id)
DELETE /api/models/{name}/environment       Delete cached environment
```

#### Jobs (Background Tasks)
```
GET    /api/jobs                   List recent jobs
GET    /api/jobs/{task_id}         Get job status & progress
POST   /api/jobs/{task_id}/cancel  Cancel running job
WS     /api/jobs/stream            WebSocket for real-time job progress updates
```

#### Exports
```
POST   /api/exports/csv          Generate CSV export (returns task_id)
POST   /api/exports/camtrap-dp   Generate Camtrap DP export
GET    /api/exports/{id}/download  Download completed export
```

#### Logging & Diagnostics
```
POST   /api/logs                 Forward frontend logs to backend (batched)
GET    /api/logs/export          Export all logs as ZIP (backend.log, frontend.log, electron.log, system-info.json)
GET    /api/system/info          Get system information (OS, Python version, Node version, hardware, disk space)
```

### Request/Response Examples

#### Import Files
```http
POST /api/deployments/{id}/import
Content-Type: application/json

{
  "source_path": "/Volumes/SD_CARD_01/DCIM",
  "copy_files": true,
  "event_gap_seconds": 60
}

Response 202 Accepted:
{
  "task_id": "abc-123-def",
  "status": "pending",
  "message": "Import job queued"
}
```

#### Poll Job Progress
```http
GET /api/jobs/abc-123-def

Response 200 OK:
{
  "task_id": "abc-123-def",
  "status": "running",
  "progress": {
    "current": 150,
    "total": 500,
    "percent": 30
  },
  "message": "Processing files..."
}
```

#### Run Model
```http
POST /api/models/megadetector/run
Content-Type: application/json

{
  "deployment_id": "deployment-xyz",
  "file_ids": ["file-1", "file-2", ...],  // Optional, null = all files
  "confidence_threshold": 0.1,
  "batch_size": 32
}

Response 202 Accepted:
{
  "task_id": "task-456",
  "model_run_id": "run-789"
}
```

#### Create Annotation
```http
POST /api/detections
Content-Type: application/json

{
  "file_id": "file-1",
  "bbox_x": 0.25,
  "bbox_y": 0.30,
  "bbox_width": 0.15,
  "bbox_height": 0.20,
  "category": "white-tailed deer",
  "source": "human",
  "confidence": null
}

Response 201 Created:
{
  "id": "det-abc",
  "file_id": "file-1",
  "bbox_x": 0.25,
  ...
  "created_at": "2025-12-12T10:30:00Z"
}
```

### Error Handling

Standard HTTP status codes:
- `200 OK` - Success
- `201 Created` - Resource created
- `202 Accepted` - Async job started
- `400 Bad Request` - Invalid input
- `404 Not Found` - Resource doesn't exist
- `409 Conflict` - Constraint violation (e.g., duplicate name)
- `422 Unprocessable Entity` - Validation error
- `500 Internal Server Error` - Server error

Errors return structured JSON:
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid bounding box coordinates",
    "details": {
      "bbox_x": "must be between 0 and 1"
    }
  }
}
```

### Authentication & Authorization

MVP: None (local single-user)

Future:
- Add JWT-based auth when multi-user support added
- SQLAlchemy models ready for `created_by` and `updated_by` fields

---

## 6. Frontend Architecture

### Stack
- **Framework:** React 18 + TypeScript
- **Build Tool:** Vite (fast dev server, optimized production builds)
- **State Management:**
  - **Server state:** TanStack Query (React Query) - caching, refetching, optimistic updates
  - **Client state:** Zustand - UI state, selected items, filters
- **UI Components:** shadcn/ui (Radix primitives + Tailwind CSS)
- **Data Tables:** TanStack Table v8 - virtual scrolling, sorting, filtering
- **Annotation Canvas:** Konva (react-konva) - bbox drawing, editing, keyboard shortcuts
- **Video Player:** Video.js - playback with overlay support
- **Maps:** MapLibre GL JS + Deck.gl + react-map-gl
- **Charts:** Plotly (react-plotly.js)
- **Forms:** React Hook Form + Zod validation

### State Management Strategy

#### Server State (TanStack Query)

All API data managed with TanStack Query for automatic caching and refetching:

```tsx
// Example: Fetching files
const { data: files, isLoading, error } = useQuery({
  queryKey: ['files', deploymentId],
  queryFn: () => api.getFiles(deploymentId),
  refetchInterval: 10000, // Auto-refresh every 10s if job running
})

// Example: Creating detection
const createDetection = useMutation({
  mutationFn: api.createDetection,
  onSuccess: () => {
    // Invalidate and refetch
    queryClient.invalidateQueries(['detections', fileId])
  },
})
```

Benefits:
- Automatic background refetching
- Optimistic updates
- Deduplication of requests
- Pagination support
- Cache invalidation on mutations

#### Client State (Zustand)

UI state that doesn't need server sync:

```tsx
// stores/annotationStore.ts
const useAnnotationStore = create((set) => ({
  selectedFiles: [],
  annotationMode: 'select', // 'select' | 'draw' | 'edit'
  showConfidenceThreshold: 0.5,

  setSelectedFiles: (files) => set({ selectedFiles: files }),
  toggleAnnotationMode: (mode) => set({ annotationMode: mode }),
}))
```

### Handling Large Tables

**TanStack Table with Virtualization:**

```tsx
import { useVirtualizer } from '@tanstack/react-virtual'
import { useTable } from '@tanstack/react-table'

function FileTable({ deploymentId }) {
  // Fetch paginated data
  const [pagination, setPagination] = useState({ pageIndex: 0, pageSize: 100 })
  const { data } = useQuery({
    queryKey: ['files', deploymentId, pagination],
    queryFn: () => api.getFiles(deploymentId, pagination),
    keepPreviousData: true, // Smooth pagination
  })

  const table = useTable({
    data: data?.files ?? [],
    columns,
    pageCount: data?.pageCount ?? 0,
    state: { pagination },
    onPaginationChange: setPagination,
    manualPagination: true, // Server-side pagination
  })

  // Virtual scrolling for visible rows
  const parentRef = useRef()
  const rowVirtualizer = useVirtualizer({
    count: table.getRowModel().rows.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 50, // Row height
    overscan: 10,
  })

  return (
    <div ref={parentRef} style={{ height: '600px', overflow: 'auto' }}>
      {rowVirtualizer.getVirtualItems().map((virtualRow) => {
        const row = table.getRowModel().rows[virtualRow.index]
        return <TableRow key={row.id} row={row} />
      })}
    </div>
  )
}
```

- Server-side pagination (load 100-1000 rows at a time)
- Client-side virtual scrolling (render only visible rows)
- Handles 100K+ images smoothly

### Annotation & Canvas Strategy

**Fabric.js for Bounding Box Annotation:**

```tsx
import { useEffect, useRef } from 'react'
import { fabric } from 'fabric'

function AnnotationCanvas({ imageUrl, detections, onSave }) {
  const canvasRef = useRef()
  const fabricRef = useRef()

  useEffect(() => {
    // Initialize canvas
    const canvas = new fabric.Canvas(canvasRef.current)
    fabricRef.current = canvas

    // Load image
    fabric.Image.fromURL(imageUrl, (img) => {
      canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas))
      canvas.setWidth(img.width)
      canvas.setHeight(img.height)
    })

    // Load existing detections as rectangles
    detections.forEach((det) => {
      const rect = new fabric.Rect({
        left: det.bbox_x * img.width,
        top: det.bbox_y * img.height,
        width: det.bbox_width * img.width,
        height: det.bbox_height * img.height,
        fill: 'transparent',
        stroke: det.source === 'human' ? 'green' : 'yellow',
        strokeWidth: 2,
        data: { detectionId: det.id },
      })
      canvas.add(rect)
    })

    return () => canvas.dispose()
  }, [imageUrl, detections])

  const handleSave = () => {
    const objects = fabricRef.current.getObjects()
    const newDetections = objects.map((obj) => ({
      bbox_x: obj.left / fabricRef.current.width,
      bbox_y: obj.top / fabricRef.current.height,
      bbox_width: obj.width / fabricRef.current.width,
      bbox_height: obj.height / fabricRef.current.height,
    }))
    onSave(newDetections)
  }

  return (
    <div>
      <canvas ref={canvasRef} />
      <button onClick={handleSave}>Save Annotations</button>
    </div>
  )
}
```

**Keyboard Shortcuts:**

Use `react-hotkeys-hook` for global shortcuts:

```tsx
import { useHotkeys } from 'react-hotkeys-hook'

function AnnotationView() {
  useHotkeys('n', () => navigateToNextFile())
  useHotkeys('p', () => navigateToPreviousFile())
  useHotkeys('d', () => deleteSelectedDetection())
  useHotkeys('r', () => markAsReviewed())
  useHotkeys('1-9', (e) => assignCategory(e.key))
}
```

### Testing Approach

#### Unit Tests (Vitest)
- Test utility functions, hooks, stores
- Test component logic in isolation
- Fast, run on every commit

```tsx
// Example
import { renderHook } from '@testing-library/react'
import { useAnnotationStore } from './annotationStore'

test('toggles annotation mode', () => {
  const { result } = renderHook(() => useAnnotationStore())
  result.current.toggleAnnotationMode('draw')
  expect(result.current.annotationMode).toBe('draw')
})
```

#### Component Tests (React Testing Library)
- Test user interactions
- Test component rendering with various props
- Test form validation

#### End-to-End Tests (Playwright)
- Critical user flows:
  - Import files
  - Run model
  - Annotate detection
  - Export data
- Run against local build before releases

---

## 7. Packaging & Installation

### Installer Strategy: Electron + electron-builder

#### Packaging Approach

**Components Bundled in Installer:**
1. **Electron app** - Desktop shell with main.ts launcher logic
2. **Python backend** - FastAPI + dependencies bundled as single executable (PyInstaller)
3. **React frontend** - Static files bundled inside PyInstaller executable

**Downloaded on First Run:**
- **micromamba** - Downloaded to `~/AddaxAI/bin/` on first launch
- **Model weights** - Downloaded from HuggingFace when user selects a model
- **Conda environments** - Created on demand when running models

**Directory Structure After Installation:**

```
/Applications/AddaxAI.app/  (macOS example)
├── Contents/
│   ├── MacOS/
│   │   ├── AddaxAI              # Electron executable
│   │   └── Resources/
│   │       └── backend/
│   │           └── backend      # FastAPI bundled binary (PyInstaller)
│   └── Resources/
│       └── app/                 # React frontend static files (bundled in PyInstaller)
└── Info.plist

~/AddaxAI/  (user data directory - all runtime data)
├── addaxai.db                   # SQLite database
├── bin/
│   └── micromamba               # Downloaded on first run
├── envs/                        # Conda environments (created on demand)
│   ├── env-addaxai-base/
│   ├── env-pytorch/
│   ├── env-tensorflow-v1/
│   └── env-tensorflow-v2/
├── models/                      # Model weights (downloaded from HuggingFace)
│   ├── det/
│   │   ├── MD5A-0-0/
│   │   │   ├── manifest.json
│   │   │   └── md_v5a.0.0.pt
│   │   └── MD5B-0-0/
│   └── cls/
│       └── EUR-DF-v1-3/
│           ├── manifest.json
│           └── model.h5
├── logs/
│   ├── backend.log              # Backend logs (rotates at 33MB, 3 backups)
│   ├── frontend.log             # Frontend logs (rotates at 33MB, 3 backups)
│   └── electron.log             # Electron logs (rotates at 33MB, 3 backups)
└── pkgs/                        # Micromamba package cache
```

#### Startup Process

**Electron Main Process:**

```typescript
// main.ts
import { app, BrowserWindow } from 'electron'
import { spawn } from 'child_process'
import path from 'path'

let backendProcess
let redisProcess
let celeryProcess

async function startBackend() {
  const appPath = app.getAppPath()

  // Start Redis
  const redisPath = path.join(appPath, 'backend', 'redis-server')
  redisProcess = spawn(redisPath, ['--port', '6379', '--dir', getUserDataPath()])

  // Start FastAPI backend
  const backendPath = path.join(appPath, 'backend', 'api')
  backendProcess = spawn(backendPath, ['--port', '8000'])

  // Wait for backend to be ready
  await waitForBackend('http://localhost:8000/health')

  // Start Celery worker
  const celeryPath = path.join(appPath, 'backend', 'celery-worker')
  celeryProcess = spawn(celeryPath)
}

async function createWindow() {
  await startBackend()

  const win = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      contextIsolation: true,
    },
  })

  // Load frontend from backend
  win.loadURL('http://localhost:8000')
}

app.on('ready', createWindow)

app.on('will-quit', () => {
  // Graceful shutdown
  backendProcess?.kill()
  celeryProcess?.kill()
  redisProcess?.kill()
})
```

#### Bundling Backend with PyInstaller

```python
# backend.spec (PyInstaller spec)
a = Analysis(
    ['backend/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('backend/models/manifests', 'models/manifests'),
        ('backend/scripts', 'scripts'),
    ],
    hiddenimports=['celery', 'sqlalchemy', 'fastapi'],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
)
pyz = PYZ(a.pure, a.zipped_data)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='api',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window
)
```

#### electron-builder Configuration

```json
{
  "build": {
    "appId": "com.addaxai.cameratrap",
    "productName": "AddaxAI",
    "mac": {
      "category": "public.app-category.education",
      "target": ["dmg", "zip"],
      "icon": "assets/icon.icns",
      "hardenedRuntime": true,
      "gatekeeperAssess": false,
      "entitlements": "entitlements.mac.plist",
      "entitlementsInherit": "entitlements.mac.plist"
    },
    "win": {
      "target": ["nsis"],
      "icon": "assets/icon.ico"
    },
    "linux": {
      "target": ["AppImage", "deb"],
      "category": "Science"
    },
    "files": [
      "dist-electron/**/*",
      "dist-frontend/**/*",
      "backend/dist/**/*"
    ],
    "extraResources": [
      {
        "from": "binaries/micromamba-${os}",
        "to": "micromamba"
      },
      {
        "from": "binaries/redis-server-${os}",
        "to": "backend/redis-server"
      }
    ]
  }
}
```

#### Application Data Storage

All application data is stored in `~/AddaxAI/`:

- **Database:** `~/AddaxAI/addaxai.db` (SQLite database with project/site/deployment/file/detection data)
- **Conda environments:** `~/AddaxAI/envs/env-{env_name}/`
- **Model weights:** `~/AddaxAI/models/{det|cls}/{model_id}/`
- **Logs:** `~/AddaxAI/logs/` (backend.log, frontend.log, electron.log)
- **Binaries:** `~/AddaxAI/bin/micromamba`
- **Package cache:** `~/AddaxAI/pkgs/` (micromamba package cache)

**Cross-platform note:** On Windows, this becomes `%USERPROFILE%\AddaxAI\`

#### Auto-Update Support (Future)

electron-builder supports auto-update via:
- macOS: Built-in update mechanism
- Windows: Squirrel.Windows
- Linux: AppImage auto-update

For MVP: Manual download and install new versions

#### Code Signing Considerations

**macOS:**
- Requires Apple Developer account ($99/year)
- Sign with `codesign` and notarize with Apple
- Without signing, users get "unidentified developer" warning (can bypass with right-click > Open)

**Windows:**
- Code signing certificate available (already obtained)
- Sign with signtool.exe during electron-builder packaging
- Prevents SmartScreen warnings for users

**Linux:**
- No signing required

**MVP Approach:**
- Sign Windows builds with existing certificate
- macOS: Initially distribute unsigned .dmg with bypass instructions (right-click > Open)
- Add macOS code signing ($99/year Apple Developer) when user base justifies the cost

---

## 8. Project Plan & Milestones

### Phase 1: Foundation (Weeks 1-3)

**Milestone 1.1: Project Setup**
- Initialize monorepo structure (`backend/`, `frontend/`, `electron/`)
- Set up FastAPI backend with SQLAlchemy models
- Set up React frontend with Vite + TypeScript
- Basic Electron shell that launches backend and loads frontend
- SQLite database with Alembic migrations
- CI/CD: GitHub Actions for linting and tests

**Milestone 1.2: Core Data Model**
- Implement Project, Site, Deployment, File models
- Database migrations
- Basic CRUD API endpoints for projects and sites
- Frontend: Basic navigation, create project/site forms
- File storage setup (copy files to project directory)

### Phase 2: Import & Browse (Weeks 4-6)

**Milestone 2.1: File Import**
- Import workflow: select folder, copy files, extract EXIF
- AsyncIO worker + Redis/LiteQ integration for background import jobs
- WebSocket for real-time progress updates
- Progress tracking UI with live updates
- File list view with pagination (TanStack Table)
- Image thumbnail generation and caching

**Milestone 2.2: Event Grouping**
- Event derivation logic (time-based clustering)
- Event list view
- Event detail view (all files in event)
- Basic filtering (date range, file type)

### Phase 3: ML Integration (Weeks 7-10)

**Milestone 3.1: Environment Manager**
- Model manifest format (YAML schema)
- Environment manager (create, verify, cache)
- micromamba integration (create env, run scripts)
- Single model support: MegaDetector
- Download model weights from Hugging Face/GitHub

**Milestone 3.2: Model Execution**
- AsyncIO task for model inference with WebSocket progress streaming
- Run MegaDetector on selected files
- Store detections in database
- Detection list view (show predictions on file detail page)
- Real-time progress bar during model execution

**Milestone 3.3: Multi-Model Support**
- Add species classifier model
- Model selection UI
- Sequential model runs (detection → classification pipeline)
- GPU vs CPU detection and fallback

### Phase 4: Review & Annotation (Weeks 11-13)

**Milestone 4.1: Detection Review UI**
- File detail page with image viewer
- Display existing detections as bounding boxes
- Filter by confidence threshold
- Mark detections as reviewed
- Navigate between files with keyboard shortcuts (n/p)

**Milestone 4.2: Annotation Tools**
- Konva (react-konva) canvas integration
- Draw new bounding boxes
- Edit existing bounding boxes (resize, move)
- Delete detections
- Assign categories (species dropdown)
- Save annotations to database with audit log tracking

**Milestone 4.3: Batch Operations**
- Select multiple files
- Batch apply model
- Batch delete files
- Batch mark as reviewed

### Phase 5: Visualization & Export (Weeks 14-16)

**Milestone 5.1: Statistics & Charts**
- Dashboard with summary stats (total files, detections, species counts)
- Species distribution chart (bar chart)
- Detection timeline (line chart)
- Detection confidence histogram

**Milestone 5.2: Maps**
- Site map view (MapLibre GL JS + Deck.gl)
- Display sites as markers with lat/lon
- Click site to view deployments
- Edit site location by dragging marker
- Add heatmap layer for detection density
- Implement hexbin visualization for camera coverage analysis

**Milestone 5.3: Export**
- CSV export (all detections with file metadata)
- Camtrap DP export (standard format)
- Export job queue (AsyncIO task with WebSocket progress)
- Download completed export files

### Phase 6: Packaging & Hardening (Weeks 17-20)

**Milestone 6.1: Electron Packaging**
- PyInstaller backend bundling
- electron-builder configuration
- Bundle micromamba and Redis
- macOS .dmg installer
- Test installation on clean machine

**Milestone 6.2: Cross-Platform Support**
- Windows installer (.exe)
- Linux AppImage
- Platform-specific binary bundling
- Test on all three platforms

**Milestone 6.3: Polish & QA**
- Error handling and user feedback
- Loading states and spinners
- Empty states (no projects, no files, etc.)
- Keyboard shortcut documentation
- In-app help/onboarding
- End-to-end tests (Playwright)
- Performance optimization (large datasets)

### Phase 7: Beta Release (Week 21+)

**Milestone 7.1: Documentation**
- User guide (installation, basic workflows)
- Developer documentation (architecture, setup)
- Model manifest documentation
- README with screenshots

**Milestone 7.2: Beta Testing**
- Internal testing with real camera trap data
- Invite domain experts for feedback
- Bug fixes and UX improvements

**Milestone 7.3: Public Release**
- GitHub release with installers
- MIT license
- Community engagement (forum, issue tracker)

---

### Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Model environments fail to install on some systems** | Medium | High | Thorough testing on clean VMs, fallback to manual environment setup instructions, verbose error messages, comprehensive logging system for remote debugging |
| **Large datasets (1M+ images) cause performance issues** | Medium | High | Early performance testing with synthetic large datasets, database query optimization, pagination, lazy loading |
| **GPU detection fails or is unreliable** | Medium | Medium | CPU fallback always available, clear messaging about GPU status, test on various hardware |
| **File format compatibility issues** | Low | Medium | Use battle-tested libraries (Pillow, OpenCV), document supported formats, graceful error handling |
| **micromamba download blocked by firewalls** | Medium | Medium | Bundle common environments in installer (increases size), provide offline installation guide |
| **Cross-platform inconsistencies** | Medium | Medium | Continuous testing on all platforms, use cross-platform libraries, avoid platform-specific code |
| **Database corruption or migration failures** | Low | High | Regular backups (automated), migration tests, rollback capability, database integrity checks |
| **Background jobs hang or crash** | Medium | Medium | Timeout mechanisms, restart failed tasks, Celery monitoring, comprehensive logging system captures full stack traces and progress state |
| **Installer blocked by antivirus/Gatekeeper** | High | Medium | Code signing (future), clear bypass instructions, build trust with community |
| **Team capacity (1-2 devs) causes delays** | Medium | High | Ruthless scope prioritization, use proven libraries vs custom code, avoid premature optimization |
| **Users unable to effectively report bugs** | High | Medium | Comprehensive logging system with one-click export (backend, frontend, Electron logs + system info), logs capture errors, user actions, and full stack traces for remote debugging |

---

## 9. Non-Functional Considerations

### Performance Targets

#### Import
- **Target:** 1000 images imported and processed (EXIF extraction, thumbnail generation) in <2 minutes on consumer laptop
- **Bottlenecks:** Disk I/O (copying files), EXIF parsing
- **Mitigations:** Parallel processing (Celery workers), optimize EXIF library usage, batch database inserts

#### Model Inference
- **Target:** MegaDetector processes 100 images in <5 minutes on CPU, <1 minute on GPU
- **Bottlenecks:** Model compute, I/O loading images
- **Mitigations:** Batch processing, GPU acceleration when available, model-specific optimizations

#### UI Responsiveness
- **Target:** Table with 10K rows renders in <500ms, scrolling smooth at 60fps
- **Bottlenecks:** DOM rendering
- **Mitigations:** Virtual scrolling (only render visible rows), server-side pagination, debounced filtering

#### Database Queries
- **Target:** Complex queries (e.g., filter 100K files by species + confidence) return in <200ms
- **Bottlenecks:** Full table scans
- **Mitigations:** Proper indexing, query optimization, EXPLAIN QUERY PLAN analysis

### Concurrency Model

#### Backend (FastAPI + AsyncIO)
- **FastAPI:** Async/await for I/O-bound operations (database queries, file reads, WebSocket connections)
- **AsyncIO worker pool:** Async tasks for long-running jobs (ML inference, image processing, imports)
- **SQLite:** Write-ahead logging (WAL) for concurrent reads during writes
- **Redis + LiteQ:** Handles concurrent task queue operations and WebSocket pub/sub

#### Frontend
- **React:** Single-threaded, but non-blocking via TanStack Query (background fetching)
- **Web Workers:** Offload heavy computations if needed (e.g., client-side image filtering)

### Caching Strategy

#### Backend
- **API responses:** No server-side HTTP caching (local app, TanStack Query handles client cache)
- **Thumbnails:** Generate once, store in project directory, serve with Cache-Control headers
- **Model weights:** Download once, cache indefinitely in `~/AddaxAI/models/{det|cls}/{model_id}/`
- **Environments:** Create once in `~/AddaxAI/envs/`, reuse until deleted

#### Frontend (TanStack Query)
- **Default stale time:** 5 minutes (data considered fresh, no refetch)
- **Cache time:** 30 minutes (cached data kept in memory)
- **Automatic refetch:** On window focus, on network reconnect
- **Optimistic updates:** Mutations update cache immediately before server confirms

#### Database
- **SQLite cache:** 64MB in-memory cache for frequently accessed pages
- **Query results:** No application-level query cache (SQLite handles this)

### Privacy & Security

#### Local Data
- **Data location:** All data stored locally in user's home directory (`~/AddaxAI/`)
- **No cloud sync:** Data never leaves user's machine (MVP)
- **File permissions:** Standard OS permissions, no special requirements

#### Network
- **Localhost only:** Backend binds to `127.0.0.1` (not accessible from network)
- **No authentication:** Single-user local app, no auth needed
- **HTTPS:** Not needed for localhost (browsers allow)

#### Future Considerations
- **Multi-user:** Add JWT authentication when cloud sync added
- **Encryption at rest:** Optional database encryption for sensitive projects
- **Audit logs:** Track who edited what (for collaborative use)

### Telemetry (Optional, Opt-In)

**What to collect (with user consent):**
- Crash reports (stack traces, OS version, app version)
- Performance metrics (import duration, model run duration)
- Usage patterns (which models are popular, typical dataset sizes)

**How it works:**
- **Offline-first:** Telemetry events stored locally in SQLite
- **Periodic upload:** When internet available, batch upload to server (opt-in)
- **Privacy:** No personal data, no file names, no image content
- **Open source:** Telemetry code fully visible, auditable

**Implementation:**
- Use open-source telemetry (e.g., PostHog self-hosted, Sentry for errors)
- Clear opt-in during first launch
- Settings to disable anytime

---

## General Build Order (MVP-First Approach)

**Strategy**: Build a Minimum Viable Product (MVP) first to validate the entire deployment pipeline (Electron packaging, code signing, cross-platform installers) before investing heavily in advanced features. This de-risks the project early.

**MVP Scope**:
- Deployment creation with folder selection (no file copying)
- File scanning with EXIF extraction
- MegaDetector (animal detection) + Simple Classifier (species)
- Basic file browsing with thumbnails
- Image viewer with bounding box visualization
- Simple detection timeline chart
- Electron packaging and distribution

### Phase 1: MVP Development

#### Setup & Infrastructure
1. ✅ Initialize monorepo with backend/, frontend/, electron/ directories
2. ✅ Set up FastAPI backend with basic project structure
3. ✅ Set up React + Vite + TypeScript frontend
4. ✅ Configure Electron shell to launch backend and load frontend (main.ts complete with health checks)
5. ✅ Set up SQLite database with SQLAlchemy models and Alembic migrations
6. ✅ Implement basic CI/CD (linting, type-checking, tests)
7. ✅ Create initial Project, Site, Deployment, File database models
7a. ✅ Implement backend logging with RotatingFileHandler (Python `logging` module)
7b. ✅ Create frontend logger with batching (forwards to POST /api/logs every 5s)
7c. ❌ Add Electron Winston logger for main process events (not implemented)
7d. ❌ Create log export endpoint (GET /api/logs/export - ZIP with all logs + system-info.json) (not implemented)
7e. ❌ Add "Export Logs" button to Settings page with privacy warning (not implemented)

#### Core Functionality (MVP)
8. ✅ Implement CRUD API endpoints for projects and sites
9. ✅ Build frontend navigation and basic UI layout (sidebar with project-scoped tools)
10. ✅ Create project/site creation and edit forms
10a. ✅ Create deployments page UI → Now called AnalysesPage (queue management UI)
11. ✅ Update deployment model for no-copy file storage (folder_path approach)
11a. ✅ Implement deployment CRUD API endpoints (backend/app/api/routers/deployments.py)
11b. ✅ Build file scanner service (recursive scan with EXIF extraction) (backend/app/services/folder_scanner.py + preview-folder endpoint)
11c. ✅ Create deployment creation dialog with folder selection (AddDeploymentDialog.tsx)
11d. ✅ Implement ImagesPage to display scanned files (grid with thumbnails, click to view modal with detections)

#### ML Integration (MVP - MegaDetector + Classifier Only)
20. ✅ Design simple model manifest format (JSON, with support for det/cls model organization)
21. ✅ Build environment manager for micromamba (EnvironmentManager with env creation, validation, and caching)
22. ✅ Integrate micromamba for isolated Python environments (environments stored in ~/AddaxAI/envs/)
23. ✅ Implement model weight download and caching (HuggingFaceRepoDownloader with progress tracking)
23a. ✅ Add download progress bar with speed display (updates every 500ms via WebSocket)
24. ✅ Create AsyncIO task for model inference execution (detection_worker.py with progress callbacks)
25. ✅ **Integrate MegaDetector** (MegaDetectorRunner with subprocess execution in isolated envs)
26. ✅ Implement detection storage in database (Detection model with file_id, category, confidence, bbox)
26a. ✅ Create ML model preparation UI (separate buttons for "Download Weights" and "Build Environment")
26b. ✅ Add model status API endpoint (check weights_ready and env_ready status)
26c. ✅ Fix model file path resolution (pass .pt file path instead of model ID to CLI)
26d. ✅ Organize models by type (models/det/ and models/cls/ with manifest.json in each model dir)
26e. ✅ Simplify environment directory naming (ml_envs → envs)
27. ✅ Add WebSocket for real-time job progress updates (websocket.py router)
28. ✅ Implement job queue management (jobs.py router with list, create, update, delete, run_queue)
29. ❌ **Integrate species classifier** (manifests defined for EUR-DF-v1-3 and AFR-BASIC-v1 but runners not implemented)
30. ❌ **Sequential pipeline (detect → classify)** - Not wired up
31. ❌ **GPU detection and CPU fallback** - Not implemented

#### Basic Data Browsing (MVP)
32. ✅ Build file detail page with image viewer (ImagesPage with grid of image cards)
32a. ✅ Create files API router (list files with filters, get file with detections, serve image files)
32b. ✅ Add image serving endpoint (/api/files/{file_id}/image to bypass file:// protocol restrictions)
33. ✅ Implement SVG overlay for bounding box visualization (replaced Konva with simpler SVG solution)
34. ✅ Display ML detections on images (click image card → modal with full image + bounding boxes)
34a. ✅ Add color-coded bounding boxes by category (green=animal, red=person, blue=vehicle)
34b. ✅ Show detection labels with confidence percentages
34c. ✅ Add detection list below image with category, species, and confidence details
34d. ✅ Make image cards clickable to open detailed viewer modal

#### Basic Visualization (MVP)
43. ✅ Create simple dashboard with summary statistics (DashboardPage with detection counts by category)
43a. ✅ Add detection statistics API endpoint (/api/projects/{id}/detection-stats)
43b. ✅ Install recharts library for data visualization
43c. ✅ Implement pie chart showing detection category distribution (animal/person/vehicle)
43d. ✅ Add summary cards showing total detections and breakdown by category with percentages
45. ❌ **Add detection count timeline** (bar/line chart - not implemented)

#### Electron Integration (MVP)
4. ✅ Configure Electron shell to launch backend and load frontend (main.ts with uvicorn/PyInstaller support, health checks, graceful shutdown)

### Phase 2: Packaging & Distribution (CRITICAL VALIDATION)

**🚨 DECISION POINT**: Validate packaging/distribution pipeline before building more features.

53. ✅ Set up PyInstaller for backend bundling (backend.spec configured, onedir mode)
54. ✅ Configure electron-builder for macOS and Windows installers (package.json build config complete)
55. 🚧 Bundle micromamba binary (configured in extraResources but not tested end-to-end)
56. ✅ Configure Windows code signing (CSC_IDENTITY_AUTO_DISCOVERY in workflow)
56a. ✅ Fix missing requests module in PyInstaller bundle (added to requirements.txt + hiddenimports)
56b. ✅ Implement smart notarization retry (5min → 10min → 30min with ticket validation)
57. ✅ **Test installation on clean macOS machine** (v0.1.0 release validated by user)
58. ✅ **Test signed Windows installer** (GitHub Actions builds Windows .exe + .zip)
59. ❌ Create Linux AppImage configuration (not in current workflow)
60. 🚧 **Validate cross-platform compatibility** (macOS + Windows working, Linux not tested)
61. ✅ Set up GitHub Actions for automated builds and signing (build-electron.yml with parallel macOS/Windows jobs)
62. ✅ Test automated release pipeline (releases created with DMG, ZIP, EXE artifacts)

**✅ PACKAGING VALIDATED** - macOS notarized, Windows signed, releases working!

### Phase 3: Feature Expansion (After MVP Validates)

#### Advanced ML
28. ✅ Real-time progress tracking via WebSocket (already implemented)
29. ❌ Integrate species classifiers (EUR-DF, AFR-BASIC runners needed)
30. ❌ Implement multi-model support and sequential pipelines (detect → classify)
31. ❌ Add GPU detection and CPU fallback logic

#### Review & Annotation
35. ❌ Add confidence threshold filtering (API supports it, UI filter not built)
36. ❌ Implement "mark as reviewed" functionality with audit log
37. ❌ Add keyboard shortcuts for navigation (next/previous file)
38. ❌ Build bounding box drawing tool (new annotations) - Currently SVG read-only
39. ❌ Implement bounding box editing (resize, move, delete) - Konva not integrated
40. ❌ Add species category assignment (dropdown)
41. ❌ Implement annotation save/update API with audit log tracking
42. ❌ Build batch selection and operations UI

#### Event Grouping & Advanced Browsing
17. 🚧 File list view (ImagesPage exists with grid, TanStack Table not used, pagination not implemented)
18. ❌ Implement event grouping logic (Event model exists but time-clustering not integrated in workers)
19. ❌ Create event list and event detail views

#### Advanced Visualization
44. 🚧 Species distribution chart (pie chart with Recharts done, Plotly not integrated)
46. ❌ Implement site map view with MapLibre GL JS + Deck.gl
47. ❌ Add interactive site location editing (drag markers)
48. ❌ Implement heatmap layer for detection density
49. ❌ Add hexbin/grid visualization for camera coverage analysis

#### Export & Reporting
50. ❌ Build CSV export functionality (AsyncIO task with WebSocket progress)
51. ❌ Implement Camtrap DP export format
52. ❌ Create export download API and UI

#### Background Processing
13. ✅ AsyncIO workers implemented (detection_worker.py with async queue processing)
14. ✅ WebSocket manager for real-time progress updates (websocket.py router)
15. ✅ Build jobs and audit_log database tables
16. 🚧 Import job progress tracking (detection job progress works via WebSocket, file import progress not separately tracked)

#### Folder Management
11e. ❌ Add folder validation and re-linking functionality

### Phase 4: Polish & Release

63. 🚧 Error handling (basic error messages, no comprehensive error boundary)
64. 🚧 Loading states (basic spinners, not comprehensive)
65. ❌ Write in-app help/onboarding content
66. ❌ Create end-to-end tests for critical workflows (Playwright)
67. ❌ Performance optimization for large datasets (not tested with 100K+ images yet)
68. ❌ Write user documentation (installation, workflows)
69. 🚧 Developer documentation (DEVELOPERS.md exists, architecture details limited)
70. 🚧 README (exists but needs screenshots and getting started guide)
71. 🚧 Internal testing with real camera trap datasets (basic testing done, not comprehensive)
72. ❌ Beta testing with domain experts
73. ❌ Bug fixes and UX improvements based on feedback
74. ✅ GitHub release with installers (v0.1.0 published with macOS DMG/ZIP + Windows EXE/ZIP)

---

---

## CURRENT STATUS SUMMARY (Updated: Dec 2024)

### Overall Progress: **~65% of MVP Complete**

**✅ COMPLETED PHASES:**
- **Phase 1 (Foundation)**: 95% complete - All infrastructure, data models, API endpoints, and Electron shell working
- **Phase 2 (Packaging)**: 90% complete - macOS + Windows installers working, notarized/signed, automated releases ✅

**🚧 IN PROGRESS:**
- **Phase 3 (Features)**: 40% complete
  - ✅ MegaDetector integration fully working
  - ✅ File browsing with detection visualization
  - ✅ Basic dashboard with statistics
  - ❌ Species classifiers not integrated
  - ❌ Annotation editing not implemented
  - ❌ Event grouping not wired up

**❌ NOT STARTED:**
- Export functionality (CSV, Camtrap DP)
- Maps visualization (MapLibre/Deck.gl)
- Interactive annotation tools (Konva)
- GPU detection/fallback
- E2E testing
- Comprehensive documentation

### What Works Today:
1. ✅ Create projects, sites, deployments
2. ✅ Scan folders for camera trap images (EXIF extraction)
3. ✅ Download MegaDetector model + build isolated environment
4. ✅ Run MegaDetector on images (animal/person/vehicle detection)
5. ✅ View images with bounding box overlays
6. ✅ See detection statistics and pie charts
7. ✅ Install via signed DMG (macOS) or EXE (Windows)
8. ✅ Real-time progress updates via WebSocket

### Critical Gaps for Full MVP:
1. Species classifiers (EUR-DF, AFR-BASIC) - runners needed
2. Annotation editing (draw/edit bounding boxes)
3. Event grouping (time-based clustering)
4. Export functionality (CSV, Camtrap DP)
5. Performance testing with 100K+ images
6. User documentation

### Next Priorities:
Based on user feedback focusing on "contents" over packaging, recommended order:
1. Implement species classifier runners (reuse MegaDetector pattern)
2. Add CSV export for detections
3. Build event grouping logic into file import worker
4. Add confidence threshold filtering in UI
5. Performance optimization and testing with large datasets

---

## Original Plan Summary

This plan provides a realistic, implementable roadmap for building a local-first camera trap analysis platform with:

- **Pragmatic technology choices** optimized for a small team (1-2 developers) with Python/ML expertise
- **Clear architecture** separating concerns (Electron shell, FastAPI backend with AsyncIO workers, React frontend, micromamba environments)
- **Real-time updates** via WebSocket streaming for job progress (imports, ML inference, exports)
- **No Docker dependency** using micromamba for isolated Python environments
- **Scalable data model** (SQLite → Postgres migration path) supporting 100K-1M+ images with audit logging
- **Modern UI** with shadcn/ui components, Konva canvas annotation, keyboard shortcuts, and responsive design
- **Advanced geospatial analysis** using MapLibre GL JS + Deck.gl for heatmaps, hexbins, and coverage visualization
- **Cross-platform packaging** with Windows code signing, macOS and Linux support
- **Phased milestones** delivering value incrementally over 3-6 months to MVP (72 detailed implementation steps)

The architecture is designed to work fully offline (with LiteQ fallback queue) while maintaining a clear path to future cloud/hybrid deployment. All choices prioritize simplicity, maintainability, and leveraging existing open-source tools over custom solutions.
