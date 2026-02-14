# AddaxAI-WebUI
A temporary repository to build a new AddaxAI version with backend / frontend / API / webUI features. Completely separate from its original repo https://github.com/PetervanLunteren/addaxai so that we can mess around and dont have to be gentle.

# WHEREWASI
- Implementing the actual ML stuff.

# TODO
- [ ] https://github.com/agentmorris/MegaDetector/blob/main/megadetector/postprocessing/classification_postprocessing.py
- [ ] https://github.com/agentmorris/MegaDetector/blob/main/megadetector/postprocessing/postprocess_batch_results.py


Videos are always a pain... not only for visualising it, but also for human verification, depth estimation, etc. My plan was to just run videos like we do now, but also store the best frame in the .addaxai folder. We can then use that one as an image to cover for the whole video. We'll of course have the whole video as context, but we'll use the frame as an image snapshot that basically contains all info of the video. What defines the "best" frame? Probabaly the one with the most detections, right? Since I'll be counting the number of individuals based on the max n anyway. What do you think? investigate thoroughly. I'm not in a rush. Feel free to search online too. 

How do similar camera trap management systems like Agouti, Camelot, Wildlife insights, TrapTagger, WildTrax, TRAPPER AI, eMammal do it? See below stadards to keep in mind. 
- GBIF camera trap best practices - https://docs.gbif.org/camera-trap-guide/en/ - Guidance on managing, structuring, validating, and publishing camera trap data at scale.
- Camtrap-DP (TDWG camera trap data package) - https://camtrap-dp.tdwg.org/ - The de facto data standard for camera trap datasets, defining tables, fields, relationships, and controlled vocabularies.
- Darwin Core (TDWG) - https://dwc.tdwg.org/ - A widely used biodiversity data standard enabling interoperability with GBIF and other biodiversity infrastructures.
- FAIR data principles - https://www.go-fair.org/fair-principles/ - Principles for making data findable, accessible, interoperable, and reusable.
- MegaDetector documentation (Microsoft AI for Earth) - https://github.com/microsoft/CameraTraps - Standards and conventions for animal detection models commonly used in camera trap workflows.
- eMammal camera trap protocols - https://emammal.si.edu/protocols - Best practices for camera deployment, metadata capture, QA/QC, and long-term monitoring.
- WCAG accessibility standards - https://www.w3.org/WAI/standards-guidelines/wcag/ - Accessibility guidelines applicable to research dashboards and annotation tools.
- Nielsen Norman Group usability heuristics - https://www.nngroup.com/articles/ten-usability-heuristics/ - Core UX principles for evaluating interface and workflow usability.
- OCI (Operational Camera Trap Metadata Standard) - https://github.com/tdwg/camtrap-dp/blob/main/metadata/README.md - Guidance for consistent camera trap metadata capture across projects.
- Open Geospatial Consortium standards (OGC) - https://www.ogc.org/standards - Standards for spatial metadata and georeferencing, relevant when publishing precise camera trap locations.
- Snapshot Safari / Zooniverse project design guidelines - https://help.zooniverse.org/kb/ - Guidance on annotation UI/UX, workflow design, and volunteer engagement for large-scale projects.


Here is my synthesized analysis across all research threads:                                                                                           
                                                                                                                                                         
  ---                                                                                                                                                    
  Video "best frame" strategy: research findings                                                                                                         
                                                                                                                                                         
  1. How existing platforms handle video                                                                                                               

  The camera trap software ecosystem is overwhelmingly image-first. Most platforms evolved from photo-only workflows and bolted on video support later.
  Here's what I found:

  Platform: Wildlife Insights (Google)
  Video approach: Extracts frames at fixed intervals, runs MegaDetector per-frame, presents thumbnail grid to user. No "best frame" selection — user sees

    a representative still from the first or middle frame
  ────────────────────────────────────────
  Platform: Agouti
  Video approach: Supports video natively. Shows video with embedded playback, but observations are tagged at the sequence level. No auto frame
  extraction
    for thumbnails
  ────────────────────────────────────────
  Platform: TrapTagger
  Video approach: Processes video by extracting frames at configurable FPS, runs detection per-frame. Users see extracted frames as a filmstrip. No
    automatic "best frame" ranking
  ────────────────────────────────────────
  Platform: TRAPPER
  Video approach: Server-side video processing. Extracts frames for classification. Users annotate at the sequence level
  ────────────────────────────────────────
  Platform: Timelapse (Greenberg/Calgary)
  Video approach: Designed for image sequences. Treats videos by extracting frames. Uses MegaDetector's frame_results_to_video_results to collapse
  results
  ────────────────────────────────────────
  Platform: eMammal
  Video approach: Photo-focused. Videos treated as supplementary context. Manual frame selection by researcher
  ────────────────────────────────────────
  Platform: Zooniverse/Snapshot Safari
  Video approach: Shows short video clips directly to volunteers. No frame extraction — relies on humans watching the clip
  ────────────────────────────────────────
  Platform: Camelot
  Video approach: Photo-only. No video support
  ────────────────────────────────────────
  Platform: WildTrax
  Video approach: Handles video for audio+visual monitoring. Extracts frames at intervals for tagging

  Key finding: No platform I found does automatic "best frame" selection based on detection quality. They either show the video directly, extract frames
  at fixed intervals, or use the first/middle frame as a thumbnail. This is an opportunity for AddaxAI to do better.

  2. What MegaDetector actually does with video

  This is the most important finding because it's directly relevant to your codebase. MegaDetector's video_utils.py (already used by AddaxAI via
  video_detector.py) has a two-stage approach:

  Stage 1: Frame extraction — Samples frames at a configurable rate (time-based or every-N-frames). No quality-based selection. Purely temporal sampling.

  Stage 2: frame_results_to_video_results() — This is the "canonical detection" concept. It collapses frame-level detections into video-level results by
  keeping one canonical detection per category — specifically, the detection with the highest confidence for that category across all frames. The
  FrameToVideoOptions class has:
  - nth_highest_confidence = 1 (default: use the top-confidence detection)
  - include_all_processed_frames = False (default: just keep canonical)

  Each canonical detection retains its frame_number, so you know which frame it came from.

  This is essentially your "best frame" already — MegaDetector picks the frame where each category has its highest-confidence detection. But it operates
  per-category, not per-video.

  3. What the standards say

  Camtrap-DP: The media table has fields for video (fileType, timestamp), but no concept of "representative frame" or thumbnail. Observations link to
  media files, not to specific frames within video. The standard treats video as a single media unit.

  Darwin Core / Audubon Core: The associatedMedia field can link to multimedia. Audubon Core has accessURI for media. There is a ServiceAccessPoint
  concept that could theoretically reference a frame, but no standard field for "best frame" or "keyframe".

  GBIF camera trap guide: Recommends treating video sequences as events. Individual counts come from human review. No guidance on frame extraction.

  Bottom line: The standards don't constrain you here. There's no "representative frame" concept in any standard, so your implementation won't conflict
  with anything. When you export to Camtrap-DP, the video remains the media unit — the best frame is an internal implementation detail.

  4. What "best frame" should actually mean

  Your instinct of "most detections" is close but not quite right. Here's a more nuanced breakdown:

  Your VERIFICATION_PLAN.md already nails this — it calls it a "canonical frame" selected by:
  - highest detector confidence
  - minimal motion blur
  - maximal animal visibility
  - largest projected animal size
  - minimal occlusion

  In practice, given the current state of the art, I'd recommend a tiered scoring approach:

  Recommended scoring: frame_score = detection_score + quality_score

  Detection score (primary, easy to compute):
  - Sum of confidences of all animal detections in the frame — not just count. A frame with 3 animals at 0.95 confidence scores higher than 3 animals at
  0.4 confidence
  - This naturally favours frames where animals are clearest to the detector, which correlates with visibility and size

  Quality score (secondary, cheap to compute):
  - Laplacian variance (sharpness) — cv2.Laplacian(gray, cv2.CV_64F).var(). Higher = sharper = less motion blur. This is a single line of code and is the
   standard blur detection metric
  - Normalise and weight it lower than detection score (e.g., 0.8 * detection_score + 0.2 * normalised_sharpness)

  Why "most detections" alone isn't enough

  Consider a 10-second video of a leopard walking past the camera:
  - Frame 50: Leopard is distant, small, low confidence (0.3). 1 detection
  - Frame 120: Leopard is close, sharp, high confidence (0.97). 1 detection
  - Frame 180: Leopard is very close but blurry (motion). 1 detection at 0.6

  All three frames have 1 detection, but frame 120 is clearly the best representative. The confidence score differentiates them.

  Now consider a group of zebras:
  - Frame 30: 4 zebras visible, all at 0.9+ confidence. Clear
  - Frame 60: 5 zebras visible, but one is barely in frame at 0.15 confidence
  - Frame 90: 3 zebras visible (2 left the frame)

  By "most detections", frame 60 wins. By "sum of confident detections", frame 30 wins — and that's probably the better frame because the 5th zebra in
  frame 60 is barely visible.

  Sum of confidences above threshold is a better metric than raw count.

  5. Individual counting from video

  Your approach of "max N across frames" is the standard practice in the field. This is how virtually every camera trap system counts individuals from
  video:

  individual_count(species) = max over all frames of (detections of species in that frame)

  There are no practical tracking-based approaches in current camera trap tools. MegaDetector's repeat detection elimination is about removing false
  positives (sticks detected as animals), not tracking individuals. True multi-object tracking (SORT, DeepSORT, ByteTrack) across camera trap video
  exists in research but isn't deployed in any tool I found.

  The max-N approach has a known limitation: if 5 zebras walk past in sequence (never all in frame together), max-N might say 2 when there were 5. But
  this is a well-understood limitation that ecologists accept. Your VERIFICATION_PLAN.md handles this correctly — it lets the user indicate additional
  individuals and the system proposes secondary canonical frames.

  6. Practical recommendation for AddaxAI

  Given all of the above, here's what I'd recommend:

  Best frame selection algorithm:
  For each video after detection completes:
    1. Group all detections by frame_number
    2. For each frame, compute:
       score = sum(confidence for det in frame_detections
                   if det.category == "animal" and det.confidence >= threshold)
    3. Rank frames by score descending
    4. Among top-scoring frames (within 5% of max score),
       break ties by Laplacian sharpness
    5. Store winning frame_number as best_frame_number on the File record
    6. Extract and save that frame as JPEG to .addaxai/frames/{file_id}.jpg

  Why this works well for your use case:
  - You already have all the detection data (frame_number + confidence) from MegaDetector's output
  - No additional ML inference needed — it's a post-processing step on existing data
  - Laplacian sharpness is trivially cheap (~1ms per frame, and you only need to compute it for the top N candidates)
  - The extracted JPEG serves as thumbnail for the UI, canonical frame for verification, and input for depth estimation
  - It aligns perfectly with your VERIFICATION_PLAN.md design

  What to store:
  - File.best_frame_number: int | None — the winning frame number
  - File.best_frame_path: str | None — path to the extracted JPEG (e.g., .addaxai/frames/{file_id}.jpg)
  - Keep the full detection data for all frames (you already do this via frame_number on Detection records)

  When to compute it: As a final step in the detection pipeline, after all frame-level results are loaded into the database. This way it's automatic and
  deterministic.

  Does this analysis match what you were thinking? Happy to dive deeper into any aspect.







  





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