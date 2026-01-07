# Video Processing Implementation Plan

**Status:** ✅ COMPLETED
**Started:** 2026-01-07
**Completed:** 2026-01-07

## Overview

Successfully integrated video processing into AddaxAI-WebUI following the StreamlitAddaxAI architecture:
- ✅ 4 separate sequential phases (video detection, video classification, image detection, image classification)
- ✅ Each phase has its own progress bar in the frontend
- ✅ MegaDetector's `process_video` module handles video detection with automatic frame extraction
- ✅ OpenCV-based frame extraction for video classification with per-video caching
- ✅ Separate JSON files for videos and images, merged at the end
- ✅ Tested with multiple YOLOv8 classification models (NAM, TAS)

## Architecture

### StreamlitAddaxAI Workflow (We're Matching This)

```
┌─────────────────────────────────────────────────────┐
│  Deployment Folder (videos + images)                │
└─────────────────────────────────────────────────────┘
                     ↓
    ┌────────────────┴────────────────┐
    │                                  │
┌───▼────────────────┐    ┌───────────▼──────────┐
│ Videos Found?      │    │ Images Found?        │
│ (scan_videos)      │    │ (scan_images)        │
└───┬────────────────┘    └───────────┬──────────┘
    │ YES                              │ YES
    ↓                                  ↓
┌───▼─────────────────────────────────────────────────┐
│ PHASE 1: Video Detection                            │
│ - process_video module (MegaDetector)               │
│ - time_sample=0.5 (for 2 FPS)                       │
│ - Output: addaxai-run-video.json                    │
│ - Progress Bar: "Video detection"                   │
└───┬─────────────────────────────────────────────────┘
    │
    ↓
┌───▼─────────────────────────────────────────────────┐
│ PHASE 2: Video Classification (if classifier)       │
│ - SpeciesNet OR YOLOv8 classifier                   │
│ - Updates: addaxai-run-video.json                   │
│ - Progress Bar: "Video classification"              │
└───┬─────────────────────────────────────────────────┘
    │                                  │
    │                  ┌───────────────▼──────────────┐
    │                  │ PHASE 3: Image Detection     │
    │                  │ - run_detector_batch         │
    │                  │ - Output: addaxai-run-image  │
    │                  │ - Progress Bar: "Image det"  │
    │                  └───────────┬──────────────────┘
    │                              │
    │                  ┌───────────▼──────────────────┐
    │                  │ PHASE 4: Image Classification│
    │                  │ - Same classifier logic      │
    │                  │ - Updates: addaxai-run-image │
    │                  │ - Progress Bar: "Image cls"  │
    │                  └───────────┬──────────────────┘
    │                              │
    └──────────────┬───────────────┘
                   ↓
       ┌───────────────────────────┐
       │ PHASE 5: Merge JSONs      │
       │ - Combine video + image   │
       │ - Output: addaxai-run.json│
       └───────────┬───────────────┘
                   ↓
       ┌───────────────────────────┐
       │ PHASE 6: Load to Database │
       │ - Parse merged JSON       │
       │ - Create Detection records│
       └───────────────────────────┘
```

## Key Discoveries

### MegaDetector's process_video Module

**Command:**
```bash
python -m megadetector.detection.process_video \
  model.pt \
  /path/to/videos \
  --output_json_file output.json \
  --recursive \
  --time_sample 0.5 \
  --json_confidence_threshold 0.1
```

**What it does automatically:**
- ✅ Finds all videos recursively
- ✅ Extracts frames at specified time intervals (time_sample)
- ✅ Runs MegaDetector on frames IN MEMORY (no temp files!)
- ✅ Outputs JSON with correct format:
  - `frame_rate`: 24.0 (video's FPS)
  - `frames_processed`: [0, 24, 48, 72, ...] (frame indices)
  - `frame_number`: 24 (in each detection)

**No manual frame extraction needed!**

### JSON Format (Automatic from process_video)

```json
{
  "images": [
    {
      "file": "video.mp4",
      "frame_rate": 24.0,
      "frames_processed": [0, 24, 48, 72],
      "detections": [
        {
          "category": "1",
          "conf": 0.95,
          "bbox": [0.5, 0.5, 0.1, 0.2],
          "frame_number": 24
        }
      ]
    }
  ]
}
```

## Implementation Progress

### ✅ Completed

1. **Database Schema Changes**
   - `detection.frame_number` (Integer, nullable)
   - `project.video_fps` (Float, default=2.0)
   - Alembic migration created and applied
   - Detection schema includes frame_number field

2. **VideoDetectionModel** (`app/ml/inference/video_detector.py`)
   - Wrapper for process_video module
   - Async progress callback support
   - FPS → time_sample conversion
   - Progress parsing from tqdm output

3. **Refactor detection_worker.py**
   - ✅ Separate video and image file scanning
   - ✅ Implement 4 sequential phases
   - ✅ Conditional phase execution (skip if no media)
   - ✅ Progress callback routing to correct phase
   - ✅ Helper functions: run_classification_on_json, merge_json_files

4. **JSON Merge Logic**
   - ✅ Merge video and image JSON files (merge_json_files function)
   - ✅ Preserve frame_rate, frames_processed
   - ✅ Handle missing files gracefully

5. **Database Loading**
   - ✅ Extracted load_json_to_database standalone function
   - ✅ Handles video file types correctly
   - ✅ Preserves frame_number field from JSON

6. **Cleanup**
   - ✅ Removed video_utils.py (MegaDetector handles it)
   - ✅ Removed manual frame extraction code (extract_video_frames)
   - ✅ Removed JSON post-processing for frame_number (_add_frame_numbers_to_json)

7. **Frontend Progress Bars**
   - ✅ Added phase types: "video_detection", "video_classification", "image_detection", "image_classification"
   - ✅ Updated useTaskProgress.ts with new phase types
   - ✅ Updated RunQueueModal.tsx to show 4 progress bars
   - ✅ Smart progress calculation based on phase ordering

8. **Video Frame Extraction for Classification**
   - ✅ Created `backend/app/utils/video_utils.py` with OpenCV frame extraction
   - ✅ Implemented per-video frame caching (extract → classify → clear)
   - ✅ Modified detection_worker.py to group detections by file
   - ✅ Extracts only frames with detections (memory efficient)
   - ✅ Handles both videos and images in unified classification loop
   - ✅ Fixed database UNIQUE constraint error for re-analysis

9. **Testing**
   - ✅ Mixed video+image deployment with YOLOv8 classifier (NAM model)
   - ✅ Mixed video+image deployment with YOLOv8 classifier (TAS model)
   - ✅ Video classification working end-to-end
   - ✅ Frame numbers stored correctly in DB
   - ✅ Species data displayed in frontend

## File Changes

### New Files
- ✅ `backend/app/ml/inference/video_detector.py` (wrapper for MegaDetector process_video)
- ✅ `backend/app/utils/video_utils.py` (OpenCV frame extraction for classification)

### Modified Files
- ✅ `backend/app/models/detection.py` (frame_number field)
- ✅ `backend/app/models/project.py` (video_fps field)
- ✅ `backend/app/api/schemas/detection.py` (frame_number field)
- ✅ `backend/app/workers/detection_worker.py` (4-phase architecture + video frame extraction)
- ✅ `backend/app/ml/json_pipeline.py` (load_json_to_database + file_path lookup fix)
- ✅ `frontend/src/hooks/useTaskProgress.ts` (4 phase types)
- ✅ `frontend/src/components/analyses/RunQueueModal.tsx` (4 progress bars)

## Database Schema

```sql
-- Detections table
ALTER TABLE detections ADD COLUMN frame_number INTEGER;
CREATE INDEX idx_detections_frame_number ON detections(frame_number);

-- Projects table
ALTER TABLE projects ADD COLUMN video_fps REAL DEFAULT 2.0;
```

## Progress Bar Phases

### Frontend Phase Types

```typescript
type Phase =
  | "init"           // Initial setup
  | "video_detection"     // NEW: Video detection
  | "video_classification" // NEW: Video classification
  | "image_detection"     // Image detection (renamed from "detection")
  | "image_classification" // Image classification (renamed from "classification")
  | "finalize";      // Database loading
```

### Visibility Logic

```typescript
// Show video bars only if deployment has videos
showVideoDetection = deployment.video_count > 0

// Show video classification only if videos + classifier
showVideoClassification = deployment.video_count > 0 && hasClassifier

// Show image bars only if deployment has images
showImageDetection = deployment.image_count > 0

// Show image classification only if images + classifier
showImageClassification = deployment.image_count > 0 && hasClassifier
```

## Testing Checklist

- [x] Video-only deployment with no classifier
- [x] Mixed deployment (videos + images) with YOLOv8 classifier (NAM)
- [x] Mixed deployment (videos + images) with YOLOv8 classifier (TAS)
- [x] Progress bars show/hide correctly
- [x] Frame numbers stored correctly in DB
- [x] JSON format matches streamlit exactly
- [x] Species data appears in frontend
- [ ] Video-only deployment with SpeciesNet classifier
- [ ] Image-only deployment with classifier
- [ ] Video reconstruction works (bbox overlay on videos)

## Known Issues / TODOs

- [x] Frontend progress bar component updated for 4-phase support
- [x] Fixed missing `Callable` import in detection_worker.py
- [x] Video classification implementation complete
- [x] Fixed database UNIQUE constraint on file re-analysis
- [ ] Error handling for empty videos
- [ ] Checkpoint/resume support for large video sets
- [ ] Video-specific settings UI (FPS configuration)
- [ ] Cleanup duplicate deployment entries on re-analysis

## References

- StreamlitAddaxAI: `/Users/peter/Documents/Repos/streamlit-AddaxAI/utils/analysis_utils.py`
- MegaDetector process_video: `megadetector.detection.process_video`
- Example video JSON: `/Users/peter/Downloads/example-projects-small/project_Ukraine/loc_SIMON03/dep002/addaxai-run.json`
