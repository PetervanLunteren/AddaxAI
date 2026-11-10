# Video frame storage: crop + 512 thumb instead of a full best frame

Plan for a future change. Not scheduled. Captured here so we can pick it up without re-deriving the design. None of this is built. It records the current video-frame storage model, why it costs more disk than it needs to, and the smaller-footprint scheme we would switch to if storage becomes a real problem.

The likely trigger is marine BRUVS support, where videos are long and store far more per clip. See [marine-bruvs-mode.md](marine-bruvs-mode.md); the two notes interact (last section).

## Why

Beta feedback (Dan, 2026-07) flagged video frame storage as a near-MVP concern: "video users will run out of disk space." His worry was the old behaviour that wrote every detection-bearing frame to disk. That is already fixed: since 2026-05-13 (commit `13ba233`) frames are streamed through memory and only **one** best-frame JPEG per video is persisted. So the "doubling storage" fear is gone.

But the retained frame is still larger than the app strictly needs. It is stored at up to 1920 px so the full-detail zoom modal stays sharp, yet that full-size frame is used on only two rare surfaces. The two hot surfaces (the Labels crop grid and the Counts/event overview) need far less. This note is about reclaiming that gap: roughly 70% of the per-video frame footprint, at the cost of an on-demand decode on the rare surfaces.

This is a deliberate defer, not a rejection. At current scale (handfuls to low thousands of videos) the saving is not worth the added moving parts (YAGNI). It becomes worth it at tens of thousands of videos, or when marine mode multiplies per-video storage.

## Current architecture

**One JPEG per video, written at analysis time.**
- `select_best_frames_streaming` (`backend/app/ml/best_frame.py`, no-classifier path) and the classifier worker (`backend/app/ml/inference/classification_worker.py`, fused path) both decode the source video frame by frame in memory, pick the best, and write a single JPEG via `write_best_frame` (`backend/app/ml/inference/video_iter.py:142`): capped at `max_dim=1920`, `quality=80`, progressive, optimized, never upscaled.
- Path scheme: `{deployment}/.addaxai/video_frames/<relative_video_path>/frame{N:06d}.jpg`. Stored on the File row as `best_frame_path` + `best_frame_number` (`backend/app/models/file.py`).

**Everything else is derived from that one frame at read time.**

| Surface | Component | Request | Source |
|---|---|---|---|
| Labels crop grid (hot) | `LabelsTab.tsx` grid → `crop_service` | `GET /api/detections/{id}/crop?size=` (default 200, max 512) | crops the bbox out of `best_frame_path` |
| Counts / event overview (hot) | `FrameThumbnail.tsx` | `GET /api/files/{id}/image?size=thumb` | resizes `best_frame_path` to 512 in memory |
| Full-detail zoom (rare) | `DetectionDetailModal.tsx:422` | `GET /api/files/{id}/image` (full) | serves `best_frame_path` at 1920 |
| Cine-loop still (rare) | `AnnotationCanvas.tsx` in `EventDetailModal` autoplay across >1 file | `GET /api/files/{id}/image` (full) | serves `best_frame_path` at 1920 |
| Video view / filmstrip | `VideoPlayer` / `VideoFilmstrip` | plays the real file / `GET /api/files/{id}/filmstrip` | not the best frame; filmstrip decodes on demand, persists nothing |

Two facts that shape the design:
- **Crops are not persisted.** `crop_service.get_or_create_crop` (`backend/app/services/crop_service.py`) caches JPEG bytes in an **in-memory LRU of 2000 entries** (`_cache`, line 22-23), regenerated from `best_frame_path` on every cold start and evicted under churn. So today the Labels grid depends on the best frame being on disk at read time.
- **On-demand single-frame decode already exists and is accepted.** The filmstrip endpoint (`get_file_filmstrip`, `backend/app/api/routers/files.py:191`, sync `def` on purpose) opens the clip with cv2 and decodes frames live, nothing stored. `backend/app/ml/inference/video_iter.py` has the seek/decode helpers. This is the pattern the rare surfaces would reuse.

## What we measured

Frame and crop sizes, encoded with the app's real params (q80/q85, optimize, progressive). "Realistic" discounts the pure-noise worst case, which real scenes never hit:

| Artifact | Realistic size |
|---|---|
| Best frame, 1920 px q80 (1080p source) | ~150-500 KB (typ. ~300 KB) |
| Whole-frame thumb, 512 px q85 | ~50-80 KB |
| Crop, 200 px q85 | ~13 KB |
| Crop, 512 px q85 | ~82 KB |

Single-frame decode from a clip (open + seek + decode one frame): **~40-70 ms** on a local SSD with a trivial codec; realistically **~150-500 ms** for real H.264/HEVC on an external USB drive (the common camera-trap setup). A grid of 200 video crops decoded live would be ~14 s locally, worse on USB. This is exactly why the best frame exists as a decode-once cache, and why the two hot surfaces must stay served from stored bytes, not live decode.

## Proposed change: store crops + a 512 whole-frame thumb, drop the 1920 frame

At analysis time, from the full-resolution decoded best frame that is already in memory, write two things instead of one 1920 JPEG:

1. **A 512 px whole-frame thumbnail** (q85), replacing the 1920 frame as `best_frame_path`. Serves the Counts/event overview directly (already ≤512, no resize) and any whole-frame fallback.
2. **A persisted crop per detection on that frame**, written to disk under the same `.addaxai/video_frames/...` tree. Serves the Labels grid without a live decode. Cut from the full-resolution frame (better quality than cutting from the already-downscaled 1920 frame).

Read-time changes:
- Labels crop grid: serve stored crops. `crop_service` gains a disk-backed layer (persist at analysis time; the in-memory LRU stays as a hot cache in front of it).
- Counts/event overview: unchanged request, now reads the 512 thumb directly.
- Full-detail zoom and cine-loop still: switch from `best_frame_path` (1920) to **on-demand decode from the source video**, reusing the filmstrip decode path, with a spinner. Accepted: both are rare, and the zoom modal is a deliberate single action.
- Bbox redraw on a video detection (`invalidate_crop_cache`, `backend/app/api/routers/detections.py:86`): re-cut the crop via one on-demand decode. Rare, same path.

Keep the rule simple and uniform: **hot grids read stored bytes, rare full-frame views decode on demand.** No per-case heuristics.

## Expected gain

Per video, ~300 KB (one 1920 frame) becomes ~85 KB (one 512 thumb + ~1.5 crops at 200 px). Roughly **70% less**, a fixed cost per video regardless of clip length.

| | Per video | 50k videos |
|---|---|---|
| Now (1920 frame) | ~300 KB | ~15 GB |
| Crop + 512 thumb | ~85 KB | ~4 GB |

## Trade-offs and risks

- **Rare surfaces get slower.** Full-frame zoom and cine-loop become a ~150-500 ms decode with a spinner instead of instant. Explicitly accepted for the saving.
- **Requires persisting crops to disk, a new artifact and lifecycle.** Today crops are memory-only. This adds write-at-analysis-time and invalidate-on-edit disk logic. It is the bulk of the work and the main new moving part.
- **Source video must be present for the rare surfaces and for bbox re-cuts.** Verification already requires the source media on disk (images serve from `file_path`), so this is consistent, not a new constraint. If a drive is unplugged, zoom/re-crop fail the same way image verification already would.
- **Crop count scales with detections per frame.** Camera-trap frames have 1-3 animals, so ~1.5 crops/video holds. A busy frame stores more crops; see marine interaction below.
- **No backward-compat path is needed** (project convention: no users to protect), but existing runs carry 1920 frames and no persisted crops. Simplest handling: apply the new scheme to new analyses only and let reprocessing regenerate; a full migration/regeneration of old runs is not worth building. Decide at implementation time.

## Interaction with marine BRUVS mode

Marine BRUVS is the scenario that most likely forces this. Those clips are long (30+ min) and multi-species, and the MaxN model needs a peak frame **per species**, not one best frame per clip (see [marine-bruvs-mode.md](marine-bruvs-mode.md)). That multiplies persisted frames and crops per video, so a lean per-frame footprint matters far more there than for camera traps. If marine lands first, design the storage scheme with it in mind (per-species peak frames + their crops), and this camera-trap-scale change likely folds into that larger effort rather than shipping on its own.

## Effort and when to do it

Medium. It touches the write path (`best_frame.py`, `classification_worker.py`, `video_iter.py`), the crop service (add disk persistence), the image endpoint (rare surfaces to on-demand decode), and a new/extended single-frame decode endpoint, plus tests for the write path, crop persistence/invalidation, and the on-demand fallback.

Do it when either is true: real users hit a storage wall (tens of thousands of videos), or marine BRUVS support starts (fold it in there). Until then, the current one-lean-frame-per-video model is fine.

## Key files

| File | Role |
|---|---|
| `backend/app/ml/inference/video_iter.py` | `write_best_frame` (frame write), seek/decode helpers to reuse |
| `backend/app/ml/best_frame.py` | no-classifier best-frame path |
| `backend/app/ml/inference/classification_worker.py` | fused classify + best-frame path |
| `backend/app/services/crop_service.py` | crop generation; would gain disk persistence |
| `backend/app/api/routers/files.py` | `/image` (+ `?size=thumb`), `/filmstrip` on-demand decode |
| `backend/app/api/routers/detections.py` | `/crop` endpoint, `invalidate_crop_cache` |
| `backend/app/models/file.py` | `best_frame_path`, `best_frame_number` |
| `frontend/src/components/verify/FrameThumbnail.tsx` | 512 whole-frame thumb consumer |
| `frontend/src/components/verify/DetectionDetailModal.tsx` | full-frame zoom (would move to on-demand) |
| `frontend/src/components/verify/AnnotationCanvas.tsx` | cine-loop still (would move to on-demand) |
