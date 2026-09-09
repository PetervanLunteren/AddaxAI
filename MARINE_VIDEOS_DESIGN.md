# Marine videos in AddaxAI: audit, investigation and design

Written 2026-09-05 on the `marine-videos` branch. This is the design Peter chose (option A of three, see "Options considered"). Option B is written up as a future plan in `future-plans/marine-track-timeline.md` so it can be added on top of this design later without a rewrite.

## The answer in short

AddaxAI gets no marine mode and no third card. It gets a tracker. The two marine detectors (SharkTrack and the Community Fish Detector) enter the model catalog as ordinary detectors, in a new `marine` environment. A new analysis setting, "Track animals across frames", runs detection and BoT-SORT tracking in one pass and stores every box with a track id. The one rule that decides what a video shows on still surfaces ("a video detection is visible only on the best frame") gains one clause: "or it is a track's representative box". Everything downstream already reads that rule through one helper, so the Labels grid shows one card per track, X and relabel act on the whole track, MaxN takes its peak across every frame, and the Counts page gains the MaxN frame with a jump to it. Camera trap users see nothing new unless they switch tracking on.

Two increments. The first ships the environment, the four detectors, the tracker, the tracks table and the track cards. The second ships MaxN frame and time on the Counts page and in `counts.csv`, first arrival, and tracking for MegaDetector itself.

## What was investigated

| Source | What it gave |
|---|---|
| Gmail threads "Shark detector + dekstop tool" (2026-06-09 to 06-10) and "SharkTrack in AddaxAI" (2026-06-16 to 07-09), all 17 messages | Demian Chapman's requirements, Dan Morris on the Community Fish Detector, the MegaDetector package 10.0.24 RF-DETR support, the Mote test videos |
| `github.com/filippovarini/sharktrack`, the local clone, `fvarini.com/sharktrack` and its annotation pipeline page | Full source read: model, tracker config, post-processing filter, output CSV, review loop, MaxN script |
| SharkTrack paper, Varini et al. 2024, arXiv 2407.20623, all 27 pages | Architecture, training data, tracker tuning, accuracy and timing numbers |
| `github.com/filippovarini/community-fish-detector`, its releases API, the dataset repo, LILA | Model variants, weights, sizes, licences, inference script, output format |
| MegaDetector package source (`run_detector.py`, `pytorch_detector.py`, `rfdetr_detector.py`, `process_video.py`, `video_utils.py`) | How RF-DETR and unknown YOLO checkpoints load, class mapping, frame numbering, streaming |
| ultralytics docs and source (`modes/track`, `trackers/bot_sort.py`, `trackers/utils/matching.py`) | BoT-SORT API, the `lap` dependency, frame-by-frame tracking |
| `pypi.org/pypi/rfdetr` | Version 1.10.0, dependency pins |
| VIAME / DIVE docs (data formats, shortcuts), BIIGLE video manual, benthic stereo-BRUVs field manual, GlobalArchive manual, CheckEM, FishID (Griffith), Villon et al. 2024, Connolly et al. 2022 (Frontiers) | How the marine community reviews video and reports MaxN |
| The AddaxAI repo: root MD files, `DEVELOPERS.md` in full, backend workers, ML pipeline, models, crud, routers, frontend verify components, env YAMLs, catalog | The audit below |

The Okinawa paper (arXiv 2605.10449) that a web search returned for "Community Fish Detector" is unrelated. It is a different group's segmentation and tracking pipeline. It is not cited here beyond this note.

## How AddaxAI handles camera trap data today

This is the architecture a reviewer needs to hold in mind. It is what the design leaves alone and what it touches.

### Two entry points, one pipeline

The home screen offers "Analyse a folder" (a three step flow: setup, labels, save) and "Build a project" (sites, deployments, labels, counts, dashboard, insights, export). A folder run is a project with `Project.mode = "folder_run"` and one hidden deployment, so both share every worker, table and query. Promotion to a research project flips one column.

### The analysis worker

`detection_worker._process_batch_job` runs per deployment, all or nothing:

| Phase | What | Where |
|---|---|---|
| 1 | Video detection: MegaDetector's `process_video`, `--time_sample = 1 / project.video_fps`, default 1 fps, output threshold 0.01 | `ml/inference/video_detector.py` |
| 2 | Video classification: the classifier subprocess walks each video, crops the animal boxes, writes the best frame JPEG | `ml/inference/classification_worker.py` |
| 3 | Image detection with checkpoint and resume | `ml/inference/megadetector.py` |
| 4 | Image classification | same subprocess pattern |
| 5 | JSON merge | `ml/json_pipeline.py` |
| 6 | Load to database, best frame stamped from the JSON before any decode | `json_pipeline.load_json_to_database` |
| 7 | Postprocessing: smoothing, rollup, exclusion, events, MaxN | `ml/postprocessing.py` |
| 8 | Embeddings for the similarity grid | `workers/embedding_worker.py` |

Every model runs in a micromamba environment as a subprocess. Detection is hardcoded to `env-addaxai-base` in both `MegaDetectorV1000.__init__` and `VideoDetectionModel.__init__`. Classifiers pick their environment from the catalog's `env` field (`CustomClassificationModel`). The base env pins `megadetector==10.0.24`, `ultralytics==8.3.177`, torch 2.8, Python 3.11.

### The data model that matters for video

- `File`: one row per image or video. A video carries `frame_rate`, `frames_processed`, `best_frame_number`, `best_frame_path`, `duration_seconds`.
- `Detection`: one row per box per sampled frame, with `frame_number`, `category`, `confidence`, `label`, `verified`. Every sampled frame's boxes are stored at the 0.01 cap.
- `Event`: files of one deployment clustered by the independence interval (default 30 min). One 90 minute video is one file, and GoPro chapter files of one drop fall within the interval, so they become one event.
- `EventObservation`: one row per species per event with `max_n`, `max_n_file_id`, `human_count`, sex, life stage, behaviour.

### The best-frame rule

The load bearing rule for videos, stated in `ml/detection_visibility.py`:

> a video detection is visible only when `Detection.frame_number == File.best_frame_number`, or it is verified

The best frame is chosen from the JSON by summed detection confidence at or above 0.3, category blind (`scoring.choose_frame_number`), then one JPEG is written under `<folder>/.addaxai/projects/<id>/video_frames/<video>/frame{N:06d}.jpg`. Every still surface applies the rule: the Labels grid (`similarity_script._COMMON_JOINS`), the crop service (`_resolve_image_path`), embeddings (`build_embedding_input`), the label filter counts and progress (`routers/labels.py` twice), the annotated stills and species folders (`annotated_copies.py` twice, `separate_folders.py`), the Files export, `File.observation_type` (`strongest_passing_detection` over the visible surface), the MaxN species gate (`calculate_max_n_for_event`, `allowed_video_keys`), and the frontend canvas (`shouldDrawBbox`). The video player is the one surface that draws every frame's boxes.

### MaxN

`calculate_max_n_for_event` already groups detections by (file, frame, species), takes the peak count per species across every frame of every file in the event, and then drops video species that do not appear on the best frame (or were verified somewhere). So the MaxN that BRUV science needs is computed today; the best-frame gate removes species the user could not see or clean. `EventObservation` stores the winning file but not the winning frame.

### Categories

`Detection.category` and `File.observation_type` carry the detector's own class names. `json_pipeline` reads the run's `detection_categories` map and refuses an undeclared id. Classification, however, only runs on MegaDetector category `"1"` (`json_utils.extract_animal_detections`), and the category colours know only animal, person and vehicle (`label_colors.CATEGORY_COLORS`, `detection-utils.getCategoryColor`). Fifty two backend and fourteen frontend sites mention `"animal"` literally; most are counters and copy, the two that matter are named above.

### Settings that are inference time

`video_fps`, `media_filter`, `detection_image_size`, `detection_augment` are read by a new analysis and never by postprocessing. `compute_postprocessing_settings_hash` does not include them, so changing them never triggers a reprocess. A tracking flag belongs in that group.

### Deleting and migrating

Every child table is emptied leaves first in `purge_deployment_data` (embeddings, detections, observations, event files, events, files). A new child table must join that list and declare `ON DELETE CASCADE`. Migrations are immutable once shipped; `tests/db/test_migration_keeps_rows.py` runs every migration against a seeded database.

## The two models

### SharkTrack

| | |
|---|---|
| What | Single class elasmobranch detector plus BoT-SORT tracker plus a MaxN script. Paper: Varini et al. 2024, arXiv 2407.20623. MIT. |
| Detector | ultralytics YOLOv8 nano, 640 px, trained 500 epochs on 6,862 frames from 77 BRUVS clips at 25 locations, plus 587 ray images. mAP50 0.85. The checkpoint (`models/sharktrack.pt`, 6 MB) is an ultralytics `DetectionModel` with `names = {0: "elasmobranch"}`. |
| Tracker | BoT-SORT via `ultralytics==8.1.47`, 3 fps (chosen as the MOTA versus speed sweet spot, MOTA 0.77), `track_high_thresh 0.4`, `track_low_thresh 0.2`, `new_track_thresh 0.4`, `track_buffer` = 2 s of frames, `match_thresh 0.97`, `gmc_method sparseOptFlow`, no ReID. |
| Post-processing | A track is dropped when it lasts under one second or its centre moves under 8% of the frame, and its max confidence is under 0.7. Removes 40% of false positive boxes at under 0.08% true positive loss. |
| Outputs | `internal_results/output.csv` (one row per box per frame: `video_path, video_name, frame, time, xmin, ymin, xmax, ymax, w, h, confidence, label, track_metadata, track_id`), `overview.csv`, and one JPEG per track at its highest confidence frame, named `<track_id>.jpg`. |
| Review loop | The person deletes the JPEG of a false track and renames the rest to `<track_id>-<species>.jpg`. `compute_maxn.py` propagates the label to every box of the track, counts boxes per (video, frame, label), and takes the max per (video, label). Output `maxn.csv` with the frame and time of MaxN and the track ids in it. |
| Results | 89% MaxN accuracy over 207 hours at three unseen sites (Red Sea 81.8%, Anguilla 87.8%, Maldives 97.8%). 2 minutes of human work per video hour against 42 manual. 27 minutes of inference per video hour on a 2019 six core i7 laptop CPU. Up to 95% of raw boxes were false positives in turbid seagrass water, still cleaned in minutes because a track is one decision. |
| Also | Peek mode (keyframes only, no tracks, 5 times faster). GoPro chapter aggregation. Stereo prefix filter. An optional DenseNet species classifier that is site specific and not released. |

What SharkTrack proves for this design: the tracker is what turns 16,000 frames of boxes (90 minutes at 3 fps) into a few dozen decisions, one screenshot per track is enough to label a species, and MaxN from cleaned tracks matches experts.

### Community Fish Detector (CFD)

| | |
|---|---|
| What | Single class "fish" detector trained on the Community Fish Detection Dataset (17 open datasets, 1.9 M images, 935 k boxes, marine, freshwater and lab). Led by Filippo Varini and Dan Morris. Apache licence for the RF-DETR weights. No species, no tracker. |
| Variants | RF-DETR nano 640 px (AP .596, 122 MB), small 1024 px (AP .606, 127 MB), medium 1024 px (AP .609, 135 MB). A YOLOv12x variant exists but is AGPL and deprecated by the authors. Latest release `2026.07.06-release`, a container re-issue for current `rfdetr`; the old files do not load with the new code and the other way round. |
| Runtime | MegaDetector package 10.0.24 loads a `.pth` through `RFDETRDetector` (`rfdetr.from_checkpoint`, architecture and resolution read from the checkpoint, `detection_categories = {"0": "fish"}`), when the `rfdetr` package is installed. Output is standard MegaDetector JSON. Dan's words: "just pass the filename of a CFD model to run_detector_batch". |
| Dependencies | `rfdetr` 1.10.0 (2026-09-04): Python 3.10+, torch 2.2+, `transformers 5.1+`, `supervision 0.29+`, `pydantic 2`. |

Two facts that shape the pipeline design:

1. The MegaDetector package's `PTDetector` does not read class names from a plain YOLO checkpoint. With `use_model_native_classes` off (the default) it asserts class indices in {0, 1, 2} and adds one, so SharkTrack's class 0 would become "animal". SharkTrack must load through ultralytics directly. CFD loads through the package's RF-DETR class, which does carry names.
2. ultralytics' BoT-SORT needs the `lap` package and calls `check_requirements("lap>=0.5.12")` when it is missing, which tries to pip install at run time. In a frozen offline environment that fails, so `lap` must be pinned in the YAML.

## How the marine community reviews video

- **The metric is MaxN**: the maximum number of individuals of a species in any single frame of a deployment. Per species the annotator records the count, the frame and the time of MaxN, plus often time of first arrival and time to first feed. The deployment period starts when the bait lands. (Benthic stereo-BRUVs field manual, Langlois et al. 2020.)
- **EventMeasure** (SeaGIS) is the standard tool: point annotations per species per frame, MaxN queries, lengths from stereo pairs, `.EMObs` files uploaded to GlobalArchive, checked with CheckEM. No AI in it.
- **VIAME / DIVE** is the AI assisted tool: detection plus tracking pipelines, then a person reviews tracks in a track list with a timeline, confidence filter, delete, relabel, merge (`M`, `Shift+M`), keyframes and interpolation. A track is `{id, begin, end, confidencePairs, features[{frame, bounds}]}`.
- **BIIGLE** annotates keyframes and interpolates between them; its object tracking is CSRT on points and circles, no detector.
- **FishID** (Griffith, Connolly et al.) found the same thing as SharkTrack: a fish only detector finds 70 to 89% of fish, most of the human work is species labelling, and it should be spent on the generic "fish" boxes, not on confident species calls. Their AI MaxN reached 87% against experts at 85%.
- **Villon et al. 2024** defined the MaxN accuracy metric SharkTrack reports and warned that automated MaxN over-counts when tracks fragment, which is why the human pass is per track.

Lessons taken: review per track, not per frame; the deliverable is MaxN with its frame and time; a person confirms or overrides the number; nobody asks a human to re-label 16,000 frames; the peak frame per species is at a different time for each species, so no single frame can carry a deployment.

## Decisions taken with Peter (2026-09-05)

| # | Question | Decision |
|---|---|---|
| 1 | Primary user | Shark BRUVS teams first; nothing that blocks general reef fish later |
| 2 | Deliverable | MaxN per species with the frame and time of MaxN |
| 3 | Detectors | SharkTrack and all three RF-DETR CFD variants |
| 4 | Tracker | Required, tracks are the review unit |
| 5 | Home screen | Inside the two existing cards |
| 6 | Pathways | Folder runs and projects |
| 7 | Tracking for camera traps | Optional for everyone, on by default for marine detectors |
| 8 | Default fps | 3, user adjustable |
| 9 | Track storage | A `tracks` table, detections point at it |
| 10 | Storage floor | Only boxes that belong to a track |
| 11 | Track card | The highest confidence box of the track |
| 12 | Frame time | Derived, never stored |
| 13 | Review unit | One card per track in the existing grid |
| 14 | Rejecting | X marks every box of the track false |
| 15 | Split and merge | Out of scope |
| 16 | MaxN check | Counts page, MaxN frame per species with jump to peak |
| 17 | Tracker runtime | Our own script using ultralytics BoT-SORT |
| 18 | CFD environment | A new `marine` env; detectors choose their env from the catalog |
| 19 | CPU | Supported and expected; larger models flagged as slow |
| 20 | Species | Manual per track with custom labels and taxonomy lookup |
| 21 | Exports | `counts.csv` gains MaxN frame, MaxN time and first arrival |
| 22 | Phasing | Two increments |
| 23 | Test data | Mote BRUV videos, path to follow |
| 24 | Stereo | No special handling, documented |

Two assumptions carried: BRUV metadata (bait, depth, habitat) goes in the existing deployment tags and notes; UI wording is generic, nothing in the app says "marine mode".

## Options considered

- **A. The track's representative box is the visible surface.** Chosen. One table, one clause in the visibility helper, one tracker script, no new page. Described below.
- **B. The track is an entity with its own label and its own views.** A track timeline under the player, next and previous track keys, track cards with duration, two label homes kept in step. More frontend work; better for dense reef scenes. Written up in `future-plans/marine-track-timeline.md`; it is additive to A.
- **C. A video review workspace.** A second, VIAME style review UI with split, merge and manual MaxN scoring. Rejected: it builds for per individual metrics that are not the deliverable and duplicates Labels.

## The design

### 1. The user's mental model

One sentence, true in both pathways: *when tracking is on, AddaxAI follows each animal through the video, and every animal it followed is one card on the Labels page; the count per species is the most animals seen together in one frame, and the Counts page shows you that frame.*

How it looks:

- **Setup step / project settings.** The detector list now holds "SharkTrack (sharks and rays)" and "Community Fish Detector (nano, small, medium)" beside the MegaDetectors. The video section gains a switch, "Track animals across frames", with the caption: "Follows each animal through the clip so you review one card per animal instead of one per frame. Needed for underwater and other long videos. It runs the detector on every sampled frame, so at 3 fps it is about three times slower than the default 1 fps." Choosing a marine detector switches it on and sets fps to 3. Choosing MegaDetector leaves it as the user had it (off by default).
- **Progress.** The video detection phase reads "Detecting and tracking" and counts videos, as today.
- **Labels page.** One card per track, showing the animal at its clearest moment. Everything a card can do today works: relabel, X, Enter, bulk select, keyboard labels, filters, sorting. Opening a card shows that frame with the box, and the play button plays the clip with the track's boxes highlighted in the species colour. Camera trap runs without tracking look exactly as before.
- **Counts page.** Per species the observation row shows "MaxN 3 at 12:41" and a "Show" button that jumps the player to that frame. Confirm or edit the count as today.
- **Export.** `counts.csv` gains three columns. The recognition JSON carries the boxes with their track ids.

### 2. Settings

| Setting | Where | Rule |
|---|---|---|
| `Project.video_tracking: bool` | new column, `server_default false`, on `ProjectCreate` / `ProjectUpdate`, the folder run form, the Analysis settings panel, `types.ts` | Inference time, like `video_fps`. Not in the postprocessing hash. Changing it needs a re-analysis, which the existing "settings changed" path already says for `video_fps`. |
| `video_fps` | unchanged field | When tracking is on the default becomes 3 and the caption says why. Minimum stays 0.1: the tracker's buffer scales with fps, so 1 fps still works, worse. |
| Catalog `tracking_recommended: bool` | new optional field on `ModelManifest`, default false | The form switches tracking on when a detector with this flag is chosen. Explicit configuration, no name sniffing. |
| Catalog `detector_runtime: "megadetector" \| "ultralytics"` | new optional field, default `"megadetector"` | Tells the tracker script which loader to use. `.pth` RF-DETR files go through the MegaDetector package, so they stay `"megadetector"`. SharkTrack is `"ultralytics"`. |

`process_video` stays the path when tracking is off, byte for byte.

### 3. Environments

A new `marine` environment, three YAMLs under `backend/app/ml/envs/marine/{darwin,linux,windows}/environment.yml`, copied from `addaxai-base` and extended:

```
python=3.11, torch 2.8 (cu128 on linux and windows), torchvision 0.23
megadetector==10.0.24, ultralytics==8.3.177, the bundled ultralytics-yolov5 wheel, setuptools<82
lap==0.5.12            # BoT-SORT's assignment solver, must be pinned, see above
rfdetr==1.10.0          # pulls transformers 5.x, supervision, pydantic 2
```

The four marine detectors declare `"env": "marine"`. The base env is untouched, so no existing user rebuilds anything in increment 1. The env is built when the user prepares the first marine model, through the existing prepare flow, which already handles any catalog env.

Detector env from the catalog: `MegaDetectorV1000.__init__` and `VideoDetectionModel.__init__` take the env name from the manifest (`env_manager.get_python(f"env-{manifest.env}")`), exactly as `CustomClassificationModel` does. The worker passes `det_manifest.env`. `tests/ml/test_bundled_wheels.py` and `tests/ml/test_pytorch_index.py` cover the new YAMLs automatically because they read every env YAML.

Cost to state plainly: `rfdetr` brings `transformers` 5 into the marine env. That is a large install (a few hundred MB with its own dependencies) on top of torch. It only lands on machines that prepare a CFD model.

### 4. Catalog entries

Four `det` entries in `models.json`, weights mirrored to HuggingFace under `Addax-Data-Science/<model_id>` (the downloader is HuggingFace only), each repo carrying the upstream licence file and citation:

| model_id | friendly_name | env | model_fname | detector_runtime | tracking_recommended |
|---|---|---|---|---|---|
| `SHARKTRACK-1-0` | SharkTrack (sharks and rays) | marine | `sharktrack.pt` | ultralytics | true |
| `CFD-RFDETR-NANO-640-2026-07-06` | Community Fish Detector nano | marine | `cfd-rf-detr-nano-640-2026.02.02.cp-011.20260706-release.pth` | megadetector | true |
| `CFD-RFDETR-SMALL-1024-2026-07-06` | Community Fish Detector small | marine | `cfd-rf-detr-small-1024-2026.06.06.cp-016.20260706-release.pth` | megadetector | true |
| `CFD-RFDETR-MEDIUM-1024-2026-07-06` | Community Fish Detector medium | marine | `cfd-rf-detr-medium-1024-2026.03.24.cp-011.20260706-release.pth` | megadetector | true |

`description_short` of the 1024 px variants says "Slower on CPU". `min_app_version` is the release that ships increment 1 (7.7.0 if numbering continues from 7.6.0). Weights are versioned by id per the existing rule (a re-upload needs a new id), which the CFD release naming already encodes.

### 5. The tracker script

`backend/app/ml/inference/tracking_script.py`, run in the detector's env, no `app.*` imports (the same constraint as `smoothing_script.py` and `similarity_script.py`). Command line mirrors `process_video`:

```
python tracking_script.py <model> <video_folder> --output_json <path> --fps 3 \
    --detector_runtime ultralytics|megadetector [--image_size N] [--augment] [--recursive]
```

Per video:

1. Open with `cv2.VideoCapture`, read native fps and frame count, `stride = max(1, round(native_fps / fps))`, walk frames with `grab()` and `retrieve()` every stride (SharkTrack's `stride_iterator`; the Bushnell frame 0 rule from `video_iter.py` is copied, since the script cannot import it).
2. Detect: `ultralytics` runtime calls `YOLO(model).predict(frame, conf=0.2, iou=0.5, imgsz=image_size or 640)`; `megadetector` runtime calls `load_detector(model)` once and `generate_detections_one_image` per frame (RF-DETR and MegaDetector `.pt` alike). Both yield `(xyxy, conf, cls)` and a category map: `model.names` for ultralytics, `detector.detection_categories` for the package.
3. Track: one `BOTSORT(args)` per video with SharkTrack's parameters and `track_buffer = round(2 * fps)`; `tracker.update(boxes, frame)` per sampled frame. Frame numbers are the absolute indices in the source video, as `process_video` writes them.
4. Post-process SharkTrack's filter (drop a track shorter than `fps` frames or moving under 8% of the frame when its max confidence is under 0.7). Constants live at the top of the script with the paper's numbers beside them.
5. Emit one image entry per video in MegaDetector JSON: `file`, `frame_rate`, `frames_processed`, `detections` with `category`, `conf`, `bbox` (normalised x, y, w, h), `frame_number` and `track_id` (integer, per video, starts at 1). Only boxes that belong to a surviving track are written. `detection_categories` and `info` at the top as `full_image_detection.synthesize_full_image_video_json` does.
6. Print `i/N` progress per video and the `PTDetector using device` line for the device, so `VideoDetectionModel._stream_process` parses it unchanged.

Memory is one video's sightings list; frames are never written to disk. A 90 minute video at 3 fps is 16,200 detector calls; on SharkTrack's numbers that is about 27 minutes on a laptop CPU for the nano models, and a few minutes on a GPU.

In `video_detector.py`, `detect_videos_to_json` gains `tracking: bool` and `detector_runtime: str` and swaps `_build_process_video_cmd` for `_build_tracking_cmd`. Everything else in that class (streaming, cancel, the access violation retry, the JSON existence check) is reused.

The worker's phase 1 passes `project.video_tracking`. Phase 2 (video classification) is unchanged; with no classifier the best frame pass runs as today. Phase 6 ingests the JSON as today with the additions in section 7.

### 6. Representative frames

At load time, for every track, its representative frame (the frame of its highest confidence box, SharkTrack's choice) is written as a JPEG beside the best frame, with the same writer and the same cap: `write_best_frame` in `video_iter.py`, quality 80, long edge 1920. Path: `<folder>/.addaxai/projects/<id>/video_frames/<video>/frame{N:06d}.jpg`, the exact naming the best frame already uses, so a track whose representative frame is the best frame shares the file.

This is a change from "decoded on demand, nothing written to disk" said during the questions, and here is why. The crop service, the embedding worker, the annotated stills and the image endpoint all take an image path, and the embedding and annotation code runs in subprocesses that cannot reach the database. A JPEG on disk makes every one of them work with no change beyond path resolution. On demand decoding would have meant a second streaming pattern in the embedding script and a frame cache in the crop service. The cost is disk: about 300 to 500 KB per track at 1920 px. A shark drop with 40 tracks is 20 MB; a reef video with 300 tracks is about 120 MB. It sits under `.addaxai` on the user's own drive, is deleted with the deployment by `_delete_deployment_artifacts`, and `_reclaim_legacy_video_frames` must learn to keep these files (it keeps only `frame{best}.jpg` today; it will keep every frame named on a track row).

The frames are fetched with `read_frame_by_seek` in the same pass that fetches the best frame (`best_frame.select_best_frames_streaming`), one video open per file, N seeks per video at about 85 ms each. When a classifier is configured, `classification_worker._process_video_group` is the pass that walks the video; it gets the list of wanted frames the same way it gets the best frame. Both callers already fall back to frame 0 and move the stamped number with it when a container over-reports its frame count; a representative frame that cannot be read is recorded as `frame_path = NULL` and the track is still a card, with the plain tile `CropCard` already renders on a missing crop.

### 7. Data model

One migration, `tracks`:

```
tracks
  id                           String(36) primary key
  file_id                      String(36) FK files.id ON DELETE CASCADE, indexed
  track_key                    Integer          # the id from the JSON, unique per file
  start_frame                  Integer
  end_frame                    Integer
  frame_count                  Integer          # sampled frames the track spans
  max_confidence               Float
  representative_frame_number  Integer
  frame_path                   Text nullable    # the JPEG of section 6
  created_at_utc               DateTime(timezone=True)
  unique (file_id, track_key)

detections
  track_id   String(36) FK tracks.id ON DELETE SET NULL, nullable, indexed

event_observations
  max_n_frame_number   Integer nullable        # increment 2
```

No representative detection foreign key, so there is no cycle between `tracks` and `detections`: the representative box is the detection with `(track_id, frame_number == representative_frame_number)`, and a track holds one box per frame by construction. The relationship `Track.detections` sets `passive_deletes=True` per the cascade rule; `tests/models/test_cascade_config.py` enforces it. `purge_deployment_data` gains `("tracks", delete(Track).where(Track.file_id.in_(file_ids)))` after detections and before files, and `tests/api/test_delete_cascade.py` pins the order.

Ingest (`json_pipeline._load_to_database`): when a detection carries `track_id`, the track row for `(file, track_key)` is created on first sight with the running span, count, max confidence and representative frame, and the detection row points at it. A re-ingest onto an existing file row (the reprocess matcher) matches boxes by path, bbox and frame as today; tracks are not rebuilt by a reprocess, only by a re-analysis, which deletes them with everything else.

Nothing is stored twice. The track's label is the label of its boxes, read from the representative box. The track's verdict is `verified` on its boxes. Time of anything is `File.captured_at_local + frame / File.frame_rate`, computed where shown (a helper `frame_time(file, frame_number)` in `utils/media_dates.py`, returning `None` when either input is `None`).

### 8. The visibility rule, extended

The sentence in `ml/detection_visibility.py` becomes:

> a video detection is visible when it is on the best frame, or it is verified, or it is the representative box of its track

`on_visible_frame()` gains an outer join on `tracks` and the clause `and_(Detection.track_id == Track.id, Detection.frame_number == Track.representative_frame_number)`. `on_visible_frame_of(file)` and `visible_detections(file, dets)` get the same third branch; the in memory twin reads `det.track` (eager loaded where the callers already load detections). The parity test in `tests/ml/test_detection_visibility.py` covers the new branch in both lanes.

The hand copies listed in that module each get the same clause, and the list in its docstring is the checklist:

| Place | Change |
|---|---|
| `similarity_script._COMMON_JOINS` | `LEFT JOIN tracks t ON t.id = d.track_id`, `OR d.frame_number = t.representative_frame_number` |
| `routers/labels.py` (stats, unprocessed count) | same clause through the helper |
| `crud/event_observation.calculate_max_n_for_event` | `allowed_video_keys` also admits a species that has a track representative on the file; the query adds `Track.representative_frame_number` to the row |
| `ml/embedding_utils.build_embedding_input` | a representative box embeds against `track.frame_path` |
| `services/crop_service._resolve_image_path` | a representative box crops from `track.frame_path` |
| `annotated_copies.py` (two sites) | unchanged: the still per video stays the best frame, on purpose, see section 12 |
| `crud/event.py`, `crud/statistics.py` | through the helper |
| `frontend/src/lib/detection-utils.shouldDrawBbox` | gains an optional `frameNumber` argument, the frame on screen, defaulting to `file.best_frame_number`. The rule becomes "draw when `detection.frame_number === frameNumber`". A track card opens its detail on the representative frame and passes it. |

`strongest_passing_detection` and `derive_observation_type` need no change: they take the visible surface the caller hands them, which now includes representatives. A tracked video is therefore named by its strongest representative box, the same rule as a photo with several animals. The files export, folder placement and the Files tab follow.

### 9. Labels page

- **Cards.** The grid endpoints already return one card per visible detection, so with the clause above they return one card per track plus the best frame's boxes. A representative box that is also on the best frame is one card, not two. Card count in the filter bar and in the progress counters comes from the same queries.
- **Cascade.** `bulk_relabel`, `mark_detections_false`, `verify` and `bulk_dismiss` in `crud/detection.py` gain one line each: expand the incoming detection ids to every detection sharing a `track_id` (`_expand_to_tracks(db, ids)`). The endpoints do not change their contracts. `recompute_file_verified` already rolls up per file. Undo works because the rows are kept. Reprocess never overwrites verified rows, so a cleaned track survives a threshold change.
- **Detail modal.** `DetectionDetailModal` / `FileDetailModal` show the representative frame through `GET /files/{id}/image?frame=N` (new optional parameter, serves a JPEG the file's tracks name, 404 otherwise) and pass `frameNumber` to the canvas. The video button plays the clip as today; `VideoPlayer` colours boxes by species already and adds a thicker outline for boxes whose `track_id` matches the opened card, so the person sees which animal the card was.
- **Wire.** `DetectionResponse` gains `track_id: str | null` (required on the wire, like `verified` and `job_id`, pinned by the existing wire fields test). `FileWithDetections` gains `tracks: [{id, track_key, start_frame, end_frame, frame_count, representative_frame_number}]`.
- **Drawn boxes.** A human drawn box on a video has no track and behaves as today.
- **Filters.** No new filter. The existing category filter offers "elasmobranch" or "fish" because categories pass through.

### 10. Counts page (increment 2)

- `calculate_max_n_for_event` records the winning frame in `max_n_frame_number` beside `max_n_file_id` (NULL for images and for human only rows). `get_max_n_frames` returns it, and the event detail response carries `max_n_time` derived per row.
- The observation row in `EventCountPanel` shows "MaxN 3 at 12:41" and a "Show" action; `EventDetailModal` switches to the video view and seeks to `frame / frame_rate`. `VideoPlayer` already maps `currentTime` to a frame.
- First arrival per species per event: the earliest `start_frame` over the species' tracks (label read from the representative box), derived in `crud/event_observation` for the response and for the export. Not stored.
- Confirm, override, split, demographics and notes work as today. Nothing about cohorts changes.

### 11. Exports (increment 2)

`counts.csv` (and the XLSX counts sheet, which share `_table_columns`) gain three columns after `count`: `max_n_frame`, `max_n_time`, `first_arrival_time`, blank for images and for human only rows. Times are camera local ISO strings, formatted by the existing serializer. `addaxai-detections.csv` gains `track_id`. `addaxai-recognitions.json` carries `track_id` per box because it is written from the same rows. Camtrap DP is unchanged: it has no MaxN time field, `observationType` translation stays as is, and a "fish" category becomes `animal` there by the existing rule. `docs/docs/reference/exports.md` documents the columns.

### 12. Folder run save step

Species folders and annotated stills keep one still per video at the best frame. A tracked video with two species files under the strongest representative's species, like a photo with two animals. The Save step copy already says a video gets a still with the boxes beside it; it stays true. A still per track was considered and left out: nobody asked for it, and SharkTrack users get their per track pictures on the Labels page.

### 13. Colours and category copy

`label_colors.CATEGORY_COLORS` and `detection-utils.getCategoryColor` map any category that is not person or vehicle to the animal colour, instead of the "bad" red fallback. `getObservationBadge` and `useLabelOptions` render an unknown category with its own name capitalised ("Elasmobranch", "Fish") where they render "Animal" today. The counters in `json_pipeline` (`animal_count`) keep counting MegaDetector's animal class and gain nothing; the log line is the only reader.

### 14. What camera traps keep

Invariants, each pinned by an existing or new test:

- With `video_tracking` off, phase 1 runs `process_video` with the same command line as today (`tests/ml/test_detection_command.py` compares it).
- A video with no tracks has the visible surface it has today (the parity test's existing cases).
- No row in `tracks` and no non null `Detection.track_id` exists after a run without tracking (`tests/integration/test_mixed_content_pipeline.py` asserts it).
- The base env YAML hash does not change in increment 1 (a test that hashes the shipped base YAML against a pinned value would be brittle; instead the increment's diff touches no file under `envs/addaxai-base`, checked in review).
- Every existing test passes, ruff clean, `npm run typecheck` clean.

Tracking for MegaDetector arrives in increment 2 by adding `lap` to the base env, which changes its hash and offers every existing user a rebuild once. That is the accepted price for "optional for everyone", and it is why it is not in increment 1.

### 15. Increment plan

**Increment 1: tracks in, review per track.**

| Area | Files |
|---|---|
| Env | `ml/envs/marine/*/environment.yml`; `environment_manager.get_env_yaml_path` needs no change (it resolves by name) |
| Detector env | `ml/inference/megadetector.py`, `ml/inference/video_detector.py`, `workers/detection_worker.py` (pass `det_manifest.env`) |
| Catalog | `models.json` (4 entries), `ml/schemas/model_manifest.py` (`detector_runtime`, `tracking_recommended`), HuggingFace repos |
| Tracker | `ml/inference/tracking_script.py`, `video_detector._build_tracking_cmd`, `detect_videos_to_json(tracking=...)` |
| Settings | `models/project.py`, `api/schemas/project.py`, `routers/projects.py`, `routers/folder_runs.py`, `frontend/src/api/types.ts`, `FolderRunModelStep.tsx`, the project Analysis settings panel |
| Data | `models/track.py`, `models/detection.py`, migration `20260905_1200_<rev>_tracks.py`, `crud/deployment.purge_deployment_data` |
| Ingest | `ml/json_pipeline.py` (track rows, representative frame list), `ml/best_frame.py`, `ml/inference/classification_worker.py` (extra wanted frames), `main._reclaim_legacy_video_frames` (keep track frames) |
| Visibility | `ml/detection_visibility.py`, `similarity_script.py`, `routers/labels.py`, `crud/event_observation.py`, `ml/embedding_utils.py`, `services/crop_service.py`, `lib/detection-utils.ts` |
| Review | `crud/detection.py` (`_expand_to_tracks`), `routers/files.py` (`?frame=`), `api/schemas/file.py` and `detection.py` (`track_id`, `tracks`), `DetectionDetailModal.tsx`, `FileDetailModal.tsx`, `VideoPlayer.tsx`, `AnnotationCanvas.tsx` |
| Colours | `crud/label_colors.py`, `lib/detection-utils.ts`, `hooks/useLabelOptions.ts` |
| Docs | see section 17 |

**Increment 2: MaxN frame, time, first arrival, tracking for MegaDetector.**

| Area | Files |
|---|---|
| Data | migration adding `event_observations.max_n_frame_number` |
| Counts | `crud/event_observation.py` (record frame, first arrival), `api/schemas/event.py`, `EventCountPanel.tsx`, `EventDetailModal.tsx` |
| Exports | `crud/export.py`, `postprocessing_outputs/_table_columns.py`, `tables_csv.py`, `tables_xlsx.py`, `recognition_json.py`, `docs/docs/reference/exports.md` |
| Base env | add `lap==0.5.12` to `envs/addaxai-base/*/environment.yml` |
| Form | the tracking switch is offered for MegaDetector too, caption unchanged |

### 16. Tests

Backend, all under `backend/tests`:

- `ml/test_tracking_script_cmd.py`: the command builder, pure.
- `ml/test_tracking_script.py`: runs the script's post-process filter and JSON writer on synthetic sightings (no model), pins the SharkTrack constants and that only tracked boxes are written.
- `integration/test_tracking_ingest.py`: a hand written tracked JSON loads into `tracks` and `detections.track_id`, the representative frame is the max confidence frame, a re-ingest keeps track rows.
- `ml/test_detection_visibility.py`: the third branch in both lanes, parity kept.
- `api/test_track_cascade.py`: X, relabel and verify on a representative touch every box of the track and no other; a drawn box is untouched; undo works.
- `api/test_labels_track_cards.py`: one card per track, a representative on the best frame is one card.
- `test_max_n.py`: a species only on a representative frame gets an observation; MaxN is the peak across frames, not the representative's count; increment 2 adds the frame and first arrival.
- `api/test_delete_cascade.py`: tracks are purged after detections and before files.
- `db/test_migration_keeps_rows.py`: picks up the new migration on its own.
- `test_config`, `test_bundled_wheels`, `test_pytorch_index`: pick up the marine YAMLs on their own.
- The detector env change: a test that `MegaDetectorV1000` and `VideoDetectionModel` ask the env manager for `env-<manifest.env>`.

Frontend: `shouldDrawBbox` with `frameNumber` in the existing detection utils tests. Typecheck with `npm run typecheck`.

By hand, on the Mote videos when the path arrives: a shark positive and a shark negative drop through both detectors on CPU and on the Mac GPU, timing per video hour recorded, card counts against SharkTrack's own `overview.csv` on the same file, MaxN against a manual count.

### 17. Docs

- `docs/docs/understanding/detections-events-observations.mdx`: the "Videos" section gains a paragraph on tracking (one card per animal followed, MaxN across frames, the peak frame on Counts).
- `docs/docs/guides/check-labels.mdx`: a tracked video's cards.
- `docs/docs/guides/confirm-counts.mdx`: the MaxN frame and first arrival (increment 2).
- New `docs/docs/guides/underwater-videos.mdx`: which detectors, the tracking switch, 3 fps, expected speed on CPU, one deployment per drop, stereo (use one camera's files), GoPro chapters (they merge into one event by time), what the numbers mean (MaxN, time of MaxN, first arrival), citation of SharkTrack and CFD.
- `docs/docs/reference/exports.md`: the new columns.
- `DEVELOPERS.md`: a "Tracks" section next to "Best frame selection": the extended visibility rule, the representative frame files, the cascade helper, the tracker script's constraints, the marine env and why `lap` is pinned.
- `README.md` of the two HuggingFace repos: licence and citation.

## Trade-offs, risks and open questions

- **Disk per track.** Stated in section 6. Bounded by the 1920 px cap. If reef users hit hundreds of tracks per video and complain, the next step is a smaller cap for representative frames, not on demand decoding.
- **`rfdetr` dependency weight and churn.** 1.10.0 was released the day before this document. Its `transformers` pin is a floor, not a ceiling, so the marine YAML must pin `transformers` and `supervision` exact versions too, chosen when the env is first built and recorded in the YAML. A future CFD container change (it happened once already, 2026-07-06) means new catalog ids, never a re-upload.
- **Apple GPU.** ultralytics runs on MPS; whether `rfdetr` 1.10.0 does is unverified. The script must not crash on MPS failure: it selects the device the way the classifier worker does and falls back to CPU with a logged line. Check on the Mac during increment 1.
- **False positives in turbid water.** SharkTrack reports up to 95% false boxes in seagrass; its filter removes 40% of them. The rest is human work on the Labels page, which is the workflow the paper timed. The counting threshold slider (default 0.2) also hides weak tracks because a track's boxes keep their confidence.
- **A track's label from one frame.** Labelling a track from its representative box is what SharkTrack does and what the person sees. A track that switches between two animals mid way (an identity switch) carries one label for both. Split is out of scope. **Superseded 2026-09-09:** the workaround named here is gone, because boxes are no longer drawn on videos at all. What replaces it is opening the track: the card's frame-count badge swaps the grid for that animal's frames, where a verdict applies to the frames selected, so the half that followed a different animal is relabelled directly. Split stays unbuilt and is now less urgent, since the counts read per-frame labels and come out right without it.
- **Best frame still names the still.** A tracked video's annotated still and folder come from its strongest representative, which may not be the best frame's species. Consistent with photos; documented.
- **Trap nights and rates.** The dashboard's "per 100 trap nights" reads a 90 minute drop as one trap night. BRUV science normalises per drop or per soak hour. Out of scope; the Counts page and `counts.csv` are correct, the rate tiles are not meaningful for BRUVs and the underwater guide says so. A "per deployment hour" rate is a later, separate design.
- **The hook for marine statistics.** Nothing in this design tells the dashboard that a project holds BRUV drops rather than camera traps, because nothing in increment 1 or 2 needs to know. If per drop or per soak hour rates, BRUV wording on the tiles, or a GlobalArchive export are built later, the shape is one explicit column, `Project.analysis_type` (`camera_trap` or `underwater_video`), set on the same form as the tracking switch and defaulted from the detector's catalog entry the way tracking is. It is additive: no table or rule here changes when it arrives. It is deliberately not added now, so there is no second flag whose meaning overlaps with `video_tracking` while nothing reads it.
- **AVI and other formats the browser cannot play.** `VideoPlayer` plays MP4, M4V, MOV and WebM only (`PLAYABLE_FORMATS`). BRUV footage is mostly GoPro MP4, but AVI exists. Review still works for such a file, because every track card, its detail view and the MaxN frame are stills written by the pipeline; only the play button and the jump to peak have nothing to seek. The detail modal already hides the play button for an unplayable format (`isPlayableVideo`); the Counts page's "Show" action must fall back to the MaxN frame still and say why, instead of offering a seek that cannot happen. Transcoding is not built: it costs the user their file timestamps, which is the trap the Bushnell AVI users already fell into.
- **Resume.** Video detection has no checkpoint today and the tracker script does not add one in increment 1. A crash on video 40 of 50 restarts the deployment. SharkTrack's per video resume is simple to add (write per video results as they finish and skip finished files on restart) and is the first follow-up if the Mote runs show it is needed.
- **Species classification for fish.** When a marine classifier arrives, `extract_animal_detections` must stop hardcoding category `"1"`. The clean shape is a catalog field on the classifier naming which detector categories it classifies. Not built now.
- **Timelapse.** `addaxai-recognitions.json` for a tracked run holds only tracked boxes, which is the storage floor decision. Timelapse readers see fewer near noise boxes than from a MegaDetector run. Documented in exports.
- **The prompt's "positive bias".** Demian asked for a detector that errs on the side of flagging. The tracker's `new_track_thresh` of 0.4 and the filter decide that today. If shark teams want more recall, the knob is a lower `new_track_thresh`, exposed later as an advanced setting only if asked.
- **Open question for Peter.** Increment 1 ships three columns of BRUV metadata nowhere. If a user wants soak time, bait type or depth in `counts.csv`, the deployment tags already export as JSON in the deployments table of Camtrap DP; a flat column set is a later ask.

## Not built, on purpose

Split and merge, a track timeline (see the future plan), MeanCount, lengths and stereo, a stereo prefix filter, a GlobalArchive or EventMeasure export, a per track still in the Save step, a per detection timestamp column, a marine species classifier, per drop rate statistics, a third home screen card, a checkpoint for video detection.

## Unrelated issues noticed during the audit

Mentioned only, not fixed:

1. `VideoDetectionModel._stream_process` takes progress from the first `\d+/\d+` in any output line. A model that prints a resolution like `640/640` or a date would move the bar. The image detector's parser is stricter. Worth aligning when the tracker script's output is designed.
2. `extract_animal_detections` and the `animal_count` counters read MegaDetector's category `"1"` by literal, while the ingest reads categories from the JSON map. Harmless today; the classifier gate is the one that will bite when a non MegaDetector detector meets a classifier.
3. `future-plans/marine-bruvs-mode.md` is superseded by this document and by `future-plans/marine-track-timeline.md`. Recommend deleting it when this design is accepted, per the "no redundant MD files" rule. Left in place for now.
4. The `docs` promise "AddaxAI picks one frame per video and uses only that one" becomes false for tracked videos; section 17 covers the rewrite.

## Summary in plain English

AddaxAI does not need a separate marine app. It needs to follow each animal through a video instead of looking at one frame. So we add a tracker as an option, put the shark and fish detectors in the normal model list, and make each followed animal one card on the Labels page. The counting rule the marine world uses, the most animals seen together in one frame, is already how AddaxAI counts; we only stop hiding species that were never on the single best frame, and we show the peak frame on the Counts page. Camera trap users notice nothing unless they switch tracking on. The work splits in two: first the models, the tracker and the per track review; then the MaxN frame and time in the counts and exports. The main costs are one new Python environment for the fish detector and one JPEG per tracked animal on the user's drive.
