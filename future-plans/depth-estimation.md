# Depth estimation in AddaxAI: investigation and future plan

Status: investigation only, no code written.
Date: 2026-09-17
Branch the investigation ran on: `claude/depth-estimation-investigation-bvd8p7`
Repo state at time of audit: `abe69cd` on `main`, VERSION `0.0.0-dev`

This document is the raw material for a proper implementation plan. It holds the
original brief, the questions to answer, what the audit of the repo found, what the
literature says, the model comparison, the recommended design, what to deliberately
leave out, the risks, and the sources. It is written to be picked up cold months
later without rerunning the investigation.

---

## 1. The original brief

Verbatim, as given:

> I would like you to do an full audit and investigation of the repo and see how and
> if there could be a way of estimating depth in images and videos. I looked into depth
> anything a few months ago, and that seemed great, but perhaps there is something more
> suitable now. I would be interested in whether or not we can have the images and or
> video's depth mapped automatically, or with calibration points? Would the output be
> relative depth or In Meters? Can we use it to say anything about the size of the
> animal, the height, where do we measure it from? Bottom center of the bbox, or center
> of the bbox? Or an average of a few points? Or the closest of a few points?. Can we run
> it completely offline? And how do we use the results? I'm a separate table? Do we add a
> few cols to the detections table? Can we calibrate once per deployment since the
> background will be static, and we we need to run the depth model over all images? The
> background is static but the vegetation can grow, and the tree it is attached to can
> also slightly move in the wind… please report a full plan to me, which answers all the
> things above and recommend what you think is best in terms of UX UI KISS DRY YAGNI.
> Please take your time. I've got plenty of tokens. Be thorough. be honest. No sugar
> coating. And plan according to the principles of kiss dry yagni. A simple model that
> caters for all is a good model. I don't want you to edit any code. This is just an
> investigation. first do a thorough lay of the land audit investigation and read all the
> docs from the depth systems online and then suggest something to me and always end with
> a plain English summary of a few sentences and then we'll continue discussing from
> there.

## 2. Questions to answer

1. Can depth be mapped automatically, or does it need calibration points?
2. Relative depth or metres?
3. Can it say anything about animal size or height?
4. Where on the detection do we measure from: bottom centre of the bbox, centre of the
   bbox, an average of several points, or the closest of several points?
5. Can it run completely offline?
6. Where do the results live: a separate table, or extra columns on `detections`?
7. Can we calibrate once per deployment, given the background is static?
8. Do we need to run the depth model over every image?
9. How do we handle vegetation growth and a camera that sways in the wind?
10. What is best in terms of UX, UI, KISS, DRY and YAGNI?

## 3. Goals

- Give ecologists the camera-to-animal distance they need for camera trap distance
  sampling (CTDS) and density estimation, without the manual ranging-pole workflow.
- Fill the Camtrap DP fields that already exist in the export but are written empty.
- Keep compute cost near zero on large libraries.
- Work retroactively on data already in a project, with no re-analysis.
- Work at night, which is when most camera trap detections happen.
- Stay permissively licensed, since AddaxAI is used commercially.

Non-goals for a first version: individual animal identification, precise morphometrics,
movement speed, 3D reconstruction.

---

## 4. Repo audit: the lay of the land

### 4.1 Pipeline shape

`backend/app/workers/detection_worker.py` runs eight phases per deployment:

1. Video detection (MegaDetector over sampled frames)
2. Video classification
3. Image detection
4. Image classification
5. Merge JSONs
6. Load to database (preceded by taxonomy population)
7. Postprocessing (exclusion, taxonomic rollup, smoothing)
8. Embedding (DINOv2), fatal if configured

Each phase is a subprocess in a conda env, reporting progress over the same websocket
protocol (`ws_manager.send_progress`, phase name, phase progress, metrics dict).

### 4.2 The embedding feature is the template to copy

Everything a depth feature needs already exists in the shape of the DINOv2 embedding
feature. Files to read before writing anything:

| Concern | File |
|---|---|
| Catalog entry | `models.json`, key `models.emb` |
| Manifest schema | `backend/app/ml/schemas/model_manifest.py` |
| Catalog sync, download, staleness | `backend/app/ml/catalog_updater.py`, `backend/app/ml/model_storage.py`, `backend/app/ml/hf_downloader.py` |
| Manifest lookup | `backend/app/ml/manifest_manager.py` (`get_embedding_models`) |
| Subprocess wrapper | `backend/app/ml/inference/embedding_model.py` |
| The actual script | `backend/app/ml/inference/embedding_script.py` |
| Input building + DB write | `backend/app/ml/embedding_utils.py` |
| Standalone re-run job | `backend/app/workers/embedding_worker.py` |
| In-pipeline phase | `detection_worker.py`, phase 8, around line 1029 |
| Storage table | `backend/app/models/detection_embedding.py` |

`models.json` currently holds 7 `det` models, 35 `cls` models, 3 `emb` models. A `dep`
category would be a fourth key, and `catalog_updater.py` already iterates
`["det", "cls", "emb"]` in four places.

### 4.3 The export destination already exists

`backend/app/api/crud/export.py:208`, `_CAMTRAP_OBS_HEADERS`:

```
"individualPositionRadius",   # 15
"individualPositionAngle",    # 16
"individualSpeed",            # 17
```

These are written as empty strings today (see the row builder around line 1682).

Camtrap DP definitions, confirmed against
`https://raw.githubusercontent.com/tdwg/camtrap-dp/main/observations-table-schema.json`:

- `individualPositionRadius`: number, metres. "Distance from the camera to the observed
  individual identified by `individualID`."
- `individualPositionAngle`: number, degrees, range -90 to 90. "Angular distance from
  the camera view centerline to the observed individual."
- `individualSpeed`: number, m/s.

The five CSV/XLSX tables are documented at `docs/docs/reference/exports.md`. The
Detections table is the natural home for a `distance_m` column.

### 4.4 Videos are already solved as far as pixels go

`DEVELOPERS.md` line 873 already lists "Depth estimation" as an intended consumer of
`File.best_frame_path`:

> **Usage:** The best frame is the canonical image representation of a video. Use it
> anywhere you'd use a photo for an image file: Thumbnails in the UI, Human verification
> workflows, **Depth estimation**, Any future per-file visual feature.

Also relevant, `DEVELOPERS.md` "The best frame is the only frame a video detection can
be shown on": a video detection is displayable only when
`Detection.frame_number == File.best_frame_number`. Eleven modules enforce this. Note
that the design recommended below does **not** need pixels per detection, so this rule
does not constrain it. That is a feature, see section 7.

### 4.5 Environments

| Env | Contents relevant here | Always installed? |
|---|---|---|
| `env-addaxai-base` | python 3.11, torch 2.8.0+cu128, torchvision 0.23.0, pillow, tqdm, huggingface_hub, faiss-cpu, exiftool | Yes |
| `env-pytorch` | torch 2.8.0+cu128, torchvision, ultralytics, timm, **transformers 4.49.0**, opencv-headless, albumentations | Only when a classifier is configured |
| `env-tensorflow-v1`, `-v2`, `env-pywildlife` | per-model | On demand |

Env specs live at `backend/app/ml/envs/<env>/<platform>/environment.yml`. Linux/Windows
get cu128 wheels, darwin gets CPU/MPS.

Consequence: `transformers` is present but only in the env a detector-only project never
builds.

### 4.6 Data model facts

- `Detection` (`backend/app/models/detection.py`): normalised bbox `bbox_x/y/width/height`
  (nullable, all-or-nothing, enforced by a CHECK constraint and a Pydantic validator),
  `frame_number` for videos, `category`, `confidence`, label fields, `verified`.
- `File` (`backend/app/models/file.py`): `file_path`, `file_type` (image/video/frame),
  `width_px`, `height_px`, `captured_at_local`, `exif_data` (full EXIF as a JSON blob),
  `best_frame_number`, `best_frame_path`, `frame_rate`, `observation_type` (animal /
  person / vehicle / blank).
- `Deployment` (`backend/app/models/deployment.py`): already carries JSON columns
  `tags`, `camera_offsets`, `warnings`, plus audit columns like
  `classification_gate_used` and `datetime_offset_seconds`. Adding a JSON calibration
  column follows an established pattern.
- `Site`: lat, lon, elevation, habitat. Calibration does **not** belong here, see 8.7.
- `Project`: per-project model ids (`detection_model_id`, `classification_model_id`,
  `embedding_model_id`), thresholds, batch-size overrides.
- No focal length or field of view is parsed anywhere. `grep FocalLength` finds nothing
  outside the raw `exif_data` blob.
- 32 Alembic migrations exist in `backend/alembic/versions`.

### 4.7 Artifacts on disk

Per-deployment artifacts live in `{deployment.folder_path}/.addaxai/`, created by
`app.utils.fs_hidden.mkdir_hidden_addaxai`. Existing contents include
`video_frames/{video_name}/frame{N:06d}.jpg`, `results.json`, and transient
`embedding_input.json` / `embeddings.npz`.

### 4.8 Frontend

- Routes in `frontend/src/App.tsx`. Insights pages sit under
  `/projects/:projectId/insights/...` (map, timeline, activity-overlap,
  confusion-matrix, per-class-performance).
- `frontend/src/components/verify/AnnotationCanvas.tsx` (804 lines) is a react-konva
  canvas with zoom, pan, box drawing and hit testing. This is the component to reuse for
  clicking calibration points.
- `frontend/src/components/deployments/DeploymentInfoSheet.tsx` is the natural home for
  a per-deployment calibration section.
- `frontend/src/components/settings/AnalysisSettingsRows.tsx` is the shared-row pattern
  for settings that appear on more than one surface.
- `GET /api/deployments/preview-image` already exists
  (`backend/app/api/routers/deployments.py:212`) for showing a frame from a folder.

### 4.9 Existing depth references

`grep -rni depth` finds nothing relevant except the `DEVELOPERS.md:873` mention above.
Clean slate.

---

## 5. Literature review

### 5.1 The problem the ecologists actually have

Camera trap distance sampling (CTDS) needs the radial distance from camera to animal,
binned into a detection function, typically right-truncated around 15 m because fitted
functions go heavy-tailed beyond that.

- Howe, Buckland, Després-Einspenner, Kühl (2017), *Distance sampling with camera traps*,
  Methods in Ecology and Evolution.
  https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210x.12790
- Distance package CTDS vignette (worked example, truncation guidance):
  https://examples.distancesampling.org/Distance-cameratraps/camera-distill.html

The manual method: photograph a ranging pole at 1 m intervals along the camera axis,
then eyeball each animal photo against the pole photos. Henrich et al. 2024 did this by
superimposing transparent pole photos in PowerPoint.

- Henrich et al. (2024), *A semi-automated camera trap distance sampling approach for
  population density estimation*, Remote Sensing in Ecology and Conservation.
  https://zslpublications.onlinelibrary.wiley.com/doi/10.1002/rse2.362

### 5.2 Depth-based automation of that workflow

- Haucke, Kühl, Hoyer, Steinhage (2022), *Overcoming the distance estimation bottleneck
  in estimating animal abundance with camera traps*, Ecological Informatics.
  https://www.sciencedirect.com/science/article/abs/pii/S1574954121003277
  Preprint: https://arxiv.org/abs/2105.04244
  Semi-automatic calibration from reference images, manual effort cut by a factor > 21.

- Haucke, Steinhage (2022), *Automated distance estimation for wildlife camera trapping*
  (AUDIT), Ecological Informatics.
  https://www.sciencedirect.com/science/article/abs/pii/S1574954122001844
  Preprint: https://arxiv.org/abs/2202.04613
  Fully automated, needs neither reference images nor reference material capture.

- Johanns, Haucke, Steinhage, *Distance Estimation and Animal Tracking for Wildlife
  Camera Trapping*. Code, MIT licensed:
  https://github.com/PJ-cs/DistanceEstimationTracking
  Takes the camera's horizontal FOV in degrees as its only extra input, outputs per-frame
  animal distance in metres and 3D position, writes CSV.

- DrivenData competition, *Deep Chimpact: Depth Estimation for Wildlife Conservation*:
  https://www.drivendata.org/competitions/82/competition-wildlife-video-depth-estimation/

- Leorna, Brinkman (2022), *Estimating animal size or distance in camera trap images:
  photogrammetry using the pinhole camera model*, Methods in Ecology and Evolution.
  https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13880
  The no-neural-network baseline: flat-ground pinhole geometry.

### 5.3 The two 2025/2026 papers that decide the model choice

- *Benchmark on monocular metric depth estimation in wildlife setting* (2025).
  https://arxiv.org/abs/2510.04723
  93 camera trap images, ground truth via calibrated ChArUco patterns.
  Tested Depth Anything V2, ML Depth Pro, ZoeDepth, Metric3D.
  **Depth Anything V2 best: MAE 0.454 m, correlation 0.962, ~0.22 s per image.**
  ZoeDepth degraded badly in natural outdoor environments.
  Also notes that reported FOV, and therefore derived focal length, often deviates from
  reality when lenses are replaced or cameras refocused in the field.

- *From relative to metric: calibrating AI-based monocular depth learning models for
  distance sampling in wildlife monitoring applications* (2026), Ecological Informatics.
  https://www.sciencedirect.com/science/article/pii/S1574954126002529
  Same conclusion: Depth Anything V2 after calibration gives consistently accurate
  distances. Metric-by-default models are worse outdoors than a calibrated relative
  model.

- Aamir, Wijers, Loveridge, Markham (2025), *A robust metric distance and height
  estimation pipeline for wildlife camera trap imagery*, Ecological Informatics.
  https://www.sciencedirect.com/science/article/pii/S1574954125005291
  SSRN preprint: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5294551
  MegaDetector + SAM2 + a choice of depth models (DPT, MiDaS 3.1, DepthAnything,
  DepthPro, UniDepthV2), with linear / polynomial / piecewise regression calibration.
  Key methodological finding: **take the median depth inside the segmentation mask, not
  the mean and not over the whole bounding box**, to avoid background contamination and
  outliers. Piecewise linear regression cuts error by up to 35% versus linear when the
  depth response is non-linear. Three reference points is the stated minimum; more and
  better spread is better.

### 5.4 Prior art in camera trap software

- TRAPPER has shipped distance estimation.
  https://trapper-project.readthedocs.io/en/docs-docs-refactor/how-to/ai-pipeline/distance-estimation/
  Design, as documented: a handful of resources get reference points marked at known
  distances (one calibration sample each), a calibration action fits a model from those
  samples and stores it as that deployment's calibration result, estimation then runs
  automatically as part of the AI pipeline. Explicitly states that camera placement
  (height, tilt, angle), not just the camera or lens model, determines the depth-to-
  distance map. Roughly 20 minutes for a first calibration. Schemas in `trapper-schemas`
  (subpackage `depth`), inference in `trapper-ai-worker`.
  **This validates per-deployment calibration as the right scope.**

- Agouti has worked on distance and speed and pushed those fields into Camtrap DP.
  https://efsa.onlinelibrary.wiley.com/doi/abs/10.2903/sp.efsa.2022.EN-7327

- McMurry et al., automated camera trap density parameters:
  https://github.com/sierramcmurry/cameratrap-density-parameter-pipeline
  https://www.biorxiv.org/content/10.64898/2026.06.14.732225v1

- Zampetti et al. (2024), *Towards an automated protocol for wildlife density estimation
  using camera-traps*, Methods in Ecology and Evolution.
  https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.14450

### 5.5 Size and morphometrics from camera traps

- Paton et al. (2024), *A non-invasive approach to measuring body dimensions of wildlife
  with camera traps: a felid field trial*, Ecology and Evolution.
  https://onlinelibrary.wiley.com/doi/10.1002/ece3.11612
  Feral cat shoulder height 25.25 cm (CI 24.4, 26.1). Type of body measurement and
  posture significantly influenced accuracy.
- Cui et al. (2020), *A simple use of camera traps for photogrammetric estimation of wild
  animal traits*, Journal of Zoology.
  https://zslpublications.onlinelibrary.wiley.com/doi/10.1111/jzo.12788
- Leopard body dimensions from camera trap photographs:
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6754725/
  Careful manual photogrammetry lands within about ±5 cm. Automated bbox-based
  measurement will not come close.

### 5.6 The scale/shift ambiguity, and why it matters here

Relative monocular depth models predict affine-invariant inverse depth, up to an unknown
scale and shift, re-estimated per image. Two photos of the same static scene do not come
back on the same footing. This is the documented reason video depth models exist.

- https://arxiv.org/html/2202.01470v3 (locally scale-aligned monocular video depth)
- https://arxiv.org/html/2412.03079v1 (Align3R)
- https://openaccess.thecvf.com/content/CVPR2025/papers/Yu_Relative_Pose_Estimation_through_Affine_Corrections_of_Monocular_Depth_Priors_CVPR_2025_paper.pdf

Direct consequence: you cannot calibrate once and then apply that mapping to per-image
relative depth maps. The literature works around this with per-image alignment against a
reference. Section 7 sidesteps it instead.

---

## 6. Model comparison

| Model | Output | Size | Licence | Deps | Verdict |
|---|---|---|---|---|---|
| Depth Anything V2 Small | relative inverse depth | 24.8M params, ~100 MB | Apache-2.0 | native in `transformers` as `AutoModelForDepthEstimation` | **Recommended.** Best measured on camera trap imagery, smallest, permissive |
| Depth Anything V2 Base / Large / Giant | relative | 97.5M / 335.3M / 1.3B | CC-BY-NC-4.0 | as above | Ruled out, non-commercial |
| Depth Anything V2 Metric (Hypersim indoor max 20 m / VKITTI outdoor max 80 m) | metres | same three sizes | inherits base variant, **verify per checkpoint** | as above | Possible if you want an uncalibrated default; VKITTI is driving scenes |
| Depth Anything 3 (DA3METRIC-LARGE) | metres via `focal * out / 300` | 0.35B, ~1.4 GB | Apache-2.0 | `depth-anything-3` pip pkg + xformers, torch >= 2, 12 GB GPU min | Better geometry, but a new conda env and it needs a focal length camera traps do not report reliably. Revisit later |
| Depth Anything 3 (nested giant, outputs metres directly) | metres | 1.40B | CC-BY-NC-4.0 | as above | Ruled out |
| Video Depth Anything | temporally consistent relative | Small Apache-2.0, Base CC-BY-NC-4.0 | as DA2 | Not needed, see section 7 |
| Depth Pro (Apple) | metres + focal estimate | ~950M, ~1.9 GB | Apple ML licence, commercial use not prohibited | own pip package | Good, big, not first in the wildlife benchmark |
| UniDepthV2 | metres | ViT-S/B/L | **CC-BY-NC-4.0** | own package | Ruled out regardless of quality |
| Metric3D v2 | metres + normals | S/L/g | BSD-2 (code) | own package | Reasonable backup |
| ZoeDepth | metres | - | - | - | Degraded badly outdoors in the wildlife benchmark |

Repos:
- https://github.com/DepthAnything/Depth-Anything-V2
- https://github.com/DepthAnything/Video-Depth-Anything
- https://github.com/ByteDance-Seed/Depth-Anything-3 and https://arxiv.org/abs/2511.10647
- https://github.com/apple/ml-depth-pro and https://arxiv.org/pdf/2410.02073
- https://github.com/lpiccinelli-eth/UniDepth and https://arxiv.org/abs/2502.20110
- https://github.com/YvanYin/Metric3D
- transformers docs: https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/depth_anything_v2.md
- ONNX export, if ever wanted: https://github.com/fabio-sim/Depth-Anything-ONNX

Licence action item: the Apache-2.0 / CC-BY-NC split follows the encoder size. Confirm
the exact licence on the specific checkpoint before mirroring it to
`Addax-Data-Science/<model_id>` on HuggingFace. CC-BY-NC permits redistribution but
forbids commercial use, and AddaxAI's users include commercial consultancies, so
anything CC-BY-NC is out.

---

## 7. The core architectural decision

**Run the depth model once per deployment on one animal-free frame. Never per image.**

The reasoning:

You are not measuring the depth of the animal. You are measuring the depth of the ground
at the point where the animal is standing. That ground does not move. So one calibrated
depth map of the static scene answers every detection in that deployment, forever, with
a lookup at the bottom edge of the bounding box.

What this buys, in order of importance:

1. **Night works.** Depth models are poor on monochrome IR, and most camera trap animal
   detections happen at night. Build the reference from a daylight blank and the night
   detections read their distance off daylight geometry. The pixel at (u, v) is 7.2 m
   away whether it is noon or 3 a.m. Every per-image pipeline in the literature has this
   problem; this design does not. This is the single biggest practical advantage.

2. **The scale/shift ambiguity disappears.** There is only ever one depth map, so there
   is nothing to align.

3. **Compute is free.** One inference per deployment instead of one per image. On a
   million-image library that is the difference between free and thirty-plus GPU hours.
   It also means a bigger, slower model is affordable if it ever proves better.

4. **Vegetation growth mostly does not matter.** Grass growing changes what a depth model
   sees, so a per-image pipeline moves with it. A fixed reference map keeps stating the
   ground geometry, which is still true.

5. **Video needs nothing special.** A lookup needs no pixels, so the best-frame-only rule
   does not apply. Detections on every sampled frame get a distance, which is the raw
   material for `individualSpeed` later.

6. **Retroactive.** Existing projects become calibratable with no re-analysis. Given how
   much data users already have loaded, this is a large practical win.

The cost, stated honestly: the reference map is only valid while the camera holds still.
Tree sway of a few pixels matters near the horizon and not near the camera, which is
exactly why the literature right-truncates at 15 m anyway. Snow changes ground level a
little. A camera that has been re-aimed is a new deployment by every field protocol, and
AddaxAI already supports deployment splitting.

Worth knowing: a flat-ground pinhole fit with no neural network at all (Leorna and
Brinkman) would do nearly as well on a flat open trail, in about fifty lines of numpy.
The depth model earns its place on slopes, in undergrowth, and where the trail bends.
That is most real sites, so use it, but the cheaper option exists and is a legitimate
fallback if the model proves troublesome.

### 7.1 The maths

Calibration. The model gives inverse depth `d` with `d = a/Z + b` for unknown `a`, `b`.
Given N user points with known metric distance `Z_i` at pixel `(u_i, v_i)` where the
model reads `d_i`, fit the straight line `d_i = a * (1/Z_i) + b` by least squares.
Then `Z = a / (d - b)`.

- Two points is the algebraic minimum. Three or more is the recommendation (matches
  Aamir et al.).
- Residuals per point make a mis-click or a mistyped distance visible.
- Start with affine. Only add piecewise or polynomial regression if measured RMSE demands
  it (Aamir et al. found up to 35% improvement, but that was over per-image depth from
  several different models).

Sampling policy. Median of the depth values in a strip covering the bottom 10% of the box
height and the middle 50% of its width. Median, not mean.

Angle, optional. With a horizontal field of view `H` and image width `W`,
`f_px = (W/2) / tan(H/2)`, and `angle = atan((u_c - W/2) / f_px)` in degrees, where `u_c`
is the bbox centre x. Feeds `individualPositionAngle`.

Height, optional. `height_m ≈ (bbox_height_px / f_px) * distance_m`. Needs the same FOV.

### 7.2 Why not per image

Because per image needs: N inferences instead of 1, a per-image robust alignment step
against a reference to defeat the scale/shift ambiguity, a depth model that behaves on
night IR, and re-analysis of existing projects. Four costs for one benefit, which is
tolerance to camera movement. And camera movement is better handled by detecting it and
telling the user, or by splitting the deployment.

---

## 8. Answers to the ten questions

**1. Automatic or calibration points?** Both, but calibration is what makes the number
publishable. Compute a relative depth map automatically; require the user to supply known
distances before any metric number leaves the app.

**2. Relative or metres?** Metres, once calibrated. Strong recommendation:
`distance_m` stays NULL for an uncalibrated deployment and the export column stays empty.
Writing an uncalibrated model guess into `individualPositionRadius` puts a fabricated
number into a standards-compliant archive, and someone will publish it. Show an
indicative figure in the UI if you like, labelled as uncalibrated, but do not export it.
One rule, easy to say, defensible.

**3. Animal size?** Possible, unreliable, ship it late or not at all. It needs the camera
FOV, which is not in EXIF and which the 2025 benchmark found deviates from the spec sheet
anyway. Errors stack: about 6% from distance at 8 m, 5 to 15% from the box itself (ears,
tail, legs, posture), plus pose. A 50 cm fox lands somewhere between 40 and 62 cm.
Careful manual photogrammetry reaches ±5 cm; automated box-based measurement will not
come close. If built: optional, requires the user to type the FOV, labelled indicative,
and probably only meaningful aggregated over many detections of one species.

**4. Where to measure from?** Bottom of the box, as a median over a strip, not the centre
and not the closest point.

- Centre of bbox: in a background-only reference map this samples the treeline behind the
  animal, which on an open site is badly wrong. In a per-image map it samples the torso,
  which is fine for body depth but gives the wrong ground position.
- Single bottom-centre pixel: can land on a shadow, a leaf, or the gap between the legs.
- Closest of several points: biases toward a twig in the foreground. Avoid.
- Average over the box: contaminated by background. Aamir et al. specifically recommend
  the median, and over a mask rather than a box, for exactly this reason. We have no
  masks, and adding SAM2 is a second heavyweight model, so the bottom strip is the
  equivalent trick without the model.

Two guards to build on day one:
- If `bbox_y + bbox_height > 0.995` the feet are cut off by the frame edge. Write NULL,
  not a confident wrong number.
- Document that an animal in a tree, a bird in flight, or an animal mid-leap has no valid
  ground contact point.

**5. Offline?** Yes, completely. One weights file downloaded once through the existing
`ModelStorage` / `hf_downloader` machinery (including the HuggingFace relay fallback for
blocked networks), then pure local torch. No network at inference.

**6. Separate table or columns?** One column on `detections`: `distance_m`, float,
nullable. Not a separate table. `detection_embeddings` is correctly separate because it
holds multi-KB blobs, potentially several per detection, deleted and rebuilt on their own
schedule. A distance is one float, strictly one per detection, so a separate table would
be a join for nothing. Plus one JSON column on `deployments` for the calibration, and one
file on disk for the depth map. See section 9.

**7. Calibrate once per deployment?** Yes. Deployment, not site: re-hanging the camera
changes the mapping, and a site outlives many deployments. TRAPPER reached the same
conclusion for the same reason.

**8. Run over all images?** No. Once per deployment.

**9. Vegetation growth and wind?** Growth is largely harmless because we read ground
geometry, not appearance. Wind sway is the real limit and is shared with the manual pole
method, so we are no worse than the accepted standard. Mitigations, in order of value:
document the ~15 m truncation the literature already applies; detect gross camera shift
by phase correlation between the reference frame and a late blank frame (about twenty
lines of numpy) and warn; treat a re-aimed camera as a new deployment.

**10. UX / UI / KISS / DRY / YAGNI?** See sections 9 and 10.

---

## 9. Recommended design

### 9.1 Not a pipeline phase

Calibration is a human step that happens after the images are in, so a phase that runs
before the human acts runs for nothing. Make it a per-deployment action, like folder
relinking, triggered from the deployment surface. This removes a phase from an already
eight-phase worker, needs no project-level model setting, and means no re-analysis for
existing data.

### 9.2 Model plumbing

Add the depth model to `models.json` under a new `dep` key so it reuses `ModelStorage`,
`catalog_updater`, staleness checks, the relay download path and the download-progress
UI. That is a lot of hard-won machinery for free.

Do **not** add a model picker. One depth model, no choice. Users cannot evaluate a depth
model and should not be asked to. This is a deliberate departure from the det/cls/emb
pattern; write the reason in a comment so nobody "fixes" it later.

Manifest fields needed beyond the common ones: probably just `input_size`. Possibly a
`hf_repo` override.

### 9.3 Environment

Recommendation: add `transformers` to `env-addaxai-base` and run depth there, alongside
DINOv2 embeddings. Reason: a detector-only project should not have to build the ~5 GB
`env-pytorch` just to calibrate a deployment.

Cost: changing the base env yaml forces one env rebuild for every user on update. The
drift machinery already handles that (`find_drifted_envs`, `DEVELOPERS.md` "Environment
drift is answered per request").

Alternative considered and rejected: vendor ~400 lines of DPT head plus DINOv2 encoder
into `backend/app/ml/inference/` to avoid the dependency. Less dependency surface, more
code we own and must maintain against upstream. Take the dependency.

### 9.4 Storage

| What | Where | Why |
|---|---|---|
| `distance_m` float, nullable | new column on `detections` | strictly 1:1 with a detection, one float |
| calibration: reference file id, list of `{x, y, distance_m}` points, fitted `a`, `b`, RMSE, model id, fitted-at timestamp | new JSON column on `deployments`, e.g. `distance_calibration` | follows `tags` / `camera_offsets` / `warnings` |
| optional `camera_fov_degrees` | `deployments` | explicit, no default, NULL means angle and height are skipped |
| raw relative depth map, float16 npz, ~300 KB | `{deployment}/.addaxai/depth_reference.npz` | regenerable, keeps recalibration instant, next to `video_frames/` |

Store the raw relative map plus `(a, b)` rather than a pre-calibrated metric map, so
changing the calibration is a refit and not a re-inference.

One migration covers both columns.

### 9.5 Backend modules

Mirroring the embedding feature:

- `backend/app/ml/inference/depth_script.py`: subprocess, loads the model, writes an npz.
- `backend/app/ml/inference/depth_model.py`: wrapper, stderr progress parsing, cancel
  support, same shape as `embedding_model.py`.
- `backend/app/ml/depth_calibration.py`: the fit, the lookup, the sampling policy, the
  guards. Pure numpy, unit-testable in isolation, no model needed. This is the module the
  tests live against.
- `backend/app/workers/depth_worker.py`: build reference map, fit, write `distance_m`
  across the deployment, report progress over the existing websocket protocol.
- Router: `GET/PUT /api/deployments/{id}/calibration`, plus a job trigger.

### 9.6 Exports

- Detections CSV/XLSX: one new column `distance_m`. Document it in
  `docs/docs/reference/exports.md`.
- Camtrap DP observations: fill `individualPositionRadius`, and
  `individualPositionAngle` when `camera_fov_degrees` is set. Headers already exist.
- Folder runs: folder runs have one synthetic deployment and no site. Calibration could
  work there too, but it is not the use case. Decide explicitly; leaning toward projects
  only in v1, with the column trimmed by `_table_columns.OMITTED_COLUMNS` if so.

### 9.7 UI, in order of value

1. **Pick the reference image.** Auto-select a daylight blank near the middle of the
   deployment, which you can do already because `File.observation_type == "blank"` and
   `captured_at_local` are both indexed. Let the user page to another, for instance the
   setup photo where they walked the trail with a pole.
2. **Click points.** Reuse `AnnotationCanvas.tsx` for zoom and pan. Click a spot on the
   ground, type metres, repeat three to five times, spread near to far. Show a residual
   per point so an outlier is obvious, and allow deleting a point.
3. **Isodistance contours** at 5, 10 and 15 m drawn over the reference photo. This is the
   single most valuable element in the whole feature. It turns a black box into something
   a field ecologist can check at a glance, and it is what will make them trust or reject
   the number. Do not skip it.
4. **Badge on the deployment**: "Calibrated, 4 points, RMSE 0.31 m", or "Not calibrated".
5. Distance shown quietly on the detection detail in the Labels view.
6. Later: an Insights page, `/projects/:id/insights/distance`, with a distance histogram
   per species. That is what CTDS users want to see. Not v1; the export is enough to
   start.

### 9.8 Where the known distances come from

Three realistic sources, all served by the same UI:

- A calibration photo shot at setup: someone stands at 3, 6 and 10 m, one photo each, or
  one photo with markers. Best practice, needs a field protocol written into the docs.
- Retrofit: the user recognises features in the scene and measures them with a tape or
  rangefinder on the next site visit.
- Nothing: the deployment stays uncalibrated and exports nothing.

A fourth, for v2 at the earliest: derive a distance from a known-size animal already in
the frame ("that is a red deer, shoulder height about 1.2 m"). Crude, but it unlocks
archives where no site visit is possible.

---

## 10. What to deliberately not build

- Per-image depth inference.
- SAM2 masks. A second heavyweight model to replace a median over a strip.
- Video speed estimation, until distance is trusted. The data will already be there.
- Multiple reference frames per deployment, or seasonal recalibration.
- Polynomial or piecewise regression, before affine has been measured as insufficient.
- A truncation setting. Analysts truncate in R.
- Copying a calibration between deployments. The depth map is scene-specific.
- A depth model picker.
- A project-level "depth model" setting mirroring `embedding_model_id`.

Borderline, worth v1.5 not v1: camera-shift detection by phase correlation between the
reference and a late blank frame. About twenty lines of numpy, and it catches the one
failure mode that actually breaks this design.

---

## 11. Risks and honest limits

| Risk | Severity | Mitigation |
|---|---|---|
| Camera physically moves (wind, animal knock, re-aim) | High, the core assumption | Phase-correlation shift check; treat re-aim as a new deployment; truncate far distances |
| Animal not on the ground (tree, flight, leap) | Medium | Document. Per-species exclusion is YAGNI |
| Bbox clipped at the frame bottom | Medium, common | Write NULL when `bbox_y + bbox_height > 0.995` |
| Extrapolation beyond the calibration range | Medium | Flag it; consider capping at 1.5x the furthest calibration point |
| User calibration errors | Medium | Show per-point residuals and RMSE; allow deletion |
| Snow or seasonal ground level change | Low | Accept, document |
| Wide-angle barrel distortion at frame edges | Low | Accept |
| Licence of the specific checkpoint | Blocking if wrong | Verify before mirroring to the Addax HF org |
| Expectation management on size | Medium, reputational | Ship distance first, label size as indicative or omit it |

Accuracy to expect, stated plainly: roughly 0.45 m MAE over 0 to 15 m, per the wildlife
benchmark, for a calibrated Depth Anything V2. That is comparable to the manual ranging
pole method, which is itself somewhere around 0.2 to 0.5 m, at a tiny fraction of the
effort. That comparison is the pitch.

---

## 12. Effort estimate

Backend, roughly a week:

- `dep` catalog plumbing mirroring `emb`: half a day
- `depth_script.py` + `depth_model.py`: one day
- `depth_calibration.py` plus tests: one day
- Alembic migration: one hour
- Router, job, worker: one day
- Exports, one column plus two Camtrap DP fields: half a day

Frontend, two to three days, nearly all of it the calibration dialog.

Docs, half a day: a new page under `docs/docs/understanding/`, plus an exports page
update and a field protocol for shooting calibration photos.

Total, about a fortnight for a v1.

## 13. Step zero, before any of that

Spend one day on a throwaway script. Point it at a deployment folder, a reference image,
and three hand-measured `(pixel, metre)` pairs. Have it print distances for every
detection. Then check it against a site where you have real tape or rangefinder
measurements, and specifically check a night detection against a daylight reference map.

Everything above is conditional on that spike coming back at roughly half a metre. If it
does not, the fallback is the flat-ground pinhole fit from Leorna and Brinkman, which
needs no model at all.

---

## 14. Reproducibility notes

### 14.1 What was audited

Read in full or in part: `CONVENTIONS.md`, `README.md`, `DEVELOPERS.md` (table of
contents plus the Insights, best-frame, media-copies and custom-model sections),
`models.json`, `backend/app/models/{detection,file,deployment,site,project,detection_embedding}.py`,
`backend/app/ml/{embedding_utils,manifest_manager,model_storage,catalog_updater}.py`,
`backend/app/ml/schemas/model_manifest.py`,
`backend/app/ml/inference/{base,embedding_model}.py`,
`backend/app/workers/{embedding_worker,detection_worker,camtrap_export_worker}.py`,
`backend/app/services/crop_service.py`, `backend/app/api/crud/export.py` (headers),
`backend/app/ml/postprocessing_outputs/_table_columns.py`,
`docs/docs/reference/exports.md`, `docs/sidebars.ts`, `frontend/src/App.tsx`,
`frontend/src/components/verify/AnnotationCanvas.tsx` (header),
`frontend/src/components/settings/AnalysisSettingsRows.tsx`.

Grepped: `depth` across all source and docs (nothing relevant),
`FocalLength|focal|FOV` (nothing), `individualPosition*` in the export code.

### 14.2 Egress limits during the investigation

The session's proxy blocked `arxiv.org`, `sciencedirect.com`, `huggingface.co`,
`biorxiv.org`, `papers.ssrn.com`, `readthedocs.io`, `awesomepapers.io` and
`stangandaho.github.io`. `github.com` and `raw.githubusercontent.com` were reachable.

Therefore: everything sourced from GitHub was fetched and read directly. Everything from
the journals, arXiv, HuggingFace and the TRAPPER docs came from web-search summaries, not
from the primary text. **Before implementation, read the two 2025/2026 papers and the
TRAPPER docs in full.** The numbers quoted here (0.454 m MAE, r 0.962, 0.22 s/image, the
factor of 21, the 35% piecewise improvement, the three-point minimum) are second-hand and
should be verified against the papers.

### 14.3 Search queries that produced the useful hits

- `camera trap distance sampling monocular depth estimation animal distance`
- `"From relative to metric" calibrating monocular depth distance sampling wildlife 2026 Ecological Informatics`
- `"robust metric distance and height estimation pipeline" wildlife camera trap imagery 2025 SAM MegaDetector depth calibration median`
- `Haucke Steinhage automated distance estimation wildlife camera trapping reference image calibration bottom bounding box`
- `Henrich 2024 semi-automated camera trap distance sampling reference images calibration pole distances 2.5m intervals workflow`
- `TRAPPER project distance estimation AI pipeline UniDepthV2 reference points calibration deployment documentation`
- `monocular metric depth estimation 2026 state of the art Depth Anything V3`
- `relative depth affine invariant scale shift ambiguity per image inconsistent across frames static scene alignment`
- `number of calibration points needed relative to metric depth conversion camera trap linear piecewise regression accuracy comparison`
- `camera trap animal body size shoulder height estimation from images depth model accuracy`
- `monocular depth estimation infrared night camera trap images grayscale performance degradation`

### 14.4 Open items to settle before building

1. Confirm the licence of the exact Depth Anything V2 checkpoint to be mirrored.
2. Read the two 2025/2026 papers in full and verify the quoted numbers.
3. Read the TRAPPER distance estimation docs in full, as the closest shipped prior art.
4. Decide: relative model requiring calibration, or metric model with calibration as a
   correction. This document recommends the former. The argument for the latter is that
   uncalibrated deployments still get a number, but section 8.2 argues against exporting
   those numbers at all, which removes most of the benefit.
5. Decide whether folder runs get calibration at all.
6. Decide `transformers` in `env-addaxai-base` versus vendoring the model code.
7. Run the step-zero spike on real data, including a night check.

---

## 15. Plain English summary

Depth estimation for camera traps is a solved-enough problem, and Depth Anything V2 Small
is still the right model: 100 MB, Apache licensed, and measured as the best on camera
trap imagery once you calibrate it. The trick that makes it cheap for AddaxAI is to stop
thinking about the animal and start thinking about the ground. The background is static,
so run the model once per deployment on one empty frame, have the user click three or four
points where they measured a real distance, and after that every animal's distance is just
a lookup at the bottom edge of its bounding box. That means one model run per camera
instead of one per photo, it works for night photos even though depth models are bad at
infrared, it works for video with no extra machinery, and it works retroactively on data
already loaded without reprocessing anything. The result is one new column on the
detections table, one JSON blob on the deployment, and two Camtrap DP fields that already
have headers but are never filled. Distance is the real deliverable and should land around
half a metre of error. Animal size is derivable from the same numbers but will be wrong by
15 to 25 percent per detection, so either leave it out or label it clearly as indicative.
Before building anything, spend a day on a throwaway script and check it against real tape
measurements at a site you know.
