# Marine BRUVS video mode

Plan for a future change. Not scheduled. Captured here so we can pick it up without re-deriving the design. Marine classification models are not in the app yet, so none of this is built. This note records why the current camera-trap design does not preclude it, and the shape it should take when a marine model lands.

## Why

AddaxAI will add marine models (baited remote underwater video, BRUVS). Marine abundance analysis works differently from camera traps, and our current video handling is camera-trap-shaped in a way that is useless for marine.

Camera-trap clips are short (about 10 s) and usually show one animal, so we represent a video by a single "best frame": the Labels crop grid only shows the best frame's detections, and the count suggestion is gated to best-frame species (see `calculate_max_n_for_event` in `backend/app/api/crud/event_observation.py`).

Marine BRUVS are long (30+ min) baited stations where many species swim in and out. The standard metric is **MaxN**: the maximum number of individuals of a species visible in a single frame, taken across the whole video. The defining property that breaks the best-frame model: **each species reaches its MaxN at a different time**, so a single distilled frame cannot encode 10 species' peaks. Best-frame-only is not "slightly worse" for marine, it is structurally unable to carry the deliverable.

References: MaxN definition and the canonical annotation workflow are in [Langlois et al. 2020, Methods in Ecology and Evolution](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13470) and the [benthic stereo-BRUVs field manual](https://benthic-bruvs-field-manual.github.io/image-annotations). MeanCount as an alternative metric: [Schobernd et al. 2014](https://www.sciencedirect.com/science/article/abs/pii/S0165783615001526). AI-assisted, track-based marine annotation: [VIAME](https://www.viametoolkit.org/) and [Frontiers 2022](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2022.944582/full).

## Two facts that shape the design (easy to get wrong)

1. **Marine annotators do not label every frame.** They watch the whole video and, per species, capture the single frame where that species peaks (the MaxN frame), record the count and the time of MaxN, and annotate individuals only at that frame. So the requirement is "review the whole video and capture each species' peak," not "relabel all 1,800 frames." A crop grid of every frame would be thousands of near-duplicate crops, which is exactly what no marine tool makes you do.

2. **The modern AI-assisted marine workflow is track-based, not per-frame.** Tools like VIAME run per-frame detection plus tracking (linking one fish across frames into a track); the human corrects tracks (one correction per fish), then MaxN is derived. We have no tracking. Without it, "clean all frames" degenerates into per-frame box relabeling, the laborious thing tracks exist to avoid. Tracking is the real enabler and the biggest lift.

## The reassuring part: the core model is already marine-correct

MaxN-across-all-frames is already the shared, true model. The code computes it for both worlds, and the camera-trap best-frame behavior sits on top as one removable filter, not a baked-in assumption.

Already in place (no rework needed):
- Every frame's detections are stored, with `Detection.frame_number` (`backend/app/models/detection.py`). Nothing is dropped at load.
- `calculate_max_n_for_event` already groups by `(file, frame_number, species)` and takes the peak across all frames. The per-species MaxN suggestion already exists.
- The source video path is retained (`File.file_path`), so any frame can be re-decoded later.
- Frame seek/decode logic already exists in `backend/app/ml/inference/video_iter.py`. It is scoped to the analysis subprocess, so it is portable, not missing.
- The frontend `VideoPlayer` is most of a scrubber already: HTML5 seek, detections grouped by `frame_number`, per-frame box overlays.

The single camera-trap-specific fork point:
- The best-frame **species gate** (the `allowed_video_keys` block in `calculate_max_n_for_event`) and the best-frame-only Labels/embedding path (`backend/app/ml/embedding_utils.py`, `backend/app/services/crop_service.py`). These are the only places that assume "video = one frame." The gate is one localized, non-destructive, removable block.

Genuinely missing for marine: frame-on-demand decode in the main app, a project/deployment mode flag, time/frame of MaxN storage, tracking, and the marine review UI.

## Decision (the shape, when marine lands)

A per-project (or per-deployment) **mode fork**, not a unification. Do not marine-ify camera-trap mode; keep it simple.

- **Camera-trap mode** (today): best-frame canonical, best-frame gate on, crop-grid Labels. Unchanged.
- **Marine/BRUVS mode** (future): best-frame gate **off** (all-frame species); Labels becomes a video scrubber (decode frames on demand) rather than a crop grid; MaxN per species across the whole video with time/frame of MaxN and jump-to-peak; its own noise strategy. Note marine reintroduces the per-frame label noise the gate suppresses, so marine needs track-level confidence/confirmation rather than the best-frame gate, which is another reason tracking is the key marine primitive.

### Phased shape (roughly in order)

1. Mode flag (project/deployment); make the best-frame gate conditional on it.
2. Lift `video_iter` decode into an on-demand frame endpoint (serve arbitrary frame N from the source video).
3. Marine video-scrubber review UI, reusing the existing `VideoPlayer` and the frame-grouped overlays.
4. `EventObservation.max_n_frame_number` plus jump-to-peak and time-of-MaxN in exports. Maps cleanly to EventMeasure / GlobalArchive and the Darwin Core `organismQuantity` the observation rebuild already anticipated.
5. Tracking as the quality multiplier (biggest lift). A marine v1 can ship without it: manual MaxN scoring on the scrubber, which is exactly what EventMeasure users do today.

### Open decision for later (not now)

Whether marine v1 is manual MaxN scoring on a scrubber (cheap, matches EventMeasure, no tracking) or AI-assisted track review (expensive, matches VIAME). Recommendation: start manual. It is the proven workflow and avoids the tracking lift. MeanCount can be a later option alongside MaxN; MaxN is the dominant metric, start there.

### Per-frame timestamps (do it here, not in camera-trap mode)

Today every detection in a video inherits the video's single `File.captured_at_local` (the clip start, from exiftool). `Detection` has no time column, only `frame_number` plus `frame_rate` on the File. So a frame's real time is derivable, not stored: `file.captured_at_local + frame_number / frame_rate`.

For camera trap this is not worth doing: clips are ~10 s, so the offset is below the noise floor of events (minute-scale), activity (hourly bins), and trap nights (daily), and emitting per-frame times overstates precision the camera clock does not have.

For marine it matters. A fish at minute 25 is genuinely 23 min after one at minute 2, and the deliverables need it: time of MaxN, time to first arrival, MaxN over time. So fold the per-frame timestamp into phase 4 (time/frame of MaxN): compute it **derived** (file time + `frame_number / frame_rate`), surfaced in the marine review UI and exports. Do not add a per-detection timestamp column: it is fully derivable, and storing it would need re-applying the "Adjust dates" offset and the deployment-timezone shift, which already rewrite file time only. Guard the usual edge cases: missing `frame_rate`, NULL `captured_at_local` stays NULL.

## Guardrails to hold now (mostly already true, just do not regress)

- Keep all per-frame `Detection` rows and `frame_number`. Keep the best-frame gate non-destructive (raw per-frame labels stay).
- Keep the source video path on the File row.
- Keep the best-frame gate localized and conditional-ready (one block), do not scatter best-frame assumptions.
- Do not build anything that assumes the best-frame JPEG is the only recoverable pixels from a video.

## What not to do now (YAGNI)

No mode column, no tracking, no all-frame embedding, no scrub-MaxN UI, no `max_n_frame_number`. None is needed until a marine model ships, and each is real surface area on a camera-trap product that currently works.
