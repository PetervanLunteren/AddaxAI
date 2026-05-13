"""
Best-frame selection for video files.

Two code paths can produce a best frame per video:

1. **Fused with classification.** The classifier worker already iterates
   the source video to crop and classify each animal detection, so it
   scores sharpness in the same pass and writes the winning JPEG itself.
   This is the common case and lives in
   `app/ml/inference/classification_worker.py`.

2. **No classifier configured.** This module covers that case: it opens
   each video itself, sharpness-scores the frames that carry detections
   (or evenly-spaced samples for blank videos), picks the winner with
   the existing detection-confidence scorer, and writes the JPEG.

In both paths the picker is `scoring.pick_best_candidate` with the same
tier rules, so a classifier-on vs classifier-off run produces consistent
results.

Created 2026-02-13. Rewritten 2026-05-13 to drop bulk frame extraction.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.core.logging_config import get_logger
from app.core.media_types import VIDEO_EXTENSIONS
from app.ml.inference.scoring import compute_sharpness, pick_best_candidate, score_detections
from app.ml.inference.video_iter import (
    iter_wanted_frames,
    open_video,
    pil_to_rgb_array,
    sample_indices,
    write_best_frame,
)

logger = get_logger(__name__)

# Same constant the classifier worker uses for blank-video fallbacks.
BLANK_VIDEO_SAMPLE_COUNT = 3


def select_best_frames_streaming(
    json_path: Path,
    deployment_folder: Path,
    output_base: Path,
) -> None:
    """
    Pick the best frame for every video listed in the detection JSON and
    write its JPEG, by reading the source videos directly. Used by both
    the deployment worker and the Timelapse runner when no classifier is
    configured (or when classification was skipped).

    Updates the JSON file in-place with `best_frame_number` on each
    non-failed video entry. Writes one JPEG per video at
    `output_base/<relative_video_path>/frame{best:06d}.jpg`, matching
    the path scheme `_load_to_database` expects on `best_frame_path`.

    Videos that can't be opened or have zero decodable frames are
    skipped with a warning. The overall sweep is best-effort: a failure
    on one video never stops the others.
    """
    cv2 = _import_cv2()

    with open(json_path) as f:
        data = json.load(f)

    images = data.get("images") or []

    for img_entry in images:
        if img_entry.get("failure"):
            continue

        relative_file = img_entry["file"]
        absolute = (deployment_folder / relative_file).resolve()
        if absolute.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        detections = img_entry.get("detections") or []
        wanted: set[int] = {
            int(d["frame_number"])
            for d in detections
            if d.get("frame_number") is not None
        }

        cap = open_video(absolute)
        if cap is None:
            logger.warning(
                f"select_best_frames_streaming: could not open {absolute}, "
                "skipping"
            )
            continue

        try:
            # Blank video fallback: sample N evenly-spaced frames so the
            # sharpness scorer has something to work with.
            if not wanted:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                wanted = set(sample_indices(total_frames, BLANK_VIDEO_SAMPLE_COUNT))
                if not wanted:
                    logger.warning(
                        f"select_best_frames_streaming: {absolute} has 0 "
                        "frames according to cv2, skipping"
                    )
                    continue

            sharpness_by_frame: dict[int, float] = {}
            pixels_by_frame: dict[int, object] = {}
            for frame_num, pil_image in iter_wanted_frames(cap, wanted):
                sharpness_by_frame[frame_num] = compute_sharpness(
                    pil_to_rgb_array(pil_image)
                )
                pixels_by_frame[frame_num] = pil_image
        finally:
            cap.release()

        if not sharpness_by_frame:
            logger.warning(
                f"select_best_frames_streaming: no decodable frames for "
                f"{absolute}, skipping"
            )
            continue

        det_tuples = [
            (
                str(int(d["frame_number"])),
                float(d.get("conf", 0.0)),
                tuple(d["bbox"]),
            )
            for d in detections
            if d.get("frame_number") is not None
        ]
        frame_scores = score_detections(det_tuples)

        def get_sharpest(keys: list[str], _sb=sharpness_by_frame) -> str:
            return str(max((int(k) for k in keys), key=lambda fn: _sb.get(fn, 0.0)))

        fallback_keys = [str(fn) for fn in sorted(sharpness_by_frame)]
        best_key = pick_best_candidate(
            frame_scores,
            get_sharpest=get_sharpest,
            fallback_keys=fallback_keys,
        )
        if best_key is None:
            continue

        best_frame_number = int(best_key)
        img_entry["best_frame_number"] = best_frame_number

        # Write the chosen JPEG using the same filename scheme legacy
        # data uses, so `best_frame_path` math stays unchanged.
        relative_video_path = absolute.relative_to(deployment_folder)
        dest = output_base / relative_video_path / f"frame{best_frame_number:06d}.jpg"
        try:
            write_best_frame(pixels_by_frame[best_frame_number], dest)
        except Exception as e:
            logger.warning(
                f"select_best_frames_streaming: failed to write best frame "
                f"for {absolute}: {e}"
            )

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


def _import_cv2():
    """Lazy cv2 import so unrelated tests don't pay the cost."""
    import cv2  # type: ignore  # noqa: PLC0415

    return cv2
