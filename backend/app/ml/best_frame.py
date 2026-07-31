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

Both paths score **every detection, whatever its category**, on the
detector's own confidence, and hand the result to
`scoring.pick_best_candidate` with the same tier rules. So a
classifier-on and a classifier-off run pick the same frame for the same
video.

That was not true until 2026-07-31: path 1 scored only the animals it was
about to classify, so a clip containing only people scored nothing and
fell back to sharpness over three arbitrary frames, while path 2 scored
the people. Detection confidence is the one signal every detector emits,
which is why it is the primary score: no category vocabulary is assumed,
so a detector emitting `fish` / `shark` / `turtle` needs no change here.

Created 2026-02-13. Rewritten 2026-05-13 to drop bulk frame extraction.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.core.logging_config import get_logger
from app.core.media_types import VIDEO_EXTENSIONS
from app.ml.inference.scoring import choose_frame_number
from app.ml.inference.video_iter import (
    iter_wanted_frames,
    open_video,
    write_best_frame,
)

logger = get_logger(__name__)


def select_best_frames_streaming(
    json_path: Path,
    deployment_folder: Path,
    output_base: Path,
) -> None:
    """
    Pick the best frame for every video listed in the detection JSON and
    write its JPEG, by reading the source videos directly. Used by the
    deployment worker when no classifier is configured (or when
    classification was skipped).

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

        cap = open_video(absolute)
        if cap is None:
            logger.warning(
                f"select_best_frames_streaming: could not open {absolute}, "
                "skipping"
            )
            continue

        try:
            # Decided from the JSON alone, before any frame is decoded,
            # so we decode only the frame we are going to keep. Frame 0
            # comes along as insurance: a container can advertise more
            # frames than it yields, and the chosen one may never arrive.
            best_frame_number = choose_frame_number(
                detections, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            )
            wanted = {best_frame_number, 0}

            chosen_pixels = None
            first_pixels = None
            for frame_num, pil_image in iter_wanted_frames(cap, wanted):
                if frame_num == best_frame_number:
                    chosen_pixels = pil_image
                if frame_num == 0:
                    first_pixels = pil_image
        finally:
            cap.release()

        if chosen_pixels is None:
            # Move the number with the pixels. The stamped
            # `best_frame_number` and the written JPEG must describe the
            # same moment, or the Labels grid draws one frame's boxes
            # over another frame's picture.
            if first_pixels is None:
                logger.warning(
                    f"select_best_frames_streaming: no decodable frames for "
                    f"{absolute}, skipping"
                )
                continue
            best_frame_number, chosen_pixels = 0, first_pixels

        img_entry["best_frame_number"] = best_frame_number

        # Write the chosen JPEG using the same filename scheme legacy
        # data uses, so `best_frame_path` math stays unchanged.
        relative_video_path = absolute.relative_to(deployment_folder)
        dest = output_base / relative_video_path / f"frame{best_frame_number:06d}.jpg"
        try:
            write_best_frame(chosen_pixels, dest)
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
