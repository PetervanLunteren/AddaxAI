"""
Best-frame selection for video files.

Two code paths can produce a best frame per video:

1. **Fused with classification.** The classifier worker already iterates
   the source video to crop and classify each animal detection, so it
   picks the winning frame in the same pass and writes the JPEG itself.
   This is the common case and lives in
   `app/ml/inference/classification_worker.py`.

2. **No classifier configured.** This module covers that case: it opens
   each video itself, decides the frame from the detection JSON alone,
   fetches exactly that frame, and writes the JPEG.

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
Fetching switched from a sequential walk to a verified seek 2026-08-03.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from app.core.job_cancellation import JobCancelledError, is_cancel_requested
from app.core.logging_config import get_logger
from app.core.media_types import VIDEO_EXTENSIONS
from app.ml.inference.scoring import choose_frame_number
from app.ml.inference.video_iter import (
    iter_wanted_frames,
    open_video,
    read_frame_by_seek,
    write_best_frame,
)

logger = get_logger(__name__)


def _fetch_best_frame(
    video_path: Path, detections: list[dict]
) -> tuple[int, Image.Image] | None:
    """
    Return `(best_frame_number, pixels)` for one video, or None if
    nothing decoded (`open_video` logs why).

    The frame is decided from the JSON before any pixels are touched, so
    we can go straight to it. Seeking is 3x cheaper than walking there,
    and it matters most for the commonest case: a video with no confident
    detection is sent to its middle frame, so walking meant decoding half
    of every empty clip and discarding it.

    A seek that cannot be verified falls back to the sequential walk,
    which also collects frame 0 as insurance. If the chosen frame never
    arrives (containers over-report their frame count) we return frame 0
    instead **and move the number with the pixels**. Stamping the frame
    we wanted while returning the frame we got is what makes the Labels
    grid draw one moment's boxes over another moment's picture.
    """
    cap = open_video(video_path)
    if cap is None:
        return None
    try:
        total_frames = int(cap.get(_import_cv2().CAP_PROP_FRAME_COUNT))
        best_frame_number = choose_frame_number(detections, total_frames)
        pixels = read_frame_by_seek(cap, best_frame_number, total_frames)
    finally:
        cap.release()

    if pixels is not None:
        return best_frame_number, pixels

    # A refused seek leaves the capture somewhere we cannot reason
    # about, so the walk gets its own.
    cap = open_video(video_path)
    if cap is None:
        return None
    chosen_pixels = None
    first_pixels = None
    try:
        for frame_num, pil_image in iter_wanted_frames(
            cap, {best_frame_number, 0}
        ):
            if frame_num == best_frame_number:
                chosen_pixels = pil_image
            if frame_num == 0:
                first_pixels = pil_image
    finally:
        cap.release()

    if chosen_pixels is not None:
        return best_frame_number, chosen_pixels
    if first_pixels is not None:
        return 0, first_pixels
    return None


def select_best_frames_streaming(
    json_path: Path,
    deployment_folder: Path,
    output_base: Path,
    progress_callback: Callable[[int, int], None] | None = None,
    job_id: str | None = None,
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

    `progress_callback(done, total)` is called once before the first
    video and once per video after it. On a folder of thousands this is
    a phase of its own, long enough that the user needs to see it move.

    Videos that can't be opened or have zero decodable frames are
    skipped with a warning. The overall sweep is best-effort: a failure
    on one video never stops the others. A cancel does stop it, by
    raising `JobCancelledError` before the JSON is rewritten, so a
    cancelled run leaves no half-updated file behind.
    """
    with open(json_path) as f:
        data = json.load(f)

    videos: list[tuple[dict, Path]] = []
    for img_entry in data.get("images") or []:
        if img_entry.get("failure"):
            continue
        absolute = (deployment_folder / img_entry["file"]).resolve()
        if absolute.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append((img_entry, absolute))

    total = len(videos)
    if progress_callback:
        progress_callback(0, total)

    for done, (img_entry, absolute) in enumerate(videos, start=1):
        if job_id and is_cancel_requested(job_id):
            raise JobCancelledError()

        result = _fetch_best_frame(absolute, img_entry.get("detections") or [])
        if result is None:
            logger.warning(
                f"select_best_frames_streaming: no decodable frames for "
                f"{absolute}, skipping"
            )
        else:
            best_frame_number, chosen_pixels = result
            img_entry["best_frame_number"] = best_frame_number

            # Write the chosen JPEG using the same filename scheme legacy
            # data uses, so `best_frame_path` math stays unchanged.
            relative_video_path = absolute.relative_to(deployment_folder)
            dest = (
                output_base
                / relative_video_path
                / f"frame{best_frame_number:06d}.jpg"
            )
            try:
                write_best_frame(chosen_pixels, dest)
            except Exception as e:
                logger.warning(
                    f"select_best_frames_streaming: failed to write best frame "
                    f"for {absolute}: {e}"
                )

        if progress_callback:
            progress_callback(done, total)

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


def _import_cv2():
    """Lazy cv2 import so unrelated tests don't pay the cost."""
    import cv2  # type: ignore  # noqa: PLC0415

    return cv2
