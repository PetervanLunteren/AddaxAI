"""
Streaming video frame iteration helpers.

Used by `classification_worker.py` (subprocess, no app.* on sys.path) and
by parent-process best-frame scoring fallbacks. Self-contained: only
cv2, numpy, and PIL are imported.

`open_video` mirrors MegaDetector's video_utils backend-fallback order
(default -> FFMPEG -> AVFoundation -> DShow -> MSMF -> GStreamer) so we
inherit the same codec coverage `process_video` has in production.
`iter_wanted_frames` reads sequentially with `cap.read()`; we never
random-seek via `cap.set(POS_FRAMES, N)` because that's unreliable on
many codecs.

Created by Claude Code on 2026-05-13
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_BACKEND = -1
_BACKENDS = (
    DEFAULT_BACKEND,
    cv2.CAP_FFMPEG,
    cv2.CAP_AVFOUNDATION,
    cv2.CAP_DSHOW,
    cv2.CAP_MSMF,
    cv2.CAP_GSTREAMER,
)


def open_video(path: Path | str) -> cv2.VideoCapture | None:
    """
    Open a video with cv2 backend fallbacks. Returns a fresh VideoCapture
    positioned at frame 0, or None if every backend fails to even decode
    frame 0.

    Caller owns release().
    """
    p = str(path)
    if not os.path.isfile(p):
        logger.error(f"Video file not found: {p}")
        return None

    for backend_id in _BACKENDS:
        try:
            cap = (
                cv2.VideoCapture(p)
                if backend_id == DEFAULT_BACKEND
                else cv2.VideoCapture(p, backend_id)
            )
        except Exception as e:
            logger.debug(f"Backend {backend_id} raised on {p}: {e}")
            continue

        if not cap.isOpened():
            cap.release()
            continue

        # Probe one frame to be sure the codec actually decodes. Some
        # backends report isOpened()=True but fail on the first read.
        success, _ = cap.read()
        cap.release()
        if not success:
            continue

        # Re-open so the caller iterates from frame 0.
        return (
            cv2.VideoCapture(p)
            if backend_id == DEFAULT_BACKEND
            else cv2.VideoCapture(p, backend_id)
        )

    logger.error(f"Failed to open {p} with any cv2 backend")
    return None


def iter_wanted_frames(
    cap: cv2.VideoCapture, wanted: set[int]
) -> Iterator[tuple[int, Image.Image]]:
    """
    Yield `(frame_number, PIL.Image)` for each frame index in `wanted`,
    reading the video sequentially from the start. Stops at end-of-stream
    even if `wanted` references higher indices (some videos report a
    higher CAP_PROP_FRAME_COUNT than they can actually decode).

    Empty `wanted` yields nothing without touching the capture.
    """
    if not wanted:
        return

    last_wanted = max(wanted)
    frame_number = 0

    while True:
        success, image_bgr = cap.read()
        if not success or image_bgr is None:
            break

        if frame_number in wanted:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            yield frame_number, Image.fromarray(image_rgb)

        if frame_number >= last_wanted:
            break

        frame_number += 1


def sample_indices(total: int, count: int) -> list[int]:
    """
    Evenly-spaced frame indices from [0, total). Used for blank-video
    sharpness fallback when there are no detections to anchor on.
    Returns at most `count` indices; clamps to `total` when smaller.
    """
    if total <= 0 or count <= 0:
        return []
    if total <= count:
        return list(range(total))
    step = total / count
    return [int(i * step) for i in range(count)]


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    """PIL image (any mode) -> RGB numpy array. Used to feed
    `scoring.compute_sharpness`, which expects RGB."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)


def write_best_frame(image: Image.Image, dest: Path, quality: int = 90) -> None:
    """
    Write the best-frame JPEG to `dest`. Creates parent directories. The
    quality default matches the legacy `extract_frames_from_video --quality 80`
    bumped a notch since we now save exactly one JPEG per video.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(str(dest), "JPEG", quality=quality)
