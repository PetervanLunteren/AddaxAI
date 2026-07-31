"""
Streaming video frame iteration helpers.

Used by `classification_worker.py` (subprocess, no app.* on sys.path) and
by parent-process best-frame scoring fallbacks. Self-contained: only
cv2 and PIL are imported.

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
    Evenly-spaced frame indices from [0, total). Used by the filmstrip
    to build a temporal preview of a clip.
    Returns at most `count` indices; clamps to `total` when smaller.
    """
    if total <= 0 or count <= 0:
        return []
    if total <= count:
        return list(range(total))
    step = total / count
    return [int(i * step) for i in range(count)]


def write_best_frame(
    image: Image.Image,
    dest: Path,
    quality: int = 80,
    max_dim: int = 1920,
) -> None:
    """
    Write the best-frame JPEG to `dest`. Creates parent directories.

    `quality=80` is the visually-indistinguishable sweet spot for JPEG
    at thumbnail and modal sizes. `max_dim` caps the longest side; 4K
    cameras get downsampled here because nothing in the UI consumes
    above 1920 px (modal renders ≤1920, thumbnails ≤320, crop service
    outputs ≤512). Detection bboxes are normalised so the cap doesn't
    affect overlay positioning. `optimize=True` and progressive encoding
    cost negligible CPU at write time and shave a further ~5-10%.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if image.mode != "RGB":
        image = image.convert("RGB")
    if max(image.size) > max_dim:
        # Copy so we never mutate a frame the caller might still hold;
        # `thumbnail` operates in-place and shrinks only (never upscales).
        image = image.copy()
        image.thumbnail((max_dim, max_dim), Image.LANCZOS)
    image.save(
        str(dest),
        "JPEG",
        quality=quality,
        optimize=True,
        progressive=True,
    )
