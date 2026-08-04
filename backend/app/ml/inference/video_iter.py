"""
Streaming video frame iteration helpers.

Used by `classification_worker.py` (subprocess, no app.* on sys.path) and
by parent-process best-frame scoring fallbacks. Self-contained: only
cv2 and PIL are imported.

`open_video` mirrors MegaDetector's video_utils backend-fallback order
(default -> FFMPEG -> AVFoundation -> DShow -> MSMF -> GStreamer) so we
inherit the same codec coverage `process_video` has in production.

**How many frames you want decides how to fetch them.** Measured on real
camera-trap clips: one seek costs ~85 ms, one sequentially walked frame
~1.6 ms, so a seek is worth about 55 walked frames. Two helpers, and the
caller picks by count:

- `read_frame_by_seek` for a single frame. Seeking to the middle of a
  clip beats walking to it by 3x, and blank videos always want the
  middle frame.
- `iter_wanted_frames` for a set. It walks from frame 0 with
  `grab()`/`retrieve()` and never seeks. Seeking each of the 11 to 51
  frames a classifier crops is 1.6x to 9x *slower* than one walk.

Until 2026-08 this module never seeked at all, on the grounds that it is
unreliable across codecs. That is a real risk and it is why
`read_frame_by_seek` verifies and returns None rather than guessing:
every unverified case falls back to the walk, so an awkward codec loses
the speed-up and never gets the wrong frame.

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


def read_frame_by_seek(
    cap: cv2.VideoCapture, frame_number: int, total_frames: int
) -> Image.Image | None:
    """
    Seek straight to one frame and return it, or None if we cannot prove
    we landed on it. `cap` must be freshly opened (positioned at frame
    0), and `total_frames` is the caller's `CAP_PROP_FRAME_COUNT`, which
    it already needed to choose `frame_number`.

    Returning None means "use the sequential walk instead", never "this
    video has no such frame". Callers must have that fallback: this is
    the fast path, not the only path.

    Two gates, and the order matters. The range check is the real one:
    it decides from the container's own frame count, before any backend
    is involved, so asking for frame 900 of a 10-frame clip fails here
    rather than depending on how honestly a backend clamps an
    out-of-range seek. The position check afterwards is weaker on
    purpose. FFmpeg's seek sets its internal counter to whatever you
    asked for, so `POS_FRAMES` largely reports back the request; it
    still catches a backend that does not implement the property (0 or
    garbage, which fails the comparison and falls back), so it earns its
    two lines, but it is not what makes this safe.

    What neither gate catches is variable frame rate, where a seek
    targets a timestamp while MegaDetector numbers frames by counting
    decoded ones, so the two numberings can drift. Camera traps are
    effectively all constant frame rate; `backend/scripts/check_seek_accuracy.py`
    is how we check a new camera before trusting it.
    """
    if total_frames <= 0 or not (0 <= frame_number < total_frames):
        return None

    # A fresh capture already sits on frame 0, so seeking to it would be
    # a backend quirk we can simply not run into. Frame 0 is also the
    # frame both callers fall back to, so it is worth keeping simple.
    if frame_number > 0 and not cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number):
        return None

    success, image_bgr = cap.read()
    if not success or image_bgr is None:
        return None

    # POS_FRAMES is the index of the *next* frame, so after reading
    # frame N it reads N + 1.
    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1 != frame_number:
        return None

    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def iter_wanted_frames(
    cap: cv2.VideoCapture, wanted: set[int]
) -> Iterator[tuple[int, Image.Image]]:
    """
    Yield `(frame_number, PIL.Image)` for each frame index in `wanted`,
    reading the video sequentially from the start. Stops at end-of-stream
    even if `wanted` references higher indices (some videos report a
    higher CAP_PROP_FRAME_COUNT than they can actually decode).

    Empty `wanted` yields nothing without touching the capture.

    Frames we are skipping past are `grab()`ed rather than `read()`, which
    decodes them (unavoidable, an inter-frame codec needs them to build
    the next one) but skips copying them out into a numpy array. Same
    frames, same order, ~1.6x less work.

    One consequence: a frame that grabs but fails to *retrieve* (a
    corrupt frame mid-stream) no longer ends the iteration when we were
    only skipping past it. Before, it did, so every later frame was
    silently dropped. Now only a failed `grab()`, i.e. real
    end-of-stream, stops us.
    """
    if not wanted:
        return

    last_wanted = max(wanted)
    frame_number = 0

    while True:
        if not cap.grab():
            break

        if frame_number in wanted:
            success, image_bgr = cap.retrieve()
            if not success or image_bgr is None:
                break
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
