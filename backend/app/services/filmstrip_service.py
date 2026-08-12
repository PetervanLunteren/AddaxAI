"""
On-demand video filmstrip.

Decodes a handful of evenly-spaced (in time) low-resolution frames from a
video so the counts modal can show a gallery instead of a single best
frame. Nothing is persisted to disk: frames are decoded on request and the
result is cached in memory (a filmstrip is immutable per video file).

Reuses the decode primitives in `app.ml.inference.video_iter` so sampling
and decoding behave exactly like the rest of the pipeline. It walks with
`iter_wanted_frames` rather than seeking to each frame, which is the right
call for a 9-frame sample: a seek costs about 55 walked frames, so seeking
9 times only wins on clips longer than ~500 frames. That does leave a
30-second clip decoding roughly its whole length for 9 frames. The
in-memory cache and the frontend's prefetch hide the cost in normal use,
and switching to seeks above a length threshold is a real optimisation
nobody has needed yet.
"""

from __future__ import annotations

import base64
import io
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

import cv2
from PIL import Image

from app.core.logging_config import get_logger
from app.ml.inference.video_iter import iter_wanted_frames, open_video, sample_indices

logger = get_logger(__name__)

# 3x3 gallery of small frames. Low quality on purpose: this is a temporal
# preview, not a review surface (boxes / full quality live in the player).
FILMSTRIP_FRAME_COUNT = 9
FILMSTRIP_MAX_WIDTH = 320
FILMSTRIP_JPEG_QUALITY = 70


class FilmstripFrameData(TypedDict):
    frame_number: int
    time_seconds: float | None
    image: str  # "data:image/jpeg;base64,..."


@lru_cache(maxsize=48)
def build_filmstrip(
    file_path: str,
    frame_rate: float | None,
) -> tuple[FilmstripFrameData, ...]:
    """
    Decode up to FILMSTRIP_FRAME_COUNT evenly-spaced frames from a video.

    Returns one entry per decoded frame (fewer than the target only when the
    video is shorter, or a codec over-reports its frame count). Returns an
    empty tuple when the video can't be opened, so callers can fall back to
    the best-frame still. Cached by (path, frame_rate); the path changes on
    relink, so the cache never goes stale.
    """
    cap = open_video(Path(file_path))
    if cap is None:
        logger.warning("Filmstrip: could not open video %s", file_path)
        return ()
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        wanted = sample_indices(total, FILMSTRIP_FRAME_COUNT)
        if not wanted:
            return ()

        frames: list[FilmstripFrameData] = []
        for num, pil in iter_wanted_frames(cap, set(wanted), file_path):
            if pil.mode != "RGB":
                pil = pil.convert("RGB")
            if pil.width > FILMSTRIP_MAX_WIDTH:
                ratio = FILMSTRIP_MAX_WIDTH / pil.width
                pil = pil.resize(
                    (FILMSTRIP_MAX_WIDTH, int(pil.height * ratio)), Image.LANCZOS
                )
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=FILMSTRIP_JPEG_QUALITY)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            frames.append(
                FilmstripFrameData(
                    frame_number=num,
                    time_seconds=(num / frame_rate) if frame_rate else None,
                    image=f"data:image/jpeg;base64,{b64}",
                )
            )
        return tuple(frames)
    finally:
        cap.release()
