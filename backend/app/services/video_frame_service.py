"""
One frame of a video, decoded on request.

Nothing per frame is stored for a video beyond its cover and one crop
per track; any other frame a person looks at (a track's frame in the
Files viewer, the frame a count was made on, the card of a drawn box)
is decoded here with ffmpeg and kept in a small memory cache. A
keyframe seek (``-ss`` before ``-i``) then an accurate trim to the
frame, with the platform's hardware decoder when it has one, costs a
fraction of a second on most clips and a couple of seconds deep inside
a long-GOP HEVC file; the cache covers going back and forth.

ffmpeg comes from the app's own environment, which every install has,
so the viewer never depends on a project's detector environment still
being there.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

import cv2

from app.core.logging_config import get_logger
from app.ml.inference.video_iter import open_video
from app.utils.ffmpeg_bin import APP_ENV, resolve_ffmpeg

logger = get_logger(__name__)

# The same cap the cover frame gets, so a decoded frame is the size a
# cover is in the viewer.
FRAME_LONG_EDGE = 1920
FRAME_CACHE_SIZE = 64


def frame_seek_cmd(ffmpeg: str, video: Path, frame_number: int, fps: float) -> list[str]:
    """Decode frame ``frame_number`` of ``video`` as one JPEG on stdout.

    ``-ss`` before ``-i`` seeks to the last keyframe and decodes forward
    to the target time, so the first output frame is the first frame at
    or after it; aiming half a frame early keeps float rounding from
    landing on the next frame. The scale caps the long edge and keeps
    both sides even.
    """
    seek = max(0.0, (frame_number - 0.5) / fps)
    scale = (
        f"scale=w=trunc(iw*min(1\\,{FRAME_LONG_EDGE}/max(iw\\,ih))/2)*2:"
        f"h=trunc(ih*min(1\\,{FRAME_LONG_EDGE}/max(iw\\,ih))/2)*2"
    )
    return [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-hwaccel", "auto",
        "-ss", f"{seek:.6f}",
        "-i", str(video),
        "-frames:v", "1",
        "-vf", scale,
        "-f", "image2",
        "-c:v", "mjpeg",
        "-q:v", "3",
        "pipe:1",
    ]


def _frame_rate_of(video_path: str) -> float | None:
    cap = open_video(Path(video_path))
    if cap is None:
        return None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    return fps if fps > 0 else None


@lru_cache(maxsize=FRAME_CACHE_SIZE)
def decode_frame_jpeg(
    video_path: str, frame_number: int, frame_rate: float | None
) -> bytes | None:
    """JPEG bytes of one frame, or None when it cannot be decoded (a
    frame past the end, a file that will not open, ffmpeg missing).
    Cached by path, frame and rate; the path changes on relink, so the
    cache never goes stale."""
    if frame_number < 0:
        return None
    fps = frame_rate or _frame_rate_of(video_path)
    if not fps:
        return None
    try:
        ffmpeg = resolve_ffmpeg(APP_ENV)
    except RuntimeError as e:
        logger.warning(f"Frame decode: {e}")
        return None
    result = subprocess.run(
        frame_seek_cmd(ffmpeg, Path(video_path), frame_number, fps),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout:
        stderr = result.stderr.decode(errors="replace").strip()[-300:]
        logger.warning(
            f"Frame decode: frame {frame_number} of {video_path} gave nothing "
            f"(exit {result.returncode}): {stderr}"
        )
        return None
    return result.stdout
