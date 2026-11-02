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

A track's frames are decoded in chunks, one seek for up to
TRACK_CHUNK of them (`warm_track_frames`). Opening a 182-frame track
used to be 182 seeks into a 9 GB clip, four at a time; on an external
drive that chokes on concurrent random reads each took minutes and the
whole app queued behind them (2026-11-02). The frames of a track are
contiguous, so one pass from the chunk's first frame to its last,
keeping only the sampled frames, replaces them.

ffmpeg comes from the app's own environment, which every install has,
so the viewer never depends on a project's detector environment still
being there.
"""

from __future__ import annotations

import subprocess
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path

import cv2

from app.core.logging_config import get_logger
from app.ml.inference.video_iter import open_video
from app.utils.ffmpeg_bin import APP_ENV, resolve_ffmpeg

logger = get_logger(__name__)

# The same cap the cover frame gets, so a decoded frame is the size a
# cover is in the viewer.
FRAME_LONG_EDGE = 1920
# Deep enough to hold a long track while someone scrolls through its
# frames on the Detections tab without it evicting itself. A cached
# frame is a JPEG of a couple of hundred KB, so this is tens of MB.
FRAME_CACHE_SIZE = 256
# How many frames may be decoding at once. Every decode is one ffmpeg
# process, and the endpoints that ask for them are sync `def`, so they
# run in Starlette's threadpool: without a bound, scrolling a long
# track would take most of that pool and every other request in the app
# would queue behind video decoding.
MAX_CONCURRENT_DECODES = 4
_decode_slots = threading.BoundedSemaphore(MAX_CONCURRENT_DECODES)
# How many of a track's frames one seek decodes. The cache holds four
# chunks, more than the grid of an opened track has in flight.
TRACK_CHUNK = 64

# The decoded frames, newest last, keyed by path, frame and rate; the
# path changes on relink, so the cache never goes stale. One dict for
# the single decode and the chunk decode, so a chunk fills what the
# crop endpoint then reads.
_frames: OrderedDict[tuple[str, int, float], bytes] = OrderedDict()
_frames_lock = threading.Lock()
# One decode of a clip at a time: the second request for a chunk waits
# for the first instead of seeking into the same file beside it.
_clip_locks: dict[str, threading.Lock] = {}


def _cached(key: tuple[str, int, float]) -> bytes | None:
    with _frames_lock:
        jpeg = _frames.get(key)
        if jpeg is not None:
            _frames.move_to_end(key)
        return jpeg


def _store(key: tuple[str, int, float], jpeg: bytes) -> None:
    with _frames_lock:
        _frames[key] = jpeg
        _frames.move_to_end(key)
        while len(_frames) > FRAME_CACHE_SIZE:
            _frames.popitem(last=False)


def _clip_lock(video_path: str) -> threading.Lock:
    with _frames_lock:
        return _clip_locks.setdefault(video_path, threading.Lock())


def _scale_filter() -> str:
    """Cap the long edge at FRAME_LONG_EDGE and keep both sides even."""
    return (
        f"scale=w=trunc(iw*min(1\\,{FRAME_LONG_EDGE}/max(iw\\,ih))/2)*2:"
        f"h=trunc(ih*min(1\\,{FRAME_LONG_EDGE}/max(iw\\,ih))/2)*2"
    )


def _seek_time(frame_number: int, fps: float) -> str:
    """Half a frame early, so float rounding cannot land on the next frame."""
    return f"{max(0.0, (frame_number - 0.5) / fps):.6f}"


def frame_seek_cmd(ffmpeg: str, video: Path, frame_number: int, fps: float) -> list[str]:
    """Decode frame ``frame_number`` of ``video`` as one JPEG on stdout.

    ``-ss`` before ``-i`` seeks to the last keyframe and decodes forward
    to the target time, so the first output frame is the first frame at
    or after it.
    """
    return [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-hwaccel", "auto",
        "-ss", _seek_time(frame_number, fps),
        "-i", str(video),
        "-frames:v", "1",
        "-vf", _scale_filter(),
        "-f", "image2",
        "-c:v", "mjpeg",
        "-q:v", "3",
        "pipe:1",
    ]


def frames_seek_cmd(
    ffmpeg: str, video: Path, first: int, offsets: list[int], fps: float, out_dir: Path
) -> list[str]:
    """Decode the frames ``first + o`` for each offset, as numbered JPEGs
    in ``out_dir``, with one seek.

    After the seek the select filter counts frames from the target, so
    offset 0 is the frame the single decode gives for ``first``.
    Passthrough timing keeps the filter from duplicating or dropping
    frames to fit an output rate, and the output frame count stops the
    decode right after the last wanted frame.
    """
    pick = "+".join(f"eq(n\\,{o})" for o in offsets)
    return [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-hwaccel", "auto",
        "-ss", _seek_time(first, fps),
        "-i", str(video),
        "-vf", f"select='{pick}',{_scale_filter()}",
        "-fps_mode", "passthrough",
        "-frames:v", str(len(offsets)),
        "-f", "image2",
        "-c:v", "mjpeg",
        "-q:v", "3",
        str(out_dir / "%06d.jpg"),
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


def _ffmpeg() -> str | None:
    try:
        return resolve_ffmpeg(APP_ENV)
    except RuntimeError as e:
        logger.warning(f"Frame decode: {e}")
        return None


def decode_frame_jpeg(
    video_path: str, frame_number: int, frame_rate: float | None
) -> bytes | None:
    """JPEG bytes of one frame, or None when it cannot be decoded (a
    frame past the end, a file that will not open, ffmpeg missing).
    Cached by path, frame and rate."""
    if frame_number < 0:
        return None
    fps = frame_rate or _frame_rate_of(video_path)
    if not fps:
        return None
    key = (video_path, frame_number, fps)
    jpeg = _cached(key)
    if jpeg is not None:
        return jpeg
    ffmpeg = _ffmpeg()
    if ffmpeg is None:
        return None
    with _decode_slots:
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
    _store(key, result.stdout)
    return result.stdout


def decode_frames_jpeg(
    video_path: str, frame_numbers: list[int], frame_rate: float | None
) -> dict[int, bytes]:
    """JPEG bytes of the listed frames, decoded with one seek and put in
    the cache. Frames ffmpeg did not deliver (past the end) are left out.
    """
    wanted = sorted({n for n in frame_numbers if n >= 0})
    if not wanted:
        return {}
    fps = frame_rate or _frame_rate_of(video_path)
    if not fps:
        return {}
    ffmpeg = _ffmpeg()
    if ffmpeg is None:
        return {}
    first = wanted[0]
    offsets = [n - first for n in wanted]
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as out:
        out_dir = Path(out)
        with _decode_slots:
            result = subprocess.run(
                frames_seek_cmd(ffmpeg, Path(video_path), first, offsets, fps, out_dir),
                capture_output=True,
                check=False,
            )
        files = sorted(out_dir.glob("*.jpg"))
        if result.returncode != 0 and not files:
            stderr = result.stderr.decode(errors="replace").strip()[-300:]
            logger.warning(
                f"Frame decode: frames {first} to {wanted[-1]} of {video_path} gave "
                f"nothing (exit {result.returncode}): {stderr}"
            )
            return {}
        got = {n: path.read_bytes() for n, path in zip(wanted, files, strict=False)}
    for n, jpeg in got.items():
        _store((video_path, n, fps), jpeg)
    logger.info(
        f"Decoded {len(got)} of {len(wanted)} frames from frame {first} of "
        f"{Path(video_path).name} in {time.monotonic() - started:.1f}s"
    )
    return got


def warm_track_frames(
    video_path: str, track_frames: list[int], start: int, frame_rate: float | None
) -> None:
    """When frame ``start`` is not cached, put the chunk of the track's
    frames from it on in the cache: up to TRACK_CHUNK of ``track_frames``
    (ascending), decoded with one seek. A cached start is a no-op, so
    the chunks line up from each missed frame and a request for the
    frame after a chunk's first does not seek for the one frame past
    its end. One decode of a clip at a time, so the requests for a
    chunk's other frames wait for the first one's decode and then read
    the cache.
    """
    try:
        at = track_frames.index(start)
    except ValueError:
        return
    fps = frame_rate or _frame_rate_of(video_path)
    if not fps:
        return
    with _clip_lock(video_path):
        if _cached((video_path, start, fps)) is not None:
            return
        chunk = track_frames[at : at + TRACK_CHUNK]
        missing = [n for n in chunk if _cached((video_path, n, fps)) is None]
        decode_frames_jpeg(video_path, missing, fps)
