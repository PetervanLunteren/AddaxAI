"""
Check that fetching one video frame by seeking returns the same pixels as
walking to it, over a folder of real videos.

Why this exists: `read_frame_by_seek` is what stops AddaxAI decoding half
of every clip to write one thumbnail, and it is only used when it can
verify where it landed, so an awkward codec loses the speed-up rather
than getting the wrong picture. That makes it safe on formats nobody has
tested, but it says nothing about whether it *fires*. This script answers
that, on whatever camera footage you point it at, and it compares decoded
pixels rather than trusting any cv2 property.

Run it against a new camera make before assuming videos got faster, and
send it to a beta tester when their footage looks slow:

    backend/venv/bin/python backend/scripts/check_seek_accuracy.py <folder>

A "MISMATCH" line is serious: it means the seek was verified and still
returned the wrong frame, which would put one moment's picture behind
another moment's boxes. Nothing has ever produced one; if something does,
that codec needs a guard in `read_frame_by_seek`, not a wider tolerance.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_BACKEND))

import cv2  # noqa: E402

from app.core.media_types import VIDEO_EXTENSIONS  # noqa: E402
from app.ml.inference.video_iter import (  # noqa: E402
    iter_wanted_frames,
    open_video,
    read_frame_by_seek,
)

# The frame a blank video asks for is its middle one, so that is the case
# worth measuring. The quarter points are there to catch a codec that
# only behaves near a keyframe.
FRACTIONS = (0.25, 0.5, 0.75)


def check_one(path: Path) -> tuple[int, int, int, float, float]:
    """Return (exact, refused, mismatched, seek_seconds, walk_seconds)."""
    exact = refused = mismatched = 0
    seek_seconds = walk_seconds = 0.0

    cap = open_video(path)
    if cap is None:
        print(f"  {path.name}: could not open, skipped")
        return 0, 0, 0, 0.0, 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    for fraction in FRACTIONS:
        target = int(total * fraction)

        seek_cap = open_video(path)
        if seek_cap is None:
            continue
        started = time.perf_counter()
        try:
            seeked = read_frame_by_seek(seek_cap, target, total)
        finally:
            seek_cap.release()
        seek_seconds += time.perf_counter() - started

        walk_cap = open_video(path)
        if walk_cap is None:
            continue
        started = time.perf_counter()
        walked = None
        try:
            for num, image in iter_wanted_frames(walk_cap, {target}):
                if num == target:
                    walked = image
        finally:
            walk_cap.release()
        walk_seconds += time.perf_counter() - started

        if seeked is None:
            refused += 1
        elif walked is None or list(seeked.getdata()) != list(walked.getdata()):
            mismatched += 1
            print(
                f"  MISMATCH {path.name} frame {target}: the seek was "
                f"verified but returned different pixels than the walk"
            )
        else:
            exact += 1

    return exact, refused, mismatched, seek_seconds, walk_seconds


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2

    folder = Path(sys.argv[1]).expanduser()
    if not folder.is_dir():
        print(f"Not a folder: {folder}")
        return 2

    videos = sorted(
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        print(f"No videos found under {folder}")
        return 2

    print(f"Checking {len(videos)} video(s) under {folder}\n")
    exact = refused = mismatched = 0
    seek_seconds = walk_seconds = 0.0
    for path in videos:
        e, r, m, s, w = check_one(path)
        exact += e
        refused += r
        mismatched += m
        seek_seconds += s
        walk_seconds += w

    checked = exact + refused + mismatched
    if not checked:
        print("Nothing was decodable.")
        return 1

    print(f"\n{checked} frame fetches across {len(videos)} video(s)")
    print(f"  exact (fast path used) : {exact} ({exact / checked:.0%})")
    print(f"  refused (fell back)    : {refused} ({refused / checked:.0%})")
    print(f"  MISMATCHED             : {mismatched}")
    if walk_seconds > 0:
        print(
            f"\n  seeking {seek_seconds:.1f}s vs walking {walk_seconds:.1f}s "
            f"= {walk_seconds / seek_seconds:.1f}x faster"
        )
    if mismatched:
        print("\nA mismatch means a verified seek returned the wrong frame.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
