"""
Shared scoring logic for picking the best video frame.

Used by:
- best_frame.py: classifier-off runs, which open the videos themselves
- classification_worker.py (subprocess): classifier-on runs, which score
  in the same pass that crops and classifies

Both decide entirely from the detection JSON, before any frame is
decoded, so the caller can decode exactly the frame it is going to keep.
See `choose_frame_number`.

The classifier subprocess runs in `env-pytorch` (Python 3.8), so this
module must stay 3.8-importable. We use `from __future__ import
annotations` to defer PEP 585 generic syntax in function signatures
and variable annotations, and `typing.Tuple` for the runtime `Bbox`
alias (PEP 585 `tuple[...]` only supports subscription on 3.9+).

Dependencies: stdlib only. No app imports, no cv2, no numpy.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Tuple  # noqa: UP035  -- runtime alias must work on py3.8 subprocess

CONFIDENCE_THRESHOLD = 0.3
TOP_FRACTION = 0.9  # candidates within 90% of top score

# Runtime alias — must be subscript-compatible on Python 3.8. `tuple[...]`
# isn't (that's PEP 585, 3.9+), so use `typing.Tuple[...]`. The alias
# itself is still referenced inside function annotations elsewhere in
# this module, where __future__ deferral keeps everything string-based.
Bbox = Tuple[float, float, float, float]  # noqa: UP006  -- (x, y, width, height); py3.8 runtime


def compute_union_area(boxes: list[Bbox]) -> float:
    """
    Compute the true geometric union area of axis-aligned rectangles.

    Uses coordinate compression: collect unique x/y values, iterate grid cells,
    sum areas of cells covered by at least one rectangle.

    Args:
        boxes: list of (x, y, width, height) tuples (normalised or pixel coords).
    Returns:
        Union area (same units as input coords, squared).
    """
    # Convert (x, y, w, h) -> (x1, y1, x2, y2), skip degenerate
    rects: list[tuple[float, float, float, float]] = []
    for x, y, w, h in boxes:
        if w <= 0 or h <= 0:
            continue
        rects.append((x, y, x + w, y + h))

    if not rects:
        return 0.0

    # Collect unique x and y coordinates
    xs = sorted({r[0] for r in rects} | {r[2] for r in rects})
    ys = sorted({r[1] for r in rects} | {r[3] for r in rects})

    area = 0.0
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            cx, cy = xs[i], ys[j]
            cw, ch = xs[i + 1] - xs[i], ys[j + 1] - ys[j]
            # Check if any rectangle covers this cell
            for x1, y1, x2, y2 in rects:
                if x1 <= cx and cx + cw <= x2 and y1 <= cy and cy + ch <= y2:
                    area += cw * ch
                    break

    return area


def score_detections(
    detections: list[tuple[str, float, Bbox]],
) -> dict[str, tuple[int, float]]:
    """
    Score candidates as (confidence_bin, sqrt_area) tuples.

    Tiered lexicographic scoring:
    1. confidence_bin (primary): sum of round(conf * 100) per candidate.
    2. sqrt(union_bbox_area) (secondary): dampens area so 2x difference → 1.4x.

    Args:
        detections: list of (candidate_key, confidence, bbox) tuples.
            bbox is (x, y, width, height).
    Returns:
        Dict mapping candidate_key -> (conf_bin, sqrt_area).
        Keys with no qualifying detections are omitted.
    """
    conf_bins: dict[str, int] = {}
    boxes: dict[str, list[Bbox]] = defaultdict(list)

    for key, conf, bbox in detections:
        if conf < CONFIDENCE_THRESHOLD:
            continue
        conf_bins[key] = conf_bins.get(key, 0) + round(conf * 100)
        boxes[key].append(bbox)

    return {
        key: (conf_bin, math.sqrt(compute_union_area(boxes[key])))
        for key, conf_bin in conf_bins.items()
    }


def pick_best_candidate(
    scores: dict[str, tuple[int, float]],
) -> str | None:
    """
    Pick the best candidate key using tiered lexicographic selection.

    1. Tier 1 — confidence bin: only candidates with the highest conf_bin.
    2. Tier 2 — sqrt(area): among conf ties, within TOP_FRACTION of best,
       then the largest of those. Ties resolve to the first candidate
       seen, which for video frames is the earliest frame because the
       detection list arrives in frame order.

    Returns None for empty scores; a blank video has no detection to
    anchor on and its caller picks a frame by position instead.

    There used to be a third tier, Laplacian sharpness, and a blank-video
    branch that delegated to it. Both are gone. Sharpness required the
    pixels of every candidate frame, which forced the caller to decode
    and hold each one before it could know which single frame it wanted.
    Measured over real deployments the tier never once broke a tie:
    summed detection confidence decided every video that had detections
    at all. Paying a decode of the whole candidate set for a tiebreak
    that does not fire is a bad trade, and dropping it lets this module
    answer from the JSON alone.

    Args:
        scores: output of score_detections(), mapping key -> (conf_bin, sqrt_area).
    """
    if not scores:
        return None

    # Tier 1: highest confidence bin
    best_conf = max(s[0] for s in scores.values())
    candidates = [k for k, s in scores.items() if s[0] == best_conf]

    if len(candidates) == 1:
        return candidates[0]

    # Tier 2: among confidence ties, best sqrt(area) within TOP_FRACTION
    best_area = max(scores[k][1] for k in candidates)
    threshold = best_area * TOP_FRACTION
    area_candidates = [k for k in candidates if scores[k][1] >= threshold]

    if len(area_candidates) == 1:
        return area_candidates[0]

    return max(area_candidates, key=lambda k: scores[k][1])


def choose_frame_number(
    detections: list[dict],
    total_frames: int,
) -> int:
    """
    Decide a video's best frame from its detections alone. No pixels.

    `detections` is every detection on the video, any category, each with
    `frame_number`, `conf` and `bbox`. Scoring on the detector's own
    confidence regardless of category is what makes this one rule for
    every detector and classifier combination (see `best_frame.py`).

    Blank videos, and videos whose detections are all below
    CONFIDENCE_THRESHOLD, get the middle frame. There is nothing to aim
    at, so position is as good a choice as any, and it beats the first
    frame because camera traps often start on the empty scene that
    triggered them.

    The returned frame is the caller's target to decode, not a promise
    that it decodes: a container can report more frames than it can
    actually yield. Callers must handle the frame never arriving and
    must keep `best_frame_number` and the written JPEG in agreement.
    """
    scored = [
        (
            str(int(det["frame_number"])),
            float(det.get("conf", 0.0)),
            tuple(det["bbox"]),
        )
        for det in detections
        if det.get("frame_number") is not None
    ]
    best_key = pick_best_candidate(score_detections(scored))
    if best_key is not None:
        return int(best_key)
    return max(0, total_frames // 2)
