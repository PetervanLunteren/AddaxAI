"""
Shared scoring logic for picking the best candidate from a set.

Used by:
- best_frame.py: picks the best video frame during detection pipeline
- event.py CRUD: picks the representative file for an event

Dependencies: only cv2, numpy. No app imports.
"""

from collections import defaultdict
from collections.abc import Callable

import cv2
import numpy as np

CONFIDENCE_THRESHOLD = 0.3
TOP_FRACTION = 0.9  # candidates within 90% of top score

Bbox = tuple[float, float, float, float]  # (x, y, width, height)


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
) -> dict[str, float]:
    """
    Score candidates by n_detections * sum(confidence) * union_bbox_area.

    Prioritises frames with the most individuals visible, then rewards
    larger / more confident detections for easier species identification.

    Args:
        detections: list of (candidate_key, confidence, bbox) tuples.
            bbox is (x, y, width, height).
    Returns:
        Dict mapping candidate_key -> score.
        Keys with no qualifying detections are omitted.
    """
    confs: dict[str, float] = {}
    boxes: dict[str, list[Bbox]] = defaultdict(list)

    for key, conf, bbox in detections:
        if conf < CONFIDENCE_THRESHOLD:
            continue
        confs[key] = confs.get(key, 0.0) + conf
        boxes[key].append(bbox)

    return {
        key: len(boxes[key]) * conf * compute_union_area(boxes[key])
        for key, conf in confs.items()
    }


def compute_sharpness(image_np: np.ndarray) -> float:
    """
    Image sharpness via Laplacian variance. Higher = sharper.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def pick_best_candidate(
    scores: dict[str, float],
    get_sharpest: Callable[[list[str]], str] | None = None,
    fallback_keys: list[str] | None = None,
) -> str | None:
    """
    Pick the best candidate key from a scored set.

    Algorithm:
    1. If scores is empty (blank): call get_sharpest(fallback_keys).
    2. Else: find candidates within TOP_FRACTION of max score.
    3. If 1 candidate: return it.
    4. If multiple + get_sharpest provided: return get_sharpest(candidates).
    5. If multiple + no get_sharpest: return the highest-scoring one.

    Args:
        scores: output of score_detections().
        get_sharpest: takes a list of candidate keys, returns the sharpest one.
            Called lazily only when needed (tiebreaker or blank case).
        fallback_keys: keys to pass to get_sharpest when scores is empty.
    """
    if not scores:
        if get_sharpest and fallback_keys:
            return get_sharpest(fallback_keys)
        return None

    best_score = max(scores.values())
    threshold = best_score * TOP_FRACTION
    candidates = [k for k, s in scores.items() if s >= threshold]

    if len(candidates) == 1:
        return candidates[0]

    if get_sharpest:
        return get_sharpest(candidates)

    # No sharpness tiebreaker — return the highest-scoring one
    return max(candidates, key=lambda k: scores[k])
