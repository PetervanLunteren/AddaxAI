"""
Shared scoring logic for picking the best candidate from a set.

Used by:
- best_frame.py: picks the best video frame during detection pipeline
- event.py CRUD: picks the representative file for an event

Dependencies: only cv2, numpy. No app imports.
"""

import math
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


def compute_sharpness(image_np: np.ndarray) -> float:
    """
    Image sharpness via Laplacian variance. Higher = sharper.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def pick_best_candidate(
    scores: dict[str, tuple[int, float]],
    get_sharpest: Callable[[list[str]], str] | None = None,
    fallback_keys: list[str] | None = None,
) -> str | None:
    """
    Pick the best candidate key using tiered lexicographic selection.

    Algorithm:
    1. If scores is empty (blank): call get_sharpest(fallback_keys).
    2. Tier 1 — confidence bin: only candidates with the highest conf_bin.
    3. Tier 2 — sqrt(area): among conf ties, within TOP_FRACTION of best.
    4. Tier 3 — sharpness: tiebreaker among remaining candidates.

    Args:
        scores: output of score_detections(), mapping key -> (conf_bin, sqrt_area).
        get_sharpest: takes a list of candidate keys, returns the sharpest one.
            Called lazily only when needed (tiebreaker or blank case).
        fallback_keys: keys to pass to get_sharpest when scores is empty.
    """
    if not scores:
        if get_sharpest and fallback_keys:
            return get_sharpest(fallback_keys)
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

    # Tier 3: sharpness tiebreaker
    if get_sharpest:
        return get_sharpest(area_candidates)

    return max(area_candidates, key=lambda k: scores[k][1])
