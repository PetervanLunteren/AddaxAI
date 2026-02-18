"""
Shared scoring logic for picking the best candidate from a set.

Used by:
- best_frame.py: picks the best video frame during detection pipeline
- event.py CRUD: picks the representative file for an event

Dependencies: only cv2, numpy. No app imports.
"""

from collections.abc import Callable

import cv2
import numpy as np

CONFIDENCE_THRESHOLD = 0.3
TOP_FRACTION = 0.9  # candidates within 90% of top score


def score_detections(detections: list[tuple[str, float]]) -> dict[str, float]:
    """
    Sum detection confidences >= CONFIDENCE_THRESHOLD, grouped by candidate key.

    Args:
        detections: list of (candidate_key, confidence) tuples.
            Caller pre-filters to animal-only detections.
    Returns:
        Dict mapping candidate_key -> summed confidence.
        Keys with no qualifying detections are omitted.
    """
    scores: dict[str, float] = {}
    for key, conf in detections:
        if conf < CONFIDENCE_THRESHOLD:
            continue
        scores[key] = scores.get(key, 0.0) + conf
    return scores


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
