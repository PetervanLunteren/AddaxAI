"""
CRUD for the Calibration page: the sample to draw from, the answers,
and the per-label summary with its curve.

The math lives in `app/ml/calibration.py`; this module only reads and
writes rows. Calibration is kept apart from verification: it never
writes a label or a verified flag. It reads `verified` only where the
Labels grid does, to decide which detections are in scope.
"""

from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sqlalchemy import func
from sqlalchemy.orm import Query, Session

from app.api.crud.performance import build_taxonomy_lookup
from app.api.schemas.calibration import (
    CalibrationAnswerValue,
    CalibrationLabel,
    CalibrationPoint,
    CalibrationResponse,
    CalibrationSampleItem,
    CalibrationThreshold,
)
from app.core.logging_config import get_logger
from app.ml.calibration import (
    MIN_RIGHT,
    MIN_WRONG,
    TARGETS,
    draw_sample,
    fit_calibration,
    has_enough_answers,
    rule_of_three_bound,
)
from app.ml.detection_visibility import on_visible_frame
from app.ml.label_exclusion import is_a_real_detection, threshold_or_verified
from app.models import CalibrationAnswer, Deployment, Detection, File, Project

logger = get_logger(__name__)

# Detections per Calibrate batch, and the score bins it is spread over.
BATCH_SIZE = 20
N_BINS = 5
# Points on the curve sent to the page.
CURVE_POINTS = 50


def _get_project(db: Session, project_id: str) -> Project:
    project = db.query(Project).filter(Project.id == project_id).first()
    if project is None:
        raise ValueError(f"Project {project_id} not found")
    return project


def _in_project(query: Query[Any], project_id: str) -> Query[Any]:
    return (
        query.join(File, File.id == Detection.file_id)
        .join(Deployment, Deployment.id == File.deployment_id)
        .filter(Deployment.project_id == project_id)
    )


def _pool(db: Session, project: Project, *columns: Any) -> Query[Any]:
    """Detections the sample draws from: what the Labels grid shows, with an AI label.

    Same scope as the grid (`threshold_or_verified` plus the visible-frame
    rule), restricted to detections the classifier labelled and scored.
    The AI's own call (`original_label`) is used, so a relabelled
    detection still counts under the label the AI gave it. The "nothing
    here" labels are left out: "Is this a blank?" is not a question about
    a species.
    """
    return (
        _in_project(db.query(*columns), project.id)
        .filter(threshold_or_verified(project.counting_threshold))
        .filter(on_visible_frame())
        .filter(Detection.original_label.isnot(None))
        .filter(Detection.original_label_confidence.isnot(None))
        .filter(is_a_real_detection(Detection.original_label))
    )


def _summarise(
    answers: list[tuple[float, str]],
) -> tuple[int, int, int, list[CalibrationThreshold] | None,
           list[CalibrationPoint] | None, float | None]:
    right = sum(1 for _, a in answers if a == "right")
    wrong = sum(1 for _, a in answers if a == "wrong")
    unsure = sum(1 for _, a in answers if a == "unsure")
    if not has_enough_answers(right, wrong):
        bound = rule_of_three_bound(right) if wrong == 0 else None
        return right, wrong, unsure, None, None, bound

    judged = [(s, a) for s, a in answers if a != "unsure"]
    scores = np.array([s for s, _ in judged])
    try:
        fit = fit_calibration(scores, np.array([1 if a == "right" else 0 for _, a in judged]))
    except (RuntimeError, np.linalg.LinAlgError):
        # One label that will not fit must not take every other row down
        # with it; the row then shows its counts without a curve.
        logger.exception("Calibration fit failed for %d answers", len(judged))
        return right, wrong, unsure, None, None, None
    grid = np.linspace(scores.min(), scores.max(), CURVE_POINTS)
    p, lower, upper = fit.predict(grid)
    curve = [
        CalibrationPoint(score=float(s), p=float(pi), lower=float(lo), upper=float(hi))
        for s, pi, lo, hi in zip(grid, p, lower, upper, strict=True)
    ]
    thresholds = [CalibrationThreshold(target=t, score=fit.threshold(t)) for t in TARGETS]
    return right, wrong, unsure, thresholds, curve, None


def get_calibration(db: Session, project_id: str) -> CalibrationResponse:
    """One row per AI label in the project, busiest first."""
    project = _get_project(db, project_id)

    pool_counts: dict[str, int] = dict(
        _pool(db, project, Detection.original_label, func.count(Detection.id))
        .group_by(Detection.original_label)
        .all()
    )

    answers: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for label, score, answer in _in_project(
        db.query(CalibrationAnswer.label, CalibrationAnswer.score, CalibrationAnswer.answer)
        .join(Detection, Detection.id == CalibrationAnswer.detection_id),
        project.id,
    ).all():
        answers[label].append((score, answer))

    taxonomy = build_taxonomy_lookup(db, project)
    rows: list[CalibrationLabel] = []
    for label in sorted(
        set(pool_counts) | set(answers),
        key=lambda name: (-pool_counts.get(name, 0), name.lower()),
    ):
        right, wrong, unsure, thresholds, curve, bound = _summarise(answers[label])
        tax = taxonomy.get(label.lower())
        plain = label.replace("_", " ")
        rows.append(
            CalibrationLabel(
                label=label,
                common_name=(tax.common_name if tax and tax.common_name else plain),
                scientific_name=(tax.scientific_name if tax and tax.scientific_name else plain),
                detections=pool_counts.get(label, 0),
                right=right,
                wrong=wrong,
                unsure=unsure,
                right_needed=max(0, MIN_RIGHT - right),
                wrong_needed=max(0, MIN_WRONG - wrong),
                thresholds=thresholds,
                curve=curve,
                no_error_bound=bound,
            )
        )
    return CalibrationResponse(labels=rows)


def draw_batch(
    db: Session,
    project_id: str,
    label: str,
    rng: np.random.Generator | None = None,
) -> list[CalibrationSampleItem]:
    """The next batch to ask about: unanswered detections of `label`, spread over the scores.

    The score range and the draw per bin happen in SQL, so a label with
    hundreds of thousands of detections hands Python at most
    `N_BINS * BATCH_SIZE` rows, not all of them. Each bin brings a full
    batch, so a bin that runs dry can still hand its share on.
    """
    project = _get_project(db, project_id)
    score = Detection.original_label_confidence
    unanswered = (
        _pool(db, project, Detection.id, score)
        .filter(Detection.original_label == label)
        .filter(Detection.id.notin_(db.query(CalibrationAnswer.detection_id)))
    )
    lo, hi = unanswered.with_entities(func.min(score), func.max(score)).one()
    if lo is None:
        return []
    width = (hi - lo) / N_BINS
    candidates: list[tuple[str, float]] = []
    for b in range(N_BINS):
        in_bin = unanswered
        if b > 0:
            in_bin = in_bin.filter(score >= lo + b * width)
        if b < N_BINS - 1:
            in_bin = in_bin.filter(score < lo + (b + 1) * width)
        candidates += in_bin.order_by(func.random()).limit(BATCH_SIZE).all()
    rng = rng if rng is not None else np.random.default_rng()
    picked = draw_sample(
        candidates, batch_size=BATCH_SIZE, n_bins=N_BINS, rng=rng, score_range=(lo, hi)
    )
    # Shown in random order, so a run of low scores does not prime the answer.
    rng.shuffle(picked)
    scores = dict(candidates)
    return [CalibrationSampleItem(detection_id=i, score=scores[i]) for i in picked]


def save_answer(
    db: Session, detection_id: str, label: str, answer: CalibrationAnswerValue
) -> CalibrationAnswer:
    """Store (or replace) the answer for one detection, with the AI's label and score now.

    `label` is the one the question showed. A reprocess rewrites
    `original_label` on unverified detections, so between drawing the
    batch and answering it may have changed; the answer is then about a
    label the detection no longer carries, and is refused.
    """
    detection = db.get(Detection, detection_id)
    if detection is None:
        raise ValueError(f"Detection {detection_id} not found")
    if detection.original_label != label:
        raise ValueError(f"Detection {detection_id} is no longer labelled {label}")
    project = (
        db.query(Project)
        .join(Deployment, Deployment.project_id == Project.id)
        .join(File, File.deployment_id == Deployment.id)
        .filter(File.id == detection.file_id)
        .one()
    )
    # Only what the sample can draw: an answer on anything else would feed
    # the curve with a detection the page never asks about.
    if _pool(db, project, Detection.id).filter(Detection.id == detection_id).first() is None:
        raise ValueError(f"Detection {detection_id} is not in the calibration sample")

    row = (
        db.query(CalibrationAnswer)
        .filter(CalibrationAnswer.detection_id == detection_id)
        .first()
    )
    if row is None:
        row = CalibrationAnswer(detection_id=detection_id)
        db.add(row)
    row.label = detection.original_label
    row.score = detection.original_label_confidence
    row.answer = answer
    row.answered_at_utc = datetime.now(UTC)
    db.commit()
    return row
