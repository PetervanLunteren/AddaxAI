"""
CRUD for the confusion matrix + classification report insight views.

Per-detection comparison of original machine prediction
(Detection.original_label captured at JSON load) vs current label
(after human verification or relabel). Only verified detections count.
Metrics are computed server-side so the React page stays thin and the
math is unit-testable in isolation.
"""

from collections import Counter, defaultdict
from datetime import date
from typing import Literal

from sqlalchemy.orm import Session

from app.api.schemas.performance import ClassMetrics, PerformanceResponse
from app.models.deployment import Deployment
from app.models.detection import Detection
from app.models.file import File
from app.models.label_taxonomy import LabelTaxonomy
from app.models.project import Project

Rank = Literal["class", "order", "family", "genus", "species"]

DETECTOR_CATEGORIES: tuple[str, ...] = ("animal", "person", "vehicle")
RANKS: tuple[Rank, ...] = ("class", "order", "family", "genus", "species")
OTHER_BUCKET = "other"
UNCLASSIFIED = "animal"  # animal detections with no classifier output


def _taxon_at_rank(row: LabelTaxonomy | None, rank: Rank) -> str | None:
    """
    Value used to group a taxonomy row at the requested rank.

    At the species rank we use the unique leaf name rather than
    `taxon_species` alone, which is often just the species epithet
    (e.g. 'pardus') and would collide across genera.
    """
    if row is None:
        return None
    if rank == "species":
        return row.name
    return getattr(row, f"taxon_{rank}", None)


def _display_for(name: str, row: LabelTaxonomy | None) -> str:
    """Human-friendly label for the matrix axis."""
    if name in DETECTOR_CATEGORIES or name == OTHER_BUCKET:
        return name
    if row is not None and row.display_name:
        return row.display_name
    return name.replace("_", " ")


def _build_taxonomy_lookup(
    db: Session, project: Project,
) -> dict[str, LabelTaxonomy]:
    """
    Map label name (lowercased) to the taxonomy row used to roll it up.

    Includes rows scoped to the project's classification model plus any
    project-scoped custom rows. Also includes built-in detector rows
    (animal / person / vehicle) which sit on a dedicated model id.
    """
    if project.classification_model_id is None:
        return {}

    rows = (
        db.query(LabelTaxonomy)
        .filter(LabelTaxonomy.classification_model_id == project.classification_model_id)
        .filter(
            (LabelTaxonomy.project_id == None)  # noqa: E711
            | (LabelTaxonomy.project_id == project.id)
        )
        .all()
    )
    lookup: dict[str, LabelTaxonomy] = {}
    for r in rows:
        lookup[r.name.lower()] = r
    return lookup


def _class_for_current(
    det: Detection, taxonomy_lookup: dict[str, LabelTaxonomy], rank: Rank,
) -> str:
    """
    Current (ground-truth) class at the requested rank.

    Person / vehicle detections always use their category. Animals with
    no label are bucketed as 'animal'. Animals with a label resolve via
    taxonomy when possible; otherwise fall back to the label itself.
    """
    if det.category in ("person", "vehicle"):
        return det.category
    if not det.label:
        return UNCLASSIFIED
    row = taxonomy_lookup.get(det.label.lower())
    value = _taxon_at_rank(row, rank)
    if value:
        return value
    return det.label


def _class_for_original(
    det: Detection,
    taxonomy_lookup: dict[str, LabelTaxonomy],
    rank: Rank,
    *,
    has_classifier: bool,
) -> str | None:
    """
    Predicted class at the requested rank, or None when we genuinely
    don't know what the model said.

    Detector-only projects never ran a classifier, so an unclassified
    animal is a valid 'animal' prediction. Classifier-enabled projects
    with no original_label indicate pre-migration data — those we skip.
    """
    if det.category in ("person", "vehicle"):
        return det.category
    if not det.original_label:
        if not has_classifier:
            return UNCLASSIFIED
        return None
    row = taxonomy_lookup.get(det.original_label.lower())
    value = _taxon_at_rank(row, rank)
    if value:
        return value
    return det.original_label


def _ordered_classes(
    all_classes: set[str], row_totals: dict[str, int],
) -> list[str]:
    """
    Stable ordering: detector categories first (if they appeared), then
    the rest by descending support with alphabetical tiebreaker.
    """
    head = [c for c in DETECTOR_CATEGORIES if c in all_classes]
    rest = sorted(
        (c for c in all_classes if c not in head),
        key=lambda c: (-row_totals.get(c, 0), c.lower()),
    )
    return head + rest


def _apply_top_n(
    ordered: list[str],
    counts: Counter,
    row_totals: dict[str, int],
    top_n: int | None,
) -> tuple[list[str], Counter, bool]:
    """
    Keep the top-N classes by row support; collapse the rest into a
    single 'other' row and column so matrix totals stay conserved.

    Detector categories in the fixed head are always kept.
    """
    if top_n is None or len(ordered) <= top_n:
        return ordered, counts, False

    head = [c for c in ordered if c in DETECTOR_CATEGORIES]
    remaining_budget = max(top_n - len(head), 0)
    tail = [c for c in ordered if c not in head]
    # tail is already sorted by descending support
    kept_tail = tail[:remaining_budget]
    dropped = set(tail[remaining_budget:])
    if not dropped:
        return ordered, counts, False

    kept = head + kept_tail + [OTHER_BUCKET]

    folded: Counter = Counter()
    for (true_c, pred_c), n in counts.items():
        t = OTHER_BUCKET if true_c in dropped else true_c
        p = OTHER_BUCKET if pred_c in dropped else pred_c
        folded[(t, p)] += n

    return kept, folded, True


def _harmonic_mean(
    precision: float | None, recall: float | None,
) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def _macro(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _weighted(
    values: list[float | None], weights: list[int],
) -> float | None:
    total_weight = 0
    acc = 0.0
    for v, w in zip(values, weights, strict=True):
        if v is None or w <= 0:
            continue
        acc += v * w
        total_weight += w
    if total_weight == 0:
        return None
    return acc / total_weight


def get_classification_performance(
    db: Session,
    project_id: str,
    *,
    site_ids: list[str] | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    rank: Rank = "species",
    top_n: int | None = 20,
) -> PerformanceResponse:
    """
    Build the confusion matrix + metrics for the given project.

    Ground truth = verified detections' current label. Prediction =
    their original machine label. Detections with no prediction (NULL
    original_label on animal detections, typically pre-migration rows)
    are excluded and surfaced in `skipped_no_prediction`.
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    has_classifier = project.classification_model_id is not None

    q = (
        db.query(Detection)
        .join(File, File.id == Detection.file_id)
        .join(Deployment, Deployment.id == File.deployment_id)
        .filter(Deployment.project_id == project_id)
    )
    if site_ids:
        q = q.filter(Deployment.site_id.in_(site_ids))
    if date_from is not None:
        q = q.filter(File.captured_at_local >= date_from)
    if date_to is not None:
        q = q.filter(File.captured_at_local <= date_to)

    all_detections: list[Detection] = q.all()
    skipped_unverified = sum(1 for d in all_detections if not d.verified)
    verified_detections = [d for d in all_detections if d.verified]

    taxonomy_lookup = _build_taxonomy_lookup(db, project)

    counts: Counter = Counter()
    row_totals: dict[str, int] = defaultdict(int)
    skipped_no_prediction = 0

    for det in verified_detections:
        predicted = _class_for_original(
            det, taxonomy_lookup, rank, has_classifier=has_classifier,
        )
        if predicted is None:
            skipped_no_prediction += 1
            continue
        true_c = _class_for_current(det, taxonomy_lookup, rank)
        counts[(true_c, predicted)] += 1
        row_totals[true_c] += 1

    all_classes: set[str] = set()
    for (t, p) in counts:
        all_classes.add(t)
        all_classes.add(p)

    ordered = _ordered_classes(all_classes, row_totals)

    ordered, counts, other_bucket_present = _apply_top_n(
        ordered, counts, row_totals, top_n,
    )

    # Rebuild totals from the (possibly folded) counts.
    matrix = [[0] * len(ordered) for _ in ordered]
    idx = {c: i for i, c in enumerate(ordered)}
    for (t, p), n in counts.items():
        matrix[idx[t]][idx[p]] = n

    row_totals_list = [sum(row) for row in matrix]
    col_totals_list = [sum(matrix[r][c] for r in range(len(ordered))) for c in range(len(ordered))]
    grand_total = sum(row_totals_list)

    display_lookup: dict[str, str] = {}
    taxonomy_id_lookup: dict[str, str | None] = {}
    for c in ordered:
        row = (
            None
            if c in DETECTOR_CATEGORIES or c == OTHER_BUCKET
            else taxonomy_lookup.get(c.lower())
        )
        display_lookup[c] = _display_for(c, row)
        taxonomy_id_lookup[c] = row.id if row is not None else None

    per_class: list[ClassMetrics] = []
    for i, c in enumerate(ordered):
        support = row_totals_list[i]
        tp = matrix[i][i]
        col_total = col_totals_list[i]
        precision = tp / col_total if col_total > 0 else None
        recall = tp / support if support > 0 else None
        f1 = _harmonic_mean(precision, recall)
        per_class.append(
            ClassMetrics(
                class_name=c,
                display_name=display_lookup[c],
                support=support,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )

    precisions = [m.precision for m in per_class]
    recalls = [m.recall for m in per_class]
    f1s = [m.f1 for m in per_class]
    supports = [m.support for m in per_class]

    return PerformanceResponse(
        rank=rank,
        classes=ordered,
        class_display_names=[display_lookup[c] for c in ordered],
        class_taxonomy_ids=[taxonomy_id_lookup[c] for c in ordered],
        matrix=matrix,
        row_totals=row_totals_list,
        col_totals=col_totals_list,
        grand_total=grand_total,
        per_class=per_class,
        macro_precision=_macro(precisions),
        macro_recall=_macro(recalls),
        macro_f1=_macro(f1s),
        weighted_precision=_weighted(precisions, supports),
        weighted_recall=_weighted(recalls, supports),
        weighted_f1=_weighted(f1s, supports),
        skipped_no_prediction=skipped_no_prediction,
        skipped_unverified=skipped_unverified,
        has_classifier=has_classifier,
        top_n_applied=top_n,
        other_bucket_present=other_bucket_present,
    )
