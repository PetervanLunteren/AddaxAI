"""
CRUD for the confusion matrix + classification report insight views.

Per-detection comparison of the machine's final label
(Detection.original_label, i.e. the label the UI showed after rollup /
smoothing) vs current label (after human verification or relabel). Only
verified detections count, so a label the human confirmed as-is lands on
the diagonal and a relabel lands off it.
Metrics are computed server-side so the React page stays thin and the
math is unit-testable in isolation.

Taxonomic rank resolution is shared with the dashboard via
`app.ml.taxonomic_rank.resolve_rank` so both views produce the same
"Higher-level taxa" and "No taxonomy" buckets from the same rules.
"""

from collections import Counter, defaultdict
from datetime import date

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.api.schemas.performance import ClassMetrics, PerformanceResponse
from app.ml.detection_visibility import on_visible_frame
from app.ml.taxonomic_rank import (
    HIGHER_LEVEL_TAXA,
    MOST_SPECIFIC,
    NO_TAXONOMY,
    TaxonomicRank,
    resolve_rank,
)
from app.models.deployment import Deployment
from app.models.detection import Detection
from app.models.file import File
from app.models.label_taxonomy import LabelTaxonomy
from app.models.project import Project

DETECTOR_CATEGORIES: tuple[str, ...] = ("animal", "person", "vehicle")
OTHER_BUCKET = "other"
# Rank-resolver buckets (not detector categories) kept fixed at the
# bottom of the class ordering and exempt from top-N collapse.
SEMANTIC_BUCKETS: tuple[str, ...] = (HIGHER_LEVEL_TAXA, NO_TAXONOMY)


def _display_for(name: str, row: LabelTaxonomy | None) -> str:
    """
    Human-friendly label for the matrix axis.

    Detector categories and the semantic buckets pass through as-is;
    species fall back to the taxonomy row's scientific_name when present.
    Non-species names (family / genus / order / class) are already
    capitalised by resolve_rank, so they reach here display-ready.
    """
    if name == OTHER_BUCKET:
        return "Other"
    if name in DETECTOR_CATEGORIES:
        return name
    if name in SEMANTIC_BUCKETS:
        return name
    if row is not None and row.scientific_name:
        return row.scientific_name
    return name.replace("_", " ")


def _common_for(name: str, row: LabelTaxonomy | None) -> str:
    """Common-name counterpart of `_display_for` for the matrix axis."""
    if name == OTHER_BUCKET:
        return "Other"
    if name in DETECTOR_CATEGORIES:
        return name
    if name in SEMANTIC_BUCKETS:
        return name
    if row is not None and row.common_name:
        return row.common_name
    return name.replace("_", " ")


def _build_taxonomy_lookup(
    db: Session, project: Project,
) -> dict[str, LabelTaxonomy]:
    """
    Map label name (lowercased) to the taxonomy row used to roll it up.

    Includes rows scoped to the project's classification model plus any
    project-scoped custom rows. Detector-only projects have no classifier
    rows, so the lookup is empty — every animal detection then lands in
    either "animal" (at "all" rank) or "No taxonomy" (at a specific rank).
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
    det: Detection,
    taxonomy_lookup: dict[str, LabelTaxonomy],
    rank: TaxonomicRank,
) -> str:
    """Current (ground-truth) class at the requested rank."""
    row = taxonomy_lookup.get(det.label.lower()) if det.label else None
    return resolve_rank(
        category=det.category,
        label=det.label,
        scientific_name=det.scientific_name,
        taxonomy_row=row,
        rank=rank,
    )


def _class_for_original(
    det: Detection,
    taxonomy_lookup: dict[str, LabelTaxonomy],
    rank: TaxonomicRank,
    *,
    has_classifier: bool,
) -> str | None:
    """
    Predicted class at the requested rank, or None when we genuinely
    don't know what the model said.

    Detector-only projects never ran a classifier, so an unclassified
    animal is a valid prediction (resolve_rank turns category="animal"
    into "animal" at rank="all" and "No taxonomy" at specific ranks).
    Classifier-enabled projects with a NULL original_label on an animal
    indicate pre-migration data and are skipped instead of resolved.
    """
    if det.category in ("person", "vehicle"):
        return det.category
    if not det.original_label:
        if has_classifier:
            return None
        return resolve_rank(
            category=det.category,
            label=None,
            scientific_name=None,
            taxonomy_row=None,
            rank=rank,
        )
    row = taxonomy_lookup.get(det.original_label.lower())
    return resolve_rank(
        category=det.category,
        label=det.original_label,
        scientific_name=row.scientific_name if row is not None else None,
        taxonomy_row=row,
        rank=rank,
    )


def _ordered_classes(
    all_classes: set[str], row_totals: dict[str, int],
) -> list[str]:
    """
    Stable ordering: detector head (animal / person / vehicle), then
    real classes by descending support with alphabetical tiebreaker,
    then the semantic buckets (Higher-level taxa / No taxonomy) pinned
    to the bottom.
    """
    head = [c for c in DETECTOR_CATEGORIES if c in all_classes]
    buckets = [c for c in SEMANTIC_BUCKETS if c in all_classes]
    rest = sorted(
        (c for c in all_classes if c not in head and c not in buckets),
        key=lambda c: (-row_totals.get(c, 0), c.lower()),
    )
    return head + rest + buckets


def _apply_top_n(
    ordered: list[str],
    counts: Counter,
    row_totals: dict[str, int],
    top_n: int | None,
) -> tuple[list[str], Counter, bool]:
    """
    Keep the top-N classes by row support; fold the rest into a single
    'other' row + column. Detector categories and the semantic buckets
    are exempt and always kept.
    """
    exempt = set(DETECTOR_CATEGORIES) | set(SEMANTIC_BUCKETS)
    real = [c for c in ordered if c not in exempt]
    if top_n is None or len(real) <= top_n:
        return ordered, counts, False

    kept_real = real[:top_n]
    dropped = set(real[top_n:])
    if not dropped:
        return ordered, counts, False

    head = [c for c in ordered if c in DETECTOR_CATEGORIES]
    buckets = [c for c in ordered if c in SEMANTIC_BUCKETS]
    kept = head + kept_real + [OTHER_BUCKET] + buckets

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
    taxonomic_rank: TaxonomicRank = MOST_SPECIFIC,
    top_n: int | None = 20,
) -> PerformanceResponse:
    """
    Build the confusion matrix + metrics for the given project.

    Ground truth = verified detections' current label. Prediction =
    the machine's final label (Detection.original_label, what the UI
    showed after rollup / smoothing). Detections with no prediction (NULL
    original_label on animal detections, typically pre-column rows)
    are excluded and surfaced in `skipped_no_prediction`.
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    has_classifier = project.classification_model_id is not None
    threshold = project.counting_threshold

    # User-facing scope: only detections that would appear in the Verify
    # page (confidence at or above the project threshold, OR already
    # verified so the override applies). Sub-threshold detections
    # cannot be verified by the user because they never surface in the
    # UI, so counting them as "unverified" is misleading.
    # See DEVELOPERS.md section "Detection threshold and verified override".
    #
    # A video's off-best-frame boxes are that same case and need the same
    # gate: only one frame per video is written to disk, so a box on any
    # other frame has no picture to open and can never be verified. The
    # footer used to read "1 verified detection of 220 ... 218 not yet
    # verified" over a grid holding 32.
    q = (
        db.query(Detection)
        .join(File, File.id == Detection.file_id)
        .join(Deployment, Deployment.id == File.deployment_id)
        .filter(Deployment.project_id == project_id)
        .filter(
            or_(
                Detection.confidence >= threshold,
                Detection.verified == True,  # noqa: E712
            )
        )
        .filter(on_visible_frame())
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
            det, taxonomy_lookup, taxonomic_rank, has_classifier=has_classifier,
        )
        if predicted is None:
            skipped_no_prediction += 1
            continue
        true_c = _class_for_current(det, taxonomy_lookup, taxonomic_rank)
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

    matrix = [[0] * len(ordered) for _ in ordered]
    idx = {c: i for i, c in enumerate(ordered)}
    for (t, p), n in counts.items():
        matrix[idx[t]][idx[p]] = n

    row_totals_list = [sum(row) for row in matrix]
    col_totals_list = [
        sum(matrix[r][c] for r in range(len(ordered)))
        for c in range(len(ordered))
    ]
    grand_total = sum(row_totals_list)

    # At the "all" / "species" rank, resolve_rank returns the
    # scientific_name as the class identifier (e.g. "S. carolinensis"),
    # which is NOT a LabelTaxonomy.name, so the by-name lookup below
    # misses and the common name can't be found. This reverse map lets
    # us recover the row (and its common_name) from that scientific
    # string. At family / order / class ranks the class is a taxon name
    # that already matches LabelTaxonomy.name, so the by-name lookup hits.
    #
    # Abbreviated scientific names collide across the model's full
    # vocabulary (Sciurus carolinensis and Sitta carolinensis are both
    # "S. carolinensis"), and the loser of the dict overwrite lends its
    # common name to the winner's class. Rows whose label actually
    # occurs in this project's verified detections win the key, so a
    # collision with a species that never appears here cannot mislabel
    # the axis.
    observed_labels = {
        d.label.lower() for d in verified_detections if d.label
    } | {
        d.original_label.lower() for d in verified_detections if d.original_label
    }
    by_scientific: dict[str, LabelTaxonomy] = {}
    for r in taxonomy_lookup.values():
        if not r.scientific_name:
            continue
        key = r.scientific_name.lower()
        if key not in by_scientific or r.name.lower() in observed_labels:
            by_scientific[key] = r

    display_lookup: dict[str, str] = {}
    common_lookup: dict[str, str] = {}
    taxonomy_id_lookup: dict[str, str | None] = {}
    for c in ordered:
        # Semantic buckets and detector categories have no taxonomy row.
        row = (
            None
            if c in DETECTOR_CATEGORIES or c == OTHER_BUCKET or c in SEMANTIC_BUCKETS
            else taxonomy_lookup.get(c.lower()) or by_scientific.get(c.lower())
        )
        display_lookup[c] = _display_for(c, row)
        common_lookup[c] = _common_for(c, row)
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
                common_name=common_lookup[c],
                scientific_name=display_lookup[c],
                support=support,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )

    # Macro / weighted averages are meant to summarise the AI's
    # per-class performance. Detector categories reflect the detector,
    # not the classifier; semantic buckets and the top-N "other" row
    # are not classes at all. Excluding all of them keeps the averages
    # honest.
    averaged_over = {
        c for c in ordered
        if c not in DETECTOR_CATEGORIES
        and c not in SEMANTIC_BUCKETS
        and c != OTHER_BUCKET
    }
    metrics_for_avg = [m for m in per_class if m.class_name in averaged_over]
    precisions = [m.precision for m in metrics_for_avg]
    recalls = [m.recall for m in metrics_for_avg]
    f1s = [m.f1 for m in metrics_for_avg]
    supports = [m.support for m in metrics_for_avg]

    return PerformanceResponse(
        taxonomic_rank=taxonomic_rank,
        classes=ordered,
        class_scientific_names=[display_lookup[c] for c in ordered],
        class_common_names=[common_lookup[c] for c in ordered],
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
