"""Shared label-exclusion filter for the Save-outputs step.

Every postprocess module follows the same rule when the user has
picked an exclusion set in the Save step's filter card:

- The filter is label-level. The exclusion set is a heterogeneous
  list of identifiers: each entry is either a ``LabelTaxonomy.id``
  (UUID, for labels that have a taxonomy mapping) or a raw
  ``Detection.label`` string (for labels that don't). The
  frontend's label-tree modal emits both kinds — UUIDs for mapped
  leaves, raw names for the "Other" branch.
- A detection is excluded when its ``label_taxonomy_id`` is in the
  set OR its ``label`` string is in the set.
- For animal files: a file with ALL its passing labelled
  detections excluded is dropped entirely. A file with some
  excluded and some included labels survives, but only the
  included labels contribute to placements / EXIF tags / rows.
- Non-animal files (observation_type human/vehicle/blank/...) are
  never affected by the filter. They are detector-level, not
  classifier-level, and the filter operates on classifier labels.

Centralising the logic here keeps the seven postprocess modules
(separate, visualise, blur, exif, csv, xlsx, recognition_json) in
sync: changing the rule means changing one module, not seven.
"""

from __future__ import annotations

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.models import Detection, File


def detection_is_excluded(
    detection: Detection,
    excluded_label_ids: frozenset[str] | None,
) -> bool:
    """Return True when the user's exclusion set covers this detection.

    Either the detection's taxonomy id or its raw label string can
    match — the set is heterogeneous (UUIDs for taxonomy-mapped
    labels, plain strings for unmapped ones from the "Other"
    branch of the label tree).
    """
    if not excluded_label_ids:
        return False
    if (
        detection.label_taxonomy_id
        and detection.label_taxonomy_id in excluded_label_ids
    ):
        return True
    if detection.label and detection.label in excluded_label_ids:
        return True
    return False


def passing_detections_for_file(
    db: Session,
    file: File,
    threshold: float,
    excluded_label_ids: frozenset[str] | None = None,
) -> list[Detection]:
    """Return the file's passing detections with exclusion applied,
    ordered by confidence descending.

    A "passing" detection is one over the project threshold (or
    verified) AND not in the user's exclusion set.
    """
    rows = db.execute(
        select(Detection)
        .where(Detection.file_id == file.id)
        .where(
            or_(
                Detection.confidence >= threshold,
                Detection.verified == True,  # noqa: E712
            )
        )
        .order_by(Detection.confidence.desc())
    ).scalars().all()

    if not excluded_label_ids:
        return list(rows)
    return [
        r for r in rows
        if not detection_is_excluded(r, excluded_label_ids)
    ]


def passing_labels_for_file(
    db: Session,
    file: File,
    threshold: float,
    excluded_label_ids: frozenset[str] | None = None,
) -> list[str]:
    """Return the file's passing label strings (deduped, order-preserved
    by descending confidence), with exclusion applied. Unlabelled
    detections (no species attribution) are skipped.
    """
    seen: set[str] = set()
    out: list[str] = []
    for det in passing_detections_for_file(
        db, file, threshold, excluded_label_ids
    ):
        if det.label and det.label not in seen:
            seen.add(det.label)
            out.append(det.label)
    return out


def file_is_dropped_by_filter(
    db: Session,
    file: File,
    threshold: float,
    excluded_label_ids: frozenset[str] | None,
) -> bool:
    """True when an animal file's labelled detections are all excluded.

    Non-animal files are never dropped by the filter. Animal files
    with no labelled detections (those that fall back to the
    ``animal/`` folder) are NOT dropped either — they have no
    species label to match against the exclusion set.
    """
    if not excluded_label_ids:
        return False
    if file.observation_type != "animal":
        return False

    rows = db.execute(
        select(Detection)
        .where(Detection.file_id == file.id)
        .where(Detection.label.isnot(None))
        .where(
            or_(
                Detection.confidence >= threshold,
                Detection.verified == True,  # noqa: E712
            )
        )
    ).scalars().all()

    if not rows:
        # No species labels at all → fall-back animal/ file. Filter
        # doesn't apply.
        return False
    return all(
        detection_is_excluded(det, excluded_label_ids) for det in rows
    )
