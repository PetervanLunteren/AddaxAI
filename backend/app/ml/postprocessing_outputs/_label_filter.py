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
- A file with ALL its passing *identified* detections excluded is
  dropped entirely. A file with some excluded and some included
  identified detections survives, but only the included labels
  contribute to placements / EXIF tags / rows.
- "Identified" means the detection carries a label or a
  ``label_taxonomy_id``. Person and vehicle detections carry the
  builtin person / vehicle taxonomy id (see ``ensure_builtin_labels``),
  so they ARE filterable: deselecting Person in the Save-step filter
  drops person files from the media copies. Unclassified-animal
  detections carry the builtin "animal" id and are filterable the
  same way.
- True blanks (no identified detection at all) are never dropped by
  the filter; the Save step's "copy empties" toggle governs those.

Centralising the logic here keeps the seven postprocess modules
(separate, visualise, blur, exif, csv, xlsx, recognition_json) in
sync: changing the rule means changing one module, not seven.
"""

from __future__ import annotations

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.ml.detection_visibility import on_visible_frame_of
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
    strongest first.

    A "passing" detection is one over the project threshold (or
    verified) AND not in the user's exclusion set.

    Strongest is verified first, then confidence, matching
    ``derive_observation_type`` and ``build_event_primary_labels``. So
    ``[0]`` is the detection that decides what the file is.
    """
    rows = db.execute(
        select(Detection)
        .where(Detection.file_id == file.id)
        # A video is written to disk as its best-frame JPEG, so only that
        # frame's detections may decide where the picture is filed. A box
        # on some other frame naming the folder is the same bug as a box
        # on some other frame being drawn on it.
        .where(on_visible_frame_of(file))
        .where(
            or_(
                Detection.confidence >= threshold,
                Detection.verified == True,  # noqa: E712
            )
        )
        .order_by(
            Detection.verified.desc(),
            Detection.confidence.desc(),
        )
    ).scalars().all()

    if not excluded_label_ids:
        return list(rows)
    return [
        r for r in rows
        if not detection_is_excluded(r, excluded_label_ids)
    ]


def strongest_label_for_file(
    db: Session,
    file: File,
    threshold: float,
    excluded_label_ids: frozenset[str] | None = None,
) -> str | None:
    """The species label of the file's strongest passing detection, or
    ``None`` when that detection carries no species.

    This deliberately reads only the *strongest* detection rather than
    the strongest *labelled* one. A clip whose best box is a person is a
    person clip, even when a weaker animal box happens to carry a
    species: taking the best label instead of the best detection is what
    filed a person in camouflage under ``chimpanzee/`` off one
    false-positive box the classifier guessed at 29%.

    ``None`` therefore means "this file is not a species", and the
    caller names it by its detector category instead.
    """
    passing = passing_detections_for_file(
        db, file, threshold, excluded_label_ids
    )
    return passing[0].label if passing else None


def file_is_dropped_by_filter(
    db: Session,
    file: File,
    threshold: float,
    excluded_label_ids: frozenset[str] | None,
) -> bool:
    """True when ALL of a file's passing, identified detections are excluded.

    Identified = the detection carries a ``label`` or a
    ``label_taxonomy_id``, which covers species labels and the builtin
    animal / person / vehicle ids alike. A file whose every identified
    detection is excluded is dropped from the media outputs. A file with
    no identified detection (a true blank) is never dropped here — the
    "copy empties" toggle owns those.
    """
    if not excluded_label_ids:
        return False

    rows = db.execute(
        select(Detection)
        .where(Detection.file_id == file.id)
        # Same visible surface as everything else. The label filter only
        # offers labels the user can see, so only those may drop a file;
        # otherwise a video could be dropped by a label that never
        # appeared in the filter and could not be unticked.
        .where(on_visible_frame_of(file))
        .where(
            or_(
                Detection.label.isnot(None),
                Detection.label_taxonomy_id.isnot(None),
            )
        )
        .where(
            or_(
                Detection.confidence >= threshold,
                Detection.verified == True,  # noqa: E712
            )
        )
    ).scalars().all()

    if not rows:
        # No identified detection → true blank / unidentified. Filter
        # doesn't apply; copy-empties governs these.
        return False
    return all(
        detection_is_excluded(det, excluded_label_ids) for det in rows
    )
