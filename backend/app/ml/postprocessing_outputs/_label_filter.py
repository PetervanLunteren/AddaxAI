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
from app.ml.label_exclusion import is_a_real_detection, threshold_or_verified
from app.ml.label_filter_ids import label_matches_filter
from app.models import Detection, File


def detection_is_excluded(
    detection: Detection,
    excluded_label_ids: frozenset[str] | None,
) -> bool:
    """Return True when the user's exclusion set covers this detection.

    Taxonomy UUIDs match the foreign key. New unmapped-label tokens
    match the effective raw label only when the taxonomy key is NULL;
    older plain-string exclusions retain their previous string matching.
    """
    return label_matches_filter(
        detection.label,
        detection.label_taxonomy_id,
        excluded_label_ids or (),
        detection.category,
    )


def passing_detections_for_file(
    db: Session,
    file: File,
    threshold: float,
    excluded_label_ids: frozenset[str] | None = None,
) -> list[Detection]:
    """Return the file's passing detections with exclusion applied,
    strongest first.

    A "passing" detection is one over the project threshold (or
    verified) AND a real observation AND not in the user's exclusion
    set. The ``is_a_real_detection`` clause is what keeps a rejected box
    ("false detection", verified first in the ordering) from deciding a
    file's folder: without it, one X press filed the whole picture under
    ``false detection/`` while ``derive_observation_type`` skipped the
    same box and called the file blank.

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
        .where(threshold_or_verified(threshold))
        .where(is_a_real_detection())
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

    Identified = the detection carries a ``label``, a category, or a
    ``label_taxonomy_id``, which covers unmapped detector classes and
    builtin animal / person / vehicle categories alike. A file whose every identified
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
                Detection.category.isnot(None),
                Detection.label_taxonomy_id.isnot(None),
            )
        )
        .where(threshold_or_verified(threshold))
    ).scalars().all()

    if not rows:
        # No identified detection → true blank / unidentified. Filter
        # doesn't apply; copy-empties governs these.
        return False
    return all(
        detection_is_excluded(det, excluded_label_ids) for det in rows
    )
