"""
Undo: snapshot the rows an action is about to change, and put them back.

See ``app/api/schemas/undo.py`` for the shape and the reasoning. The
detection endpoints call ``snapshot`` before they write; ``set_file_verified``
builds its own snapshot because only it knows which boxes a file sign-off
touches. ``restore`` is the one writer behind ``POST /api/undo``.
"""

from collections.abc import Iterable
from typing import NamedTuple, TypeVar

from sqlalchemy.orm import Session

from app.api.crud.detection import _resolve_detection_taxonomy
from app.api.crud.event_observation import (
    get_event_ids_for_files,
    recalculate_max_n_for_events,
)
from app.api.crud.file import _project_threshold_for_file, _set_observation_type
from app.api.schemas.undo import DetectionState, FileState, UndoSnapshot
from app.db.sql_params import iter_id_chunks
from app.ml.taxonomy_db import clear_classification, ensure_builtin_labels
from app.models import Detection, File

_Row = TypeVar("_Row", Detection, File)


def snapshot(db: Session, detections: list[Detection]) -> UndoSnapshot:
    """The before-state of these boxes and of the files they sit on.

    The files are included because verifying or relabelling a box can flip
    its file's ``verified`` flag through ``recompute_file_verified``, and
    undo has to put that back too. Take it before any write.
    """
    file_ids = list(dict.fromkeys(d.file_id for d in detections))
    files = _rows(db, File, file_ids)
    return UndoSnapshot(
        detections=[DetectionState.from_detection(d) for d in detections],
        files=[FileState.from_file(f) for f in files],
    )


def _rows(db: Session, model: type[_Row], ids: Iterable[str]) -> list[_Row]:
    """The rows with these ids, read in chunks: a snapshot of a tracked
    clip holds every box of its tracks, past SQLite's parameter limit."""
    rows: list[_Row] = []
    for chunk in iter_id_chunks(ids):
        rows.extend(db.query(model).filter(model.id.in_(chunk)).all())
    return rows


class Restored(NamedTuple):
    detections: list[Detection]
    file_count: int


def restore(db: Session, snap: UndoSnapshot) -> Restored:
    """Write a snapshot back, re-derive what hangs off it, and commit.

    Boxes and files get their snapshot values verbatim. ``label_taxonomy_id``
    is re-derived from the label (or, with no label, from the category's
    builtin row via ``clear_classification``), the same way relabel sets it.
    Then ``observation_type`` and the event MaxN are re-derived for every
    file involved, because both read ``verified``. ``recompute_file_verified``
    is deliberately not run: the snapshot's file flag is the truth, and the
    rollup would override it.

    Rows that no longer exist are skipped; a deleted file takes its boxes
    with it. The commit is unconditional (``get_db`` never commits).
    """
    det_ids = [s.detection_id for s in snap.detections]
    detections = _rows(db, Detection, det_ids)
    det_state = {s.detection_id: s for s in snap.detections}
    builtin_ids = ensure_builtin_labels(db) if detections else {}
    # One lookup per file and label, not per box: a track's boxes share
    # both, and a long clip's sign-off restores tens of thousands of them.
    # Keyed on the file because the label resolves within its project.
    taxonomy_ids: dict[tuple[str, str], str | None] = {}
    for det in detections:
        state = det_state[det.id]
        # Category first: clear_classification picks the builtin row from it.
        det.category = state.category
        if state.label is None:
            clear_classification(det, builtin_ids)
        else:
            det.label = state.label
            det.label_confidence = state.label_confidence
            det.scientific_name = state.scientific_name
            det.common_name = state.common_name
            key = (det.file_id, state.label)
            if key not in taxonomy_ids:
                taxonomy_ids[key] = _resolve_detection_taxonomy(db, det, state.label)
            det.label_taxonomy_id = taxonomy_ids[key]
        det.classification_method = state.classification_method
        det.verified = state.verified
        det.verified_at_utc = state.verified_at_utc

    file_ids = [s.file_id for s in snap.files]
    files = _rows(db, File, file_ids)
    file_state = {s.file_id: s for s in snap.files}
    for f in files:
        state = file_state[f.id]
        f.verified = state.verified
        f.verified_at_utc = state.verified_at_utc
    db.flush()

    touched = {f.id for f in files} | {d.file_id for d in detections}
    derive = _rows(db, File, touched)
    # One MaxN pass per event, at its own project's threshold.
    event_thresholds: dict[str, float] = {}
    for f in derive:
        threshold = _project_threshold_for_file(db, f)
        _set_observation_type(db, f, threshold)
        for event_id in get_event_ids_for_files(db, [f.id]):
            event_thresholds[event_id] = threshold
    for event_id, threshold in event_thresholds.items():
        recalculate_max_n_for_events(db, [event_id], threshold)
    db.commit()
    return Restored(detections=detections, file_count=len(files))
