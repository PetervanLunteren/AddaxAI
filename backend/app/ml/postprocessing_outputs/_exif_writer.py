"""Shared EXIF / XMP writer for the postprocessing modules.

Embeds detection summaries into files so the labels travel with the
image: a viewer like Lightroom or digiKam reads the XMP:Subject tag
as the file's "tags", standard EXIF viewers read ImageDescription as
the caption. The JSON blob in UserComment carries the full detection
breakdown for downstream scripts that want machine-readable input
without re-running AddaxAI.

Uses the same `exiftool` binary that's already on the deploy via the
`PyExifTool` runtime dep (see backend/app/utils/media_dates.py for
the read-side use). One ExifToolHelper per batch via the
``ExifBatch`` context manager so the underlying exiftool process
stays alive across all writes in a loop — process start-up is the
expensive part, the actual writes are cheap.

Symlinks are skipped at the caller. Videos accept the same tags as
JPEGs through exiftool's container-aware writing (XMP under
QuickTime for MP4 / MOV); we don't gate on file type here, the
caller decides what makes sense.
"""

from __future__ import annotations

import json
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import exiftool
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.models import Detection, File, Project

logger = get_logger(__name__)


@dataclass(frozen=True)
class DetectionTagSet:
    """Pre-computed tags for one file. Built once per file, then
    passed to ``ExifBatch.write`` for the actual exiftool call."""

    image_description: str
    species_tags: tuple[str, ...]
    software: str
    user_comment_json: str

    def to_exiftool_dict(self) -> dict[str, object]:
        """Translate into the keys ``exiftool.ExifToolHelper.set_tags``
        understands. XMP:Subject is the field Lightroom and digiKam
        read as "tags"."""
        tags: dict[str, object] = {
            "EXIF:ImageDescription": self.image_description,
            "XMP-dc:Description": self.image_description,
            "EXIF:Software": self.software,
            "EXIF:UserComment": self.user_comment_json,
        }
        if self.species_tags:
            tags["XMP-dc:Subject"] = list(self.species_tags)
        return tags


def build_tag_set(
    db: Session,
    file: File,
    project: Project,
    app_version: str,
    *,
    excluded_label_ids: frozenset[str] | None = None,
) -> DetectionTagSet | None:
    """Build the tag set for one file, threshold-aware.

    Returns ``None`` when the file has no detections above threshold
    (or no detections at all) — no point in writing empty metadata.

    When ``excluded_label_ids`` is set, labelled detections in the set
    are filtered out of the tag set. An unlabelled detection (the
    raw animal / person / vehicle category without species info) is
    never excluded by the species filter; only labelled detections
    are subject to it.
    """
    threshold = project.detection_threshold
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

    if excluded_label_ids:
        # Same filter rule as _label_filter.detection_is_excluded —
        # match by taxonomy id OR label string so the heterogeneous
        # exclusion set (UUIDs for mapped, strings for unmapped)
        # behaves consistently across modules.
        def _excluded(det: Detection) -> bool:
            if det.label_taxonomy_id and det.label_taxonomy_id in excluded_label_ids:
                return True
            if det.label and det.label in excluded_label_ids:
                return True
            return False

        rows = [r for r in rows if not _excluded(r)]

    if not rows:
        return None

    # Short summary, capped at the top 5 so ImageDescription stays
    # readable in a thumbnail viewer. Both names are written for
    # consistency (independent of any UI display preference): when a
    # detection has distinct common + scientific names they read as
    # "Eastern gray squirrel (S. carolinensis) 95%"; when they coincide
    # (rollups, builtins) the single name is shown once.
    summary_parts: list[str] = []
    # XMP:Subject tags — searchable in Lightroom / digiKam. Include both
    # the common and scientific name so the image is findable by either.
    species_seen: list[str] = []
    species_set: set[str] = set()

    def _add_tag(value: str | None) -> None:
        if not value:
            return
        cleaned = value.strip()
        if cleaned and cleaned.lower() not in {s.lower() for s in species_set}:
            species_set.add(cleaned)
            species_seen.append(cleaned)

    for det in rows[:5]:
        name = det.label or det.category or "unknown"
        fallback = name[0].upper() + name[1:] if name else "Unknown"
        pct = int(round(det.confidence * 100))
        common = det.common_name or fallback
        scientific = det.scientific_name or fallback
        if common.lower() == scientific.lower():
            summary_parts.append(f"{common} {pct}%")
        else:
            summary_parts.append(f"{common} ({scientific}) {pct}%")
        if det.label:
            _add_tag(det.common_name)
            _add_tag(det.scientific_name)
    if len(rows) > 5:
        summary_parts.append(f"+ {len(rows) - 5} more")
    image_description = ", ".join(summary_parts)

    # Software tag: app version + model lineage. Standard place to
    # record what produced the metadata.
    model_label = project.detection_model_id
    if project.classification_model_id:
        model_label = f"{model_label} + {project.classification_model_id}"
    software = f"AddaxAI {app_version} ({model_label})"

    # Full structured payload in UserComment for downstream scripts
    # that don't want to parse the summary back.
    detections_payload = [
        {
            "category": det.category,
            "label": det.label,
            "common_name": det.common_name,
            "scientific_name": det.scientific_name,
            "confidence": round(float(det.confidence), 4),
            "label_confidence": (
                round(float(det.label_confidence), 4)
                if det.label_confidence is not None
                else None
            ),
            "verified": bool(det.verified),
        }
        for det in rows
    ]
    user_comment = {
        "app": f"AddaxAI {app_version}",
        "detection_model": project.detection_model_id,
        "classification_model": project.classification_model_id,
        "detection_threshold": float(project.detection_threshold),
        "detections": detections_payload,
    }
    user_comment_json = json.dumps(user_comment, separators=(",", ":"))

    return DetectionTagSet(
        image_description=image_description,
        species_tags=tuple(species_seen),
        software=software,
        user_comment_json=user_comment_json,
    )


class ExifBatch(AbstractContextManager["ExifBatch"]):
    """Context manager owning one ExifToolHelper for a batch of writes.

    Use as::

        with ExifBatch() as batch:
            for file in files:
                batch.write(path, tag_set)

    The underlying exiftool process is started once on ``__enter__``
    and torn down on ``__exit__``. ``write`` raises on failure; the
    caller catches per-file and records the error.
    """

    def __init__(self) -> None:
        self._helper: exiftool.ExifToolHelper | None = None

    def __enter__(self) -> ExifBatch:
        self._helper = exiftool.ExifToolHelper()
        self._helper.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._helper is not None:
            self._helper.__exit__(exc_type, exc, tb)
            self._helper = None

    def write(self, path: Path, tag_set: DetectionTagSet) -> None:
        """Write the tag set onto the file at ``path``.

        ``-overwrite_original`` prevents exiftool from leaving a
        ``<filename>_original`` backup next to every modified file
        (the backups would clutter the output and double disk use).
        """
        if self._helper is None:
            raise RuntimeError("ExifBatch.write called outside its context")
        self._helper.set_tags(
            [str(path)],
            tags=tag_set.to_exiftool_dict(),
            params=["-overwrite_original"],
        )


def is_image_path(path: Path) -> bool:
    """Cheap suffix check so callers can decide whether to write EXIF.

    Video EXIF writing technically works (exiftool handles MP4/MOV)
    but the postprocess modules treat videos as best-frame JPEGs for
    visualised / blur output. Source-video writes are gated by the
    caller; this helper is the one place that defines what counts as
    a writable image format here.
    """
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
