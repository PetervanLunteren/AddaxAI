"""
Export CRUD — scoping query plus row / layer / CamTrap DP table builders.

The serializers in ``export_formats.py`` don't touch the DB; everything
here does. Row shapes line up 1:1 with the column orders declared in
``export_formats.py``.

Conventions
-----------

* ``Detection.confidence >= project.counting_threshold OR Detection.verified``
  is the user-facing filter (see DEVELOPERS.md "Detection threshold and
  verified override").
* ``Project.excluded_classes`` filters animal detections whose
  ``LabelTaxonomy.name`` matches (case-insensitive). Non-animal detections
  (person/vehicle) and files with zero in-scope detections survive so
  blank rows still fire.
* Observational datetimes (``File.captured_at_local``, ``Event.event_*_local``)
  are naive wall-clock values in ``Project.timezone`` and are formatted via
  ``to_local_iso_with_offset`` with an explicit timezone argument, so the
  formatted value carries a per-row UTC offset (DST-correct).
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Row, and_, func, or_, select, true
from sqlalchemy.orm import Session, defer

from app.api.crud.export_formats import slugify
from app.core.logging_config import get_logger
from app.db.sql_params import iter_id_chunks
from app.ml.detection_visibility import visible_detections
from app.ml.observation_type import strongest_passing_detection
from app.models import (
    Deployment,
    Detection,
    Event,
    EventObservation,
    File,
    LabelTaxonomy,
    Project,
    Site,
    event_files,
)
from app.utils.datetime_serialization import to_local_iso_with_offset

logger = get_logger(__name__)

# CamTrap DP v1.0 schema URLs (from the official TDWG repo)
CAMTRAP_DP_VERSION = "1.0"
CAMTRAP_DP_PROFILE = (
    f"https://raw.githubusercontent.com/tdwg/camtrap-dp/{CAMTRAP_DP_VERSION}"
    "/camtrap-dp-profile.json"
)
DEPLOYMENTS_SCHEMA = (
    f"https://raw.githubusercontent.com/tdwg/camtrap-dp/{CAMTRAP_DP_VERSION}"
    "/deployments-table-schema.json"
)
MEDIA_SCHEMA = (
    f"https://raw.githubusercontent.com/tdwg/camtrap-dp/{CAMTRAP_DP_VERSION}"
    "/media-table-schema.json"
)
OBSERVATIONS_SCHEMA = (
    f"https://raw.githubusercontent.com/tdwg/camtrap-dp/{CAMTRAP_DP_VERSION}"
    "/observations-table-schema.json"
)

# Flat detections CSV. One row per detection (the box grain). Lean on
# attributes (no time / place / paths — join to files.csv for those), but
# carries the full set of FK ids (file_id, deployment_id, event_id) so it
# links directly to every parent table without chained joins. Mirrors
# Camtrap-DP, whose observations table carries deploymentID / mediaID /
# eventID together.
#
# Column order follows the pipeline: ids, the detector stage (category +
# score), the classifier stage (label + score), the label's taxonomy /
# names, then geometry, then the verified flag.
#
#   detection_id    — the detection row id.
#   file_id         — FK to files.csv (time, place, paths live there).
#   deployment_id   — FK to deployments.csv (site, effort).
#   event_id        — FK to the event (also on files.csv).
#   detection_category — detector class: animal / person / vehicle.
#   detection_confidence — detector (MegaDetector) score for the box.
#   classification_label — the current species label, by model or human
#                     (provenance-neutral; see is_verified). Empty when
#                     nothing was classified (person, vehicle, unclassified).
#   classification_confidence — score for that label: the classifier's
#                     score, or 1.0 when a human assigned it.
#   ai_classification_label — the AI's final label (after geofence rollup
#                     and smoothing) = what the UI showed, kept even after
#                     a human relabel (so "AI said X, human changed to Y"
#                     is visible). Equals classification_label until a human
#                     changes it. The raw pre-rollup call is not exported;
#                     it stays in results.json. Empty for detector-only,
#                     person / vehicle, and pre-column detections.
#   ai_classification_confidence — the AI's score for that final call.
#   classification_method — who set the current label: machine or human.
#   is_verified     — this detection is human-verified (grouped with the
#                     label columns above so the provenance reads together).
#   taxon_*         — formal ranks from label_taxonomy; empty where the
#                     label has no (or partial) taxonomy.
#   frame_number    — video frame index; empty for images.
#   bbox_*          — normalized [0,1] box.
#
# Empty files do not appear here (a detections table holds detections);
# they live in files.csv. The per-event species count lives in counts.csv.
_FLAT_DETECTION_HEADERS = [
    "detection_id",
    "file_id",
    "deployment_id",
    "event_id",
    # Detector stage (MegaDetector): category + score.
    "detection_category",
    "detection_confidence",
    # Classifier stage: the current species label + score (may be a human
    # correction — see classification_method / is_verified).
    "classification_label",
    "classification_confidence",
    # The AI's original top-1 call, retained even after a human relabel, so
    # the export shows "AI said X, human changed to Y". Blank for
    # detector-only, person / vehicle, and pre-column detections.
    "ai_classification_label",
    "ai_classification_confidence",
    # Who set the current label, and whether a human confirmed it. Kept next
    # to the label columns so the provenance reads as one contiguous block.
    "classification_method",
    "is_verified",
    # Everything that describes the label: taxonomy broad -> specific, then
    # the two human-readable display names.
    "taxon_class",
    "taxon_order",
    "taxon_family",
    "taxon_genus",
    "taxon_species",
    "scientific_name",
    "common_name",
    # Geometry: video frame index, then the normalized [0,1] box.
    "frame_number",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
]

# CamTrap-DP 1.0 table schemas mandate all columns in a fixed order,
# even optional ones (the frictionless validator flags omitted columns
# as schema violations). We emit every column; unused optionals are
# blank. Column-index references elsewhere in this file assume this
# exact order.
_CAMTRAP_DEPLOYMENTS_HEADERS = [
    "deploymentID",          # 0  required
    "locationID",            # 1
    "locationName",          # 2
    "latitude",              # 3  required
    "longitude",             # 4  required
    "coordinateUncertainty", # 5
    "deploymentStart",       # 6  required
    "deploymentEnd",         # 7  required
    "setupBy",               # 8
    "cameraID",              # 9
    "cameraModel",           # 10
    "cameraDelay",           # 11
    "cameraHeight",          # 12
    "cameraDepth",           # 13
    "cameraTilt",            # 14
    "cameraHeading",         # 15
    "detectionDistance",     # 16
    "timestampIssues",       # 17
    "baitUse",               # 18
    "featureType",           # 19
    "habitat",               # 20
    "deploymentGroups",      # 21
    "deploymentTags",        # 22
    "deploymentComments",    # 23
]

_CAMTRAP_MEDIA_HEADERS = [
    "mediaID",          # 0  required
    "deploymentID",     # 1  required
    "captureMethod",    # 2
    "timestamp",        # 3  required
    "filePath",         # 4  required
    "filePublic",       # 5  required
    "fileName",         # 6
    "fileMediatype",    # 7  required
    "exifData",         # 8
    "favorite",         # 9
    "mediaComments",    # 10
]

_CAMTRAP_OBS_HEADERS = [
    "observationID",              # 0  required
    "deploymentID",               # 1  required
    "mediaID",                    # 2
    "eventID",                    # 3
    "eventStart",                 # 4  required
    "eventEnd",                   # 5  required
    "observationLevel",           # 6  required
    "observationType",            # 7  required
    "cameraSetupType",            # 8
    "scientificName",             # 9
    "count",                      # 10
    "lifeStage",                  # 11
    "sex",                        # 12
    "behavior",                   # 13
    "individualID",               # 14
    "individualPositionRadius",   # 15
    "individualPositionAngle",    # 16
    "individualSpeed",            # 17
    "bboxX",                      # 18
    "bboxY",                      # 19
    "bboxWidth",                  # 20
    "bboxHeight",                 # 21
    "classificationMethod",       # 22
    "classifiedBy",               # 23
    "classificationTimestamp",    # 24
    "classificationProbability",  # 25
    "observationTags",            # 26
    "observationComments",        # 27
]


# ---------------------------------------------------------------------------
# Scoping
# ---------------------------------------------------------------------------


def resolve_scope_deployment_ids(
    db: Session,
    project: Project,
    site_ids: list[str] | None,
    deployment_ids: list[str] | None,
) -> list[str] | None:
    """Resolve an export scope to a concrete list of deployment ids.

    Single source of truth for narrowing exports to part of a project.
    Returns ``None`` when no scope is given (export the whole project).
    Otherwise returns the project's deployments whose site is in
    ``site_ids`` OR whose id is in ``deployment_ids`` — so picking a site
    includes all its deployments. Ids from other projects are ignored by
    the ``project_id`` filter; an explicit scope that matches nothing
    returns an empty list (a legitimately empty export).
    """
    if not site_ids and not deployment_ids:
        return None
    clauses = []
    if site_ids:
        clauses.append(Deployment.site_id.in_(site_ids))
    if deployment_ids:
        clauses.append(Deployment.id.in_(deployment_ids))
    rows = (
        db.query(Deployment.id)
        .filter(Deployment.project_id == project.id)
        .filter(or_(*clauses))
        .all()
    )
    return [r[0] for r in rows]


def get_scoped_detection_rows(
    db: Session,
    project: Project,
    *,
    extra_excluded: list[str] | None = None,
    deployment_ids: list[str] | None = None,
    apply_threshold: bool = True,
) -> list[Row[Any]]:
    """
    Return every (File, Detection, Deployment, Site, LabelTaxonomy) row
    in scope for ``project``.

    LEFT JOIN on Detection: keeps files with zero in-scope detections so
    the caller can emit a blank row.

    ``extra_excluded`` augments ``project.excluded_classes`` with a
    per-call exclusion list. The folder-run Save step uses this so the
    user's "exclude these species from outputs" choice on the Save
    page applies to exports without mutating the project's persistent
    exclusion list.

    ``apply_threshold=False`` drops the detection-threshold filter so
    every stored detection is in scope. The folder-run save step uses
    this: its data exports are the complete record of the run
    (detections are only thresholded in-app and in media outputs).
    Projects-mode export endpoints keep the default threshold +
    verified-override rule (DEVELOPERS.md).
    """
    threshold_clause = (
        or_(
            Detection.confidence >= project.counting_threshold,
            Detection.verified.is_(True),
        )
        if apply_threshold
        else true()
    )

    # Outer join on Site so deployments without an assigned site still
    # appear in the export; the CSV serializer emits blank lat/lon cells
    # for those rows, and CamtrapDP / GeoJSON skip them with a separate
    # skipped_deployment_ids list.
    query = (
        select(File, Detection, Deployment, Site, LabelTaxonomy)
        # Drop File.exif_data: the full per-file EXIF JSON blob is never read
        # by any export builder (CamTrap's exifData column is emitted blank).
        # Without this, SQLite drags ~70k EXIF blobs through the ORDER BY
        # sorter's temp file and hits SQLITE_FULL ("database or disk is full")
        # on large projects even with plenty of free space on the data drive.
        .options(defer(File.exif_data))
        .select_from(File)
        .join(Deployment, File.deployment_id == Deployment.id)
        .outerjoin(Site, Deployment.site_id == Site.id)
        .outerjoin(
            Detection,
            and_(Detection.file_id == File.id, threshold_clause),
        )
        .outerjoin(
            LabelTaxonomy, LabelTaxonomy.id == Detection.label_taxonomy_id
        )
        .where(Deployment.project_id == project.id)
        .where(File.file_type.in_(("image", "video")))
        .order_by(File.captured_at_local.asc(), File.id, Detection.id)
    )

    # Optional export scope: narrow to a subset of the project's
    # deployments (see resolve_scope_deployment_ids). None = whole project.
    if deployment_ids is not None:
        query = query.where(Deployment.id.in_(deployment_ids))

    excluded = [s.lower() for s in (project.excluded_classes or [])]
    if extra_excluded:
        excluded = list({*excluded, *(s.lower() for s in extra_excluded)})
    if excluded:
        # Drop animal detections whose taxonomy name OR raw label matches
        # the excluded list (case-insensitive). ``coalesce`` turns null
        # taxonomy/label into an empty string so ``in_`` behaves predictably
        # instead of returning SQL NULL. Verified detections always
        # survive: a human relabel to an excluded species (possible when
        # the species selection hid the true class from the classifier)
        # outranks the exclusion config, same as the threshold override.
        taxonomy_name_lower = func.lower(func.coalesce(LabelTaxonomy.name, ""))
        detection_label_lower = func.lower(func.coalesce(Detection.label, ""))
        excluded_match = or_(
            taxonomy_name_lower.in_(excluded),
            detection_label_lower.in_(excluded),
        )
        query = query.where(
            or_(
                Detection.id.is_(None),
                Detection.category != "animal",
                Detection.verified.is_(True),
                ~excluded_match,
            )
        )

    return list(db.execute(query).all())


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _group_rows_by_file(
    scoped_rows: Sequence[Row[Any]],
) -> list[tuple[File, Deployment, Site, list[tuple[Detection, LabelTaxonomy | None]]]]:
    """Regroup the LEFT-JOIN output into one entry per File."""
    out: list[
        tuple[File, Deployment, Site, list[tuple[Detection, LabelTaxonomy | None]]]
    ] = []
    current_file_id: str | None = None
    current_entry: (
        tuple[File, Deployment, Site, list[tuple[Detection, LabelTaxonomy | None]]] | None
    ) = None

    for row in scoped_rows:
        file_obj, detection, deployment, site, taxonomy = row
        if current_file_id != file_obj.id:
            current_entry = (file_obj, deployment, site, [])
            out.append(current_entry)
            current_file_id = file_obj.id
        assert current_entry is not None
        if detection is not None:
            current_entry[3].append((detection, taxonomy))

    return out


def _species_label(detection: Detection, taxonomy: LabelTaxonomy | None) -> str:
    """Human-readable species name for a detection. Empty for non-animals without a label."""
    if detection.category != "animal":
        return detection.category
    if detection.scientific_name:
        return detection.scientific_name
    if detection.label:
        return detection.label
    return detection.category


def _taxon_ranks(taxonomy: LabelTaxonomy | None) -> list[str]:
    """The five formal ranks, empty string where unknown."""
    if taxonomy is None:
        return ["", "", "", "", ""]
    return [
        taxonomy.taxon_class or "",
        taxonomy.taxon_order or "",
        taxonomy.taxon_family or "",
        taxonomy.taxon_genus or "",
        taxonomy.taxon_species or "",
    ]


def _scientific_name(
    detection: Detection, taxonomy: LabelTaxonomy | None
) -> str:
    """
    Latin / scientific name. ``label_taxonomy.scientific_name`` is the single
    source of truth (see MEMORY.md project_taxonomy_scientific_name).
    """
    if detection.category != "animal":
        return ""
    if taxonomy and taxonomy.scientific_name:
        return taxonomy.scientific_name
    return ""


def _file_event(
    db: Session, file_id: str
) -> Event | None:
    """Fetch the event a file belongs to (max one; event_files has no multi-row case today)."""
    stmt = (
        select(Event)
        .join(event_files, event_files.c.event_id == Event.id)
        .where(event_files.c.file_id == file_id)
        .limit(1)
    )
    return db.execute(stmt).scalars().first()


def _events_by_file(
    db: Session, file_ids: Sequence[str]
) -> dict[str, Event]:
    """Map each file id to its event (avoids N+1).

    Chunked over ``file_ids`` to stay under SQLite's bound-parameter limit:
    one `IN (?, ?, ...)` over every file id crashes large runs with "too many
    SQL variables" (Simon's 45k-file folder run).
    """
    result: dict[str, Event] = {}
    for chunk in iter_id_chunks(file_ids):
        stmt = (
            select(event_files.c.file_id, Event)
            .join(event_files, event_files.c.event_id == Event.id)
            .where(event_files.c.file_id.in_(chunk))
        )
        for fid, event in db.execute(stmt).all():
            result[fid] = event
    return result


def _classifier_label(project: Project) -> str:
    """What to put in CamTrap DP ``classifiedBy``."""
    if project.classification_model_id:
        return project.classification_model_id
    return project.detection_model_id


def _relative_path(file_obj: File, deployment: Deployment | None) -> str:
    """File path relative to its deployment's source folder, forward slashes.

    Disambiguates duplicate filenames across cameras and stays portable
    (no machine-specific absolute prefix). Falls back to the bare
    filename when the deployment folder is unknown or the file sits
    outside it.
    """
    folder = deployment.folder_path if deployment else None
    if folder:
        try:
            return Path(file_obj.file_path).relative_to(folder).as_posix()
        except ValueError:
            pass
    return Path(file_obj.file_path).name


def _media_type_for(file_format: str | None) -> str:
    if not file_format:
        return "application/octet-stream"
    fmt = file_format.lower()
    if fmt in ("jpg", "jpeg"):
        return "image/jpeg"
    if fmt == "png":
        return "image/png"
    if fmt == "mp4":
        return "video/mp4"
    if fmt == "mov":
        return "video/quicktime"
    if fmt == "avi":
        return "video/x-msvideo"
    return "application/octet-stream"


def _iso_date_at_midnight(d: date | None, tz_name: str) -> str:
    if d is None:
        return ""
    return to_local_iso_with_offset(datetime(d.year, d.month, d.day), tz_name)


def _iso_datetime(dt: datetime | None, tz_name: str) -> str:
    if dt is None:
        return ""
    return to_local_iso_with_offset(dt, tz_name)


# ---------------------------------------------------------------------------
# Flat detections rows (one row per detection / bounding box)
# ---------------------------------------------------------------------------


def build_detection_rows(
    db: Session, project: Project, scoped_rows: Sequence[Row[Any]]
) -> tuple[list[str], list[list[Any]]]:
    """
    Build `(headers, rows)` for the flat Detections export.

    Grain: one row per detection (the box view). Lean on attributes (time /
    place / paths live in files.csv) but carries the full FK id set
    (file_id, deployment_id, event_id) for direct linkability. Files with no
    detections do not appear here; per-event counts live in counts.csv.

    `event_id` is blank when the file has no event, so every non-empty
    value resolves in the events table. See the note on `_FILES_HEADERS`.
    """
    grouped = list(_group_rows_by_file(scoped_rows))
    event_map = _events_by_file(db, [f.id for f, _d, _s, _dets in grouped])

    rows: list[list[Any]] = []
    for file_obj, deployment, _site, detections in grouped:
        deployment_id = deployment.id if deployment is not None else ""
        event = event_map.get(file_obj.id)
        event_id = event.id if event else ""
        for detection, taxonomy in detections:
            rows.append(
                [detection.id, file_obj.id, deployment_id, event_id]
                + _detection_cells(detection, taxonomy)
            )

    return _FLAT_DETECTION_HEADERS, rows


def _round_or_blank(value: float | None, ndigits: int) -> float | str:
    return round(value, ndigits) if value is not None else ""


def _detection_cells(
    detection: Detection,
    taxonomy: LabelTaxonomy | None,
) -> list[Any]:
    """The detector + classifier tail of a detections row (everything after
    the id columns).

    ``classification_label`` is the species label only — empty for person,
    vehicle, or an unclassified animal.
    """
    return [
        detection.category,
        round(detection.confidence, 6),
        detection.label or "",
        _round_or_blank(detection.label_confidence, 6),
        detection.original_label or "",
        _round_or_blank(detection.original_label_confidence, 6),
        detection.classification_method or "",
        "TRUE" if detection.verified else "FALSE",
        *_taxon_ranks(taxonomy),
        detection.scientific_name or "",
        detection.common_name or "",
        detection.frame_number if detection.frame_number is not None else "",
        _round_or_blank(detection.bbox_x, 6),
        _round_or_blank(detection.bbox_y, 6),
        _round_or_blank(detection.bbox_width, 6),
        _round_or_blank(detection.bbox_height, 6),
    ]


# ---------------------------------------------------------------------------
# Deployments rows (one row per deployment, the location / effort table)
# ---------------------------------------------------------------------------

# One row per deployment: where it was (site + coordinates) and the effort
# (date span + trap-nights). The single home for location, so files.csv and
# counts.csv carry only deployment_id and join here. Mirrors the Camtrap-DP
# deployments table.
def _format_tags(tags: dict | None) -> str:
    """Serialize a tags dict to one cell as pipe-separated key:value pairs,
    matching the Camtrap-DP deploymentTags convention (e.g. "season:wet | access:4x4")."""
    if not tags:
        return ""
    return " | ".join(f"{k}:{v}" for k, v in tags.items())


_DEPLOYMENTS_HEADERS = [
    "deployment_id",
    "site_name",
    "latitude",
    "longitude",
    "site_elevation_m",
    "site_habitat",
    "site_notes",
    "site_tags",
    "deployment_start",
    "deployment_end",
    "trap_nights",
    "deployment_notes",
    "deployment_tags",
]


def build_deployments_rows(
    db: Session,
    project: Project,
    deployment_ids: list[str] | None = None,
) -> tuple[list[str], list[list[Any]]]:
    """Build `(headers, rows)` for the Deployments export: one row per
    deployment with its site, coordinates, date span and trap-nights. The
    single home for location / effort; files and counts join here on
    deployment_id."""
    from app.api.crud.trap_nights import compute_trap_nights_for_deployments

    query = (
        db.query(Deployment, Site)
        .outerjoin(Site, Site.id == Deployment.site_id)
        .filter(Deployment.project_id == project.id)
        .order_by(Deployment.start_date_local.asc(), Deployment.id)
    )
    if deployment_ids is not None:
        query = query.filter(Deployment.id.in_(deployment_ids))
    deployments = query.all()
    trap_nights = compute_trap_nights_for_deployments(
        db, [dep.id for dep, _site in deployments]
    )

    rows: list[list[Any]] = []
    for dep, site in deployments:
        rows.append(
            [
                dep.id,
                site.name if site is not None else "",
                site.latitude if site is not None else "",
                site.longitude if site is not None else "",
                site.elevation_m if site is not None and site.elevation_m is not None else "",
                site.habitat_type or "" if site is not None else "",
                site.notes or "" if site is not None else "",
                _format_tags(site.tags) if site is not None else "",
                dep.start_date_local.isoformat() if dep.start_date_local else "",
                dep.end_date_local.isoformat() if dep.end_date_local else "",
                trap_nights.get(dep.id, ""),
                dep.notes or "",
                _format_tags(dep.tags),
            ]
        )

    return _DEPLOYMENTS_HEADERS, rows


# ---------------------------------------------------------------------------
# Files rows (one row per media file, the media / membership table)
# ---------------------------------------------------------------------------

# One row per file, including empties. This is the tidy home for "which
# files had no detections" (category=blank) and "which files are in which
# event" (event_id), instead of faking blank rows in the detections table.
# Mirrors the Camtrap-DP media table. Location lives in deployments.csv;
# join on deployment_id.
#
# event_id is blank when a file has no event, never a stand-in id. Every
# image / video lands in exactly one cluster once events are generated
# (date-less files become singleton events with NULL bounds), so in
# practice the column is always populated. If it ever is blank, that means
# what it says: no event. Every non-empty value resolves in the events
# table.
_FILES_HEADERS = [
    "file_id",
    "deployment_id",
    "event_id",
    "file_type",
    "relative_path",
    "absolute_path",
    "datetime",
    # What the camera wrote into the image's EXIF at capture time, as
    # extracted during analysis (megadetector.py's --include_exif_tags)
    # and stored in File.exif_data. Blank for videos (no EXIF), for
    # cameras that do not record the tag (most brands keep temperature
    # in maker notes the PIL-based reader cannot see), and for analyses
    # run before these tags were extracted; a re-analysis fills them,
    # reprocessing cannot (it reuses the stored results.json).
    "camera_make",
    "camera_model",
    "ambient_temperature",
    "camera_serial",
    # File-level rollup of what the file holds: the raw detector category
    # of the file's strongest passing detection, or "blank"
    # (File.observation_type). Distinct from the per-box
    # detection_category in detections.csv, and it uniquely carries "blank".
    "observation_type",
    # Everything from here to common_name describes that same strongest box.
    # Each score sits directly after what it scores, the way
    # _detection_cells lays out these two columns, because a bare
    # confidence is meaningless without its subject beside it.
    #
    # **detection_confidence is the deciding box's score, not the file's
    # highest.** strongest_passing_detection sorts on (verified, confidence),
    # so a box a human verified at 0.30 beats an unverified one at 0.99 and
    # this column then reads 0.30 — below the project's counting threshold.
    # Filtering the CSV on `detection_confidence >= x` therefore drops
    # exactly the files someone checked by hand, silently. is_verified does
    # not rescue it: that column is File.verified, true only when *every*
    # reviewable box is verified, so a part-reviewed file reads FALSE. Said
    # plainly in docs/docs/reference/exports.md; if it ever bites a user the
    # one-column answer is to append the deciding box's own verified flag.
    "detection_confidence",
    # The species of that same strongest box, its formal ranks, and its two
    # display names. Deliberately NOT the highest-confidence label anywhere
    # on the file: taking the best label instead of the best box is what
    # filed a person in camouflage under "chimpanzee" (see DEVELOPERS.md
    # "What a file is about"). Blank when the winning box carries no
    # species, or when nothing passed at all.
    #
    # The ranks are not decoration. Taxonomic rollup means
    # classification_label holds whatever rank the pipeline could reach, so
    # one column mixes species ("porcupine") with orders ("rodentia") and
    # families ("bovidae"). The ranks are the only thing that says which is
    # which, so grouping by the label alone silently merges an order with
    # the species inside it. Same names as detections.csv and counts.csv,
    # and the same pairing order as detections.csv; counts.csv has no
    # confidence columns to pair, being one row per species per event.
    #
    # These are computed from the detections in *this export's scope*,
    # while observation_type is read from the stored column, which is
    # derived over every detection. An excluded_classes list that changed
    # after analysis can therefore put the category and the species on
    # different boxes; scoping buys the stronger promise that a non-empty
    # species always resolves to a row in detections.csv under the same
    # file_id. Pinned by test_files_export_species_follow_the_export_scope.
    "classification_label",
    # 1.0 whenever a human set the label, as in detections.csv. Blank for a
    # person, a vehicle, or an animal that was never classified.
    "classification_confidence",
    "taxon_class",
    "taxon_order",
    "taxon_family",
    "taxon_genus",
    "taxon_species",
    "scientific_name",
    "common_name",
    "is_verified",
    "notes",
]


# The EXIF tags behind the four camera columns, in column order. Keys are
# the PIL tag names as they appear in File.exif_data.
_CAMERA_EXIF_KEYS = ("Make", "Model", "AmbientTemperature", "BodySerialNumber")

_BLANK_CAMERA_CELLS = ["", "", "", ""]


def _camera_cells_by_file(
    db: Session, file_ids: Sequence[str]
) -> dict[str, list[str]]:
    """The four camera cells (make, model, temperature, serial) per file id.

    Fetched in a separate chunked query rather than through
    ``get_scoped_detection_rows``: that query defers ``File.exif_data`` so
    the blobs stay out of its ORDER BY sorter (the SQLITE_FULL fix noted
    there), and this one has no ORDER BY, so reading them here is safe.
    Only the four small strings are kept; files with no stored EXIF are
    simply absent (callers fall back to blanks).
    """
    cells: dict[str, list[str]] = {}
    for chunk in iter_id_chunks(file_ids):
        stmt = select(File.id, File.exif_data).where(File.id.in_(chunk))
        for file_id, blob in db.execute(stmt):
            if not blob:
                continue
            # EXIF ASCII fields are fixed-length and NUL-padded, and the
            # detector's reader keeps the padding ('HC500 HYPERFIRE\x00\x00',
            # or a Make that is nothing but NULs). Strip NULs and whitespace:
            # bare NULs would land in the CSV verbatim and crash openpyxl
            # (IllegalCharacterError) in the XLSX export.
            cells[file_id] = [
                str(blob.get(key) or "").replace("\x00", "").strip()
                for key in _CAMERA_EXIF_KEYS
            ]
    return cells


def build_files_rows(
    db: Session,
    project: Project,
    deployment_ids: list[str] | None = None,
    scoped_rows: Sequence[Row[Any]] | None = None,
) -> tuple[list[str], list[list[Any]]]:
    """Build `(headers, rows)` for the Files export: one row per media file,
    including files with no detections (the effort table). `event_id`
    answers "which files are in this event".

    ``scoped_rows`` lets a caller that already fetched the default
    (thresholded) scoped rows for the same ``deployment_ids`` reuse them
    instead of paying the query twice. Only rows from that identical
    query are valid here: rows fetched with ``apply_threshold=False``
    are NOT equivalent — the excluded-classes clause deletes whole rows,
    so a file whose only detections are excluded animals below the
    threshold exists in the thresholded set (as a blank row) but is
    missing from the unthresholded one. The folder-run table writers
    therefore keep the internal fetch."""
    tz_name = project.timezone

    if scoped_rows is None:
        scoped_rows = get_scoped_detection_rows(
            db, project, deployment_ids=deployment_ids
        )
    grouped = list(_group_rows_by_file(scoped_rows))
    file_ids = [f.id for f, _d, _s, _dets in grouped]
    event_map = _events_by_file(db, file_ids)
    camera_map = _camera_cells_by_file(db, file_ids)

    rows: list[list[Any]] = []
    for file_obj, deployment, _site, detections in grouped:
        event = event_map.get(file_obj.id)
        event_id = event.id if event else ""
        rows.append(
            [
                file_obj.id,
                deployment.id if deployment is not None else "",
                event_id,
                file_obj.file_type or "",
                _relative_path(file_obj, deployment),
                file_obj.file_path,
                _iso_datetime(file_obj.captured_at_local, tz_name),
                *camera_map.get(file_obj.id, _BLANK_CAMERA_CELLS),
                file_obj.observation_type or "",
                *_strongest_species_cells(project, file_obj, detections),
                "TRUE" if file_obj.verified else "FALSE",
                file_obj.notes or "",
            ]
        )

    return _FILES_HEADERS, rows


def _strongest_species_cells(
    project: Project,
    file_obj: File,
    detections: Sequence[tuple[Detection, LabelTaxonomy | None]],
) -> list[Any]:
    """Everything the Files row says about the file's strongest passing
    detection, or the same width in blanks when nothing passes.

    Order: detector score, label, label score, the five formal ranks, then
    the two display names. The score/subject pairing is ``_detection_cells``'
    layout; the label + ranks + names run is ``build_observation_rows``'.

    Both confidences are read off the same ``best`` object as the label, so
    they cannot describe a different box than the species does. Rounded to
    six places like ``_detection_cells``, so a value here compares equal to
    the same field of the same box in detections.csv.

    Gated to the file's visible surface, so for a video this is its best
    frame. That is the same rule ``observation_type`` beside it now uses,
    which is what keeps the whole block describing one box, and a box the
    user can actually open. The gate runs in Python rather than in
    ``get_scoped_detection_rows`` because that query is shared with the
    Detections export, which is per box and keeps every frame by design.

    The threshold is passed explicitly even though ``get_scoped_detection_rows``
    already applied the same predicate in SQL. Re-applying it costs one pass
    over a handful of already-loaded objects and keeps these columns tied to
    the same threshold ``observation_type`` uses, so a later change to that
    query's ``apply_threshold`` cannot silently desynchronise the two.

    Names come off the detection, not off the joined ``LabelTaxonomy``. That is
    the convention ``_detection_cells`` uses, so a Files row and its
    detections.csv rows read identically with no join. (The other convention in
    this module, ``_scientific_name``, blanks non-animals and reads the taxonomy
    row; it serves the Camtrap and spatial builders.) The practical effect is
    that a box with no species still names itself: ``Person``, ``Vehicle``,
    ``Animal``, per ``resolve_label_names``.
    """
    visible = visible_detections(file_obj, [det for det, _tax in detections])
    best = strongest_passing_detection(visible, project.counting_threshold)
    if best is None:
        # Nothing passed, so there is no box to describe. Blank, never 0.0:
        # a zero would read as "the detector scored nothing" and would
        # survive a `< x` filter as if it were a real measurement. The
        # ranks come from the same helper as below; the four literals are
        # this branch's own, so keep the two widths in step by hand (the
        # row-width assertion in test_export_files_includes_empties is what
        # catches it if they drift).
        return ["", "", "", *_taxon_ranks(None), "", ""]

    # The taxonomy joined to *that* box. Read off the row tuple rather than
    # ``best.label_taxonomy``, which would lazy-load once per file.
    taxonomy = next((tax for det, tax in detections if det is best), None)
    return [
        # Non-nullable, so no blank helper; label_confidence is nullable and
        # gets one. Same two idioms as _detection_cells.
        round(best.confidence, 6),
        best.label or "",
        _round_or_blank(best.label_confidence, 6),
        *_taxon_ranks(taxonomy),
        best.scientific_name or "",
        best.common_name or "",
    ]


# ---------------------------------------------------------------------------
# Event-level observations rows (one row per species per event)
# ---------------------------------------------------------------------------

# The ecological record table: one row per event x species with the
# effective count (human-confirmed if set, else the AI count) and the
# event sign-off. This is the analysis-ready "record table" and the Counts
# page output, distinct from the per-detection Detections export above.
_OBSERVATIONS_HEADERS = [
    "event_id",
    "deployment_id",
    "event_start",
    "event_end",
    "category",
    "classification_label",
    "taxon_class",
    "taxon_order",
    "taxon_family",
    "taxon_genus",
    "taxon_species",
    "scientific_name",
    "common_name",
    # The human-confirmed count, falling back to the AI's count when the
    # event isn't confirmed.
    "count",
    "is_confirmed",
]


def build_observation_rows(
    db: Session,
    project: Project,
    deployment_ids: list[str] | None = None,
) -> tuple[list[str], list[list[Any]]]:
    """
    Build `(headers, rows)` for the event-level Observations export.

    Grain: one row per event x species, carrying the effective count
    (human-confirmed if set, else the AI count) and the event sign-off.
    Count-0 rows (a species a human removed) are skipped. This maps to the
    Counts page; the per-detection grain lives in the Detections export.
    """
    tz_name = project.timezone

    query = (
        db.query(EventObservation, Event, Deployment, LabelTaxonomy)
        .join(Event, Event.id == EventObservation.event_id)
        .join(Deployment, Deployment.id == Event.deployment_id)
        .outerjoin(
            LabelTaxonomy,
            LabelTaxonomy.id == EventObservation.label_taxonomy_id,
        )
        .filter(Deployment.project_id == project.id)
        .order_by(Event.event_start_local.asc(), Event.id)
    )
    if deployment_ids is not None:
        query = query.filter(Deployment.id.in_(deployment_ids))

    rows: list[list[Any]] = []
    for obs, event, deployment, taxonomy in query.all():
        count = obs.effective_count
        if count <= 0:
            continue
        rows.append(
            [
                event.id,
                deployment.id,
                _iso_datetime(event.event_start_local, tz_name),
                _iso_datetime(event.event_end_local, tz_name),
                obs.category,
                obs.label or "",
                *_taxon_ranks(taxonomy),
                (taxonomy.scientific_name if taxonomy else "") or "",
                (taxonomy.common_name if taxonomy else "") or "",
                count,
                "TRUE" if event.confirmed else "FALSE",
            ]
        )

    return _OBSERVATIONS_HEADERS, rows


def build_spreadsheet_sheets(
    db: Session,
    project: Project,
    deployment_ids: list[str] | None = None,
) -> list[tuple[str, list[str], list[list[Any]]]]:
    """The tables that make up the project Export page's combined
    spreadsheet: Deployments, Files, Detections, and Counts. The
    folder-run Save step writes its own two-sheet workbook (Files +
    Detections) from the same row builders.

    ``deployment_ids`` narrows every sheet to a subset of the project's
    deployments; None exports everything."""
    scoped = get_scoped_detection_rows(db, project, deployment_ids=deployment_ids)
    dep_headers, dep_rows = build_deployments_rows(db, project, deployment_ids)
    # Same query, same threshold, same scope as the fetch above — reuse
    # the rows instead of running it a second time.
    files_headers, files_rows = build_files_rows(
        db, project, deployment_ids, scoped_rows=scoped
    )
    det_headers, det_rows = build_detection_rows(db, project, scoped)
    obs_headers, obs_rows = build_observation_rows(db, project, deployment_ids)
    return [
        ("Deployments", dep_headers, dep_rows),
        ("Files", files_headers, files_rows),
        ("Detections", det_headers, det_rows),
        ("Counts", obs_headers, obs_rows),
    ]


# ---------------------------------------------------------------------------
# Spatial layers
# ---------------------------------------------------------------------------


def build_spatial_layers(
    db: Session, project: Project, scoped_rows: Sequence[Row[Any]]
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """Build the two spatial layers: deployments and species_summary.

    Both are genuinely spatial: one point per camera, and one point per
    camera x species. A per-detection "observations" layer was dropped on
    purpose -- every detection sits on its camera's exact coordinate, so
    it carried no spatial resolution beyond the camera, and the flat CSV
    already exposes latitude/longitude per row for direct GIS import.

    Returns ``(layers, skipped_deployment_ids)``. The skipped list
    contains deployments that were excluded from the GeoJSON /
    Shapefile / GeoPackage because they have no site coordinates.
    """
    # Build a deployment-level detection count map from the scoped rows so
    # we don't re-query. Also build trap-days per deployment.
    det_count_by_dep: dict[str, int] = defaultdict(int)
    site_by_deployment: dict[str, Site | None] = {}
    deployments_seen: dict[str, Deployment] = {}
    for row in scoped_rows:
        file_obj, detection, deployment, site, _taxonomy = row
        deployments_seen[deployment.id] = deployment
        site_by_deployment[deployment.id] = site
        if detection is not None:
            det_count_by_dep[deployment.id] += 1

    # Include project deployments that had no in-scope files (so the
    # deployments layer lists every deployment, not only those with
    # detections). Outer join on Site so null-site deployments are not
    # filtered out here; we skip them below when building features
    # (GeoJSON needs coordinates) and record them separately.
    all_deployments = (
        db.query(Deployment, Site)
        .outerjoin(Site, Deployment.site_id == Site.id)
        .filter(Deployment.project_id == project.id)
        .all()
    )

    from app.api.crud.trap_nights import compute_trap_nights_for_deployments

    trap_days_by_dep = compute_trap_nights_for_deployments(
        db, [d.id for d, _ in all_deployments]
    )

    deployments_features: list[dict[str, Any]] = []
    skipped_deployment_ids: list[str] = []
    for deployment, site in all_deployments:
        trap_days = trap_days_by_dep.get(deployment.id, 0)
        det_count = det_count_by_dep.get(deployment.id, 0)
        rate = (det_count / trap_days * 100) if trap_days > 0 else 0.0
        if site is None:
            skipped_deployment_ids.append(deployment.id)
            continue
        deployments_features.append(
            {
                "lon": site.longitude,
                "lat": site.latitude,
                "properties": {
                    "site_name": site.name,
                    "deployment_id": deployment.id,
                    "start_date": (
                        deployment.start_date_local.isoformat()
                        if deployment.start_date_local
                        else ""
                    ),
                    "end_date": (
                        deployment.end_date_local.isoformat()
                        if deployment.end_date_local
                        else ""
                    ),
                    "trap_days": trap_days,
                    "detection_count": det_count,
                    "detection_rate_per_100": round(rate, 2),
                },
            }
        )

    # Species summary: one point per (site, species). Aggregates classified
    # detections only -- person, vehicle, unclassified, and blank rows have
    # no label and don't belong on a species map. Carries the common label,
    # the latin name, and the taxon ranks for readable map labelling and
    # taxonomic filtering in GIS.
    trap_days_by_site: dict[str, int] = defaultdict(int)
    site_coords: dict[str, tuple[float, float]] = {}
    for deployment, site in all_deployments:
        if site is None:
            continue
        trap_days_by_site[site.id] += trap_days_by_dep.get(deployment.id, 0)
        site_coords[site.id] = (site.longitude, site.latitude)

    summary_bucket: dict[tuple[str, str], dict[str, Any]] = {}
    for row in scoped_rows:
        _file_obj, detection, _deployment, site, taxonomy = row
        if detection is None or site is None:
            continue
        label = detection.label
        if not label:
            continue
        key = (site.id, label)
        bucket = summary_bucket.get(key)
        if bucket is None:
            bucket = {
                "site_name": site.name,
                "classification_label": label,
                "scientific_name": _scientific_name(detection, taxonomy),
                "taxon_ranks": _taxon_ranks(taxonomy),
                "total_count": 0,
            }
            summary_bucket[key] = bucket
        bucket["total_count"] += 1

    species_summary_features: list[dict[str, Any]] = []
    for (site_id, _label), data in summary_bucket.items():
        if site_id not in site_coords:
            continue
        trap_days = trap_days_by_site.get(site_id, 0)
        rate = (data["total_count"] / trap_days * 100) if trap_days > 0 else 0.0
        lon, lat = site_coords[site_id]
        ranks = data["taxon_ranks"]
        species_summary_features.append(
            {
                "lon": lon,
                "lat": lat,
                "properties": {
                    "site_name": data["site_name"],
                    "classification_label": data["classification_label"],
                    "scientific_name": data["scientific_name"],
                    "taxon_class": ranks[0],
                    "taxon_order": ranks[1],
                    "taxon_family": ranks[2],
                    "taxon_genus": ranks[3],
                    "taxon_species": ranks[4],
                    "total_count": data["total_count"],
                    "detection_rate_per_100": round(rate, 2),
                },
            }
        )

    return (
        {
            "deployments": deployments_features,
            "species_summary": species_summary_features,
        },
        skipped_deployment_ids,
    )


# ---------------------------------------------------------------------------
# CamTrap DP tables
# ---------------------------------------------------------------------------


def build_camtrap_dp_tables(
    db: Session, project: Project, scoped_rows: Sequence[Row[Any]]
) -> tuple[
    list[list[Any]],
    list[list[Any]],
    list[list[Any]],
    dict[str, Any],
    list[str],
]:
    """
    Build CamTrap DP deployments/media/observations rows and datapackage dict.

    Returns (deployments_rows, media_rows, observations_rows, datapackage_dict,
    skipped_deployment_ids). Deployments without a site are skipped
    because CamtrapDP requires lat/lon; the caller can surface the list
    to the user.
    """
    tz_name = project.timezone
    classified_by = _classifier_label(project)
    not_reviewed = f"{classified_by}, not reviewed"

    deployments_rows: list[list[Any]] = []
    media_rows: list[list[Any]] = []
    observations_rows: list[list[Any]] = []
    observed_taxa: dict[str, tuple[LabelTaxonomy | None, str]] = {}
    # Events in scope, for the event-level (per-species count) observation
    # rows emitted after the per-file media rows.
    events_in_scope: dict[str, dict[str, str]] = {}
    first_captured: datetime | None = None
    last_captured: datetime | None = None
    sites_seen: dict[str, Site] = {}

    # deployments.csv: every deployment in the project with a site.
    # Null-site deployments are skipped because CamtrapDP requires
    # deploymentLocation (lat/lon). The caller receives their ids via
    # the returned `camtrap_skipped_deployment_ids` list.
    camtrap_skipped_deployment_ids: list[str] = []
    for deployment, site in (
        db.query(Deployment, Site)
        .outerjoin(Site, Deployment.site_id == Site.id)
        .filter(Deployment.project_id == project.id)
        .order_by(Deployment.start_date_local)
        .all()
    ):
        if site is None:
            camtrap_skipped_deployment_ids.append(deployment.id)
            continue
        sites_seen[site.id] = site
        camera_model = deployment.camera_model or ""
        camera_id = deployment.camera_serial or deployment.camera_model or deployment.id
        # Row order must match _CAMTRAP_DEPLOYMENTS_HEADERS exactly.
        deployments_rows.append(
            [
                deployment.id,                                                # deploymentID
                "",                                                           # locationID
                site.name or "",                                              # locationName
                site.latitude,                                                # latitude
                site.longitude,                                               # longitude
                "",                                                          # coordinateUncertainty
                _iso_date_at_midnight(deployment.start_date_local, tz_name),  # deploymentStart
                _iso_date_at_midnight(
                    deployment.end_date_local or date.today(), tz_name
                ),                                                            # deploymentEnd
                "",                                                           # setupBy
                camera_id,                                                    # cameraID
                camera_model,                                                 # cameraModel
                "",                                                           # cameraDelay
                "",                                                           # cameraHeight
                "",                                                           # cameraDepth
                "",                                                           # cameraTilt
                "",                                                           # cameraHeading
                "",                                                           # detectionDistance
                "",                                                           # timestampIssues
                "",                                                           # baitUse
                "",                                                           # featureType
                site.habitat_type or "",                                      # habitat
                "",                                                           # deploymentGroups
                _format_tags(deployment.tags),                                # deploymentTags
                deployment.notes or "",                                       # deploymentComments
            ]
        )

    # media.csv + observations.csv: iterate scoped rows grouped by file.
    # Files whose deployment has no site were already skipped at the
    # deployments.csv stage (CamtrapDP requires lat/lon). Skip their
    # media and observations rows here so the package stays consistent.
    skipped_dep_set = set(camtrap_skipped_deployment_ids)
    files_without_date = 0
    for file_obj, deployment, _site, detections in _group_rows_by_file(scoped_rows):
        if deployment.id in skipped_dep_set:
            continue
        # media.timestamp and observations.eventStart / eventEnd are
        # required datetime fields in the CamtrapDP schema, so a file
        # with no capture date cannot be represented: an empty string
        # there fails validation in the camtrapdp R package and GBIF
        # ingestion, poisoning the whole package. The export dialog
        # warns about these via /files-without-date.
        if file_obj.captured_at_local is None:
            files_without_date += 1
            continue
        if first_captured is None or file_obj.captured_at_local < first_captured:
            first_captured = file_obj.captured_at_local
        if last_captured is None or file_obj.captured_at_local > last_captured:
            last_captured = file_obj.captured_at_local

        # Row order must match _CAMTRAP_MEDIA_HEADERS exactly.
        media_rows.append(
            [
                file_obj.id,                                        # mediaID
                deployment.id,                                      # deploymentID
                "activityDetection",                                # captureMethod
                _iso_datetime(file_obj.captured_at_local, tz_name), # timestamp
                file_obj.file_path,                                 # filePath
                "false",                                            # filePublic
                "",                                                 # fileName
                _media_type_for(file_obj.file_format),              # fileMediatype
                "",                                                 # exifData
                "",                                                 # favorite
                "",                                                 # mediaComments
            ]
        )

        # Unlike files.csv / detections.csv, a missing event here falls back
        # to the file id on purpose. CamTrap-DP has no events resource:
        # eventID is a grouping key inside observations.csv, paired with the
        # eventStart / eventEnd written just below (the file's own capture
        # time). So an unclustered file is its own single-media event and the
        # row stays self-consistent. A blank eventID would instead merge every
        # unclustered file into one event under a GROUP BY.
        event = _file_event(db, file_obj.id)
        event_id = event.id if event else file_obj.id
        event_start = _iso_datetime(
            event.event_start_local if event else file_obj.captured_at_local, tz_name
        )
        event_end = _iso_datetime(
            event.event_end_local if event else file_obj.captured_at_local, tz_name
        )
        captured_iso = _iso_datetime(file_obj.captured_at_local, tz_name)
        if event is not None:
            events_in_scope.setdefault(
                event_id,
                {
                    "deployment_id": deployment.id,
                    "event_start": event_start or captured_iso,
                    "event_end": event_end or captured_iso,
                },
            )

        # The boxes about to be written are the honest test for "is this
        # file blank". This used to also short-circuit on the stored
        # `observation_type == "blank"`, which was near-equivalent while
        # that column was derived over every frame. It is not equivalent
        # now: a video whose best frame is empty but which still has
        # passing boxes on other frames would take the blank branch and
        # lose every per-box row from an archival export.
        if not detections:
            observations_rows.append(
                _camtrap_blank_row(
                    file_obj,
                    deployment.id,
                    event_id,
                    event_start or captured_iso,
                    event_end or captured_iso,
                    not_reviewed,
                )
            )
            continue

        for detection, taxonomy in detections:
            # Media-level rows are the per-box detections (one row per
            # bounding box). Box-less species are carried by the
            # event-level rows emitted after this loop, so skip any here.
            if detection.bbox_x is None:
                continue
            obs_type = _obs_type_from_category(detection.category)
            sci_name = _scientific_name(detection, taxonomy)
            species_name = _species_label(detection, taxonomy)
            if detection.category == "animal" and species_name:
                observed_taxa.setdefault(species_name, (taxonomy, species_name))

            obs_id_prefix = "obs-human" if detection.verified else "obs-ai"
            method = "human" if detection.verified else "machine"
            comments = "Human identification" if detection.verified else not_reviewed
            prob = (
                round(detection.label_confidence, 6)
                if detection.category == "animal" and detection.label_confidence is not None
                else ""
            )

            # Row order must match _CAMTRAP_OBS_HEADERS exactly. One
            # media-level row per bounding box (observationLevel="media").
            observations_rows.append(
                [
                    f"{obs_id_prefix}-{detection.id}",    # observationID
                    deployment.id,                         # deploymentID
                    file_obj.id,                           # mediaID
                    event_id,                              # eventID
                    event_start or captured_iso,           # eventStart
                    event_end or captured_iso,             # eventEnd
                    "media",                               # observationLevel
                    obs_type,                              # observationType
                    "",                                    # cameraSetupType
                    sci_name,                              # scientificName
                    1,                                     # count
                    "",                                    # lifeStage (enum)
                    "",                                    # sex (enum)
                    "",                                    # behavior
                    "",                                    # individualID
                    "",                                    # individualPositionRadius
                    "",                                    # individualPositionAngle
                    "",                                    # individualSpeed
                    round(detection.bbox_x, 6),            # bboxX
                    round(detection.bbox_y, 6),            # bboxY
                    round(detection.bbox_width, 6),        # bboxWidth
                    round(detection.bbox_height, 6),       # bboxHeight
                    method,                                # classificationMethod
                    classified_by,                         # classifiedBy
                    "",                                    # classificationTimestamp
                    prob,                                  # classificationProbability
                    "",                                    # observationTags
                    comments,                              # observationComments
                ]
            )

    # Event-level observations: one row per species per event carrying the
    # effective count (human override, else MaxN). Mutually exclusive and
    # summable, the Camtrap-DP shape for "N individuals in this sequence".
    # The per-box media rows above remain for spatial detail.
    # Chunked over the event ids to stay under SQLite's bound-parameter limit
    # (a large project can have >32k events in scope).
    for chunk in iter_id_chunks(events_in_scope.keys()):
        for obs, taxonomy in (
            db.query(EventObservation, LabelTaxonomy)
            .outerjoin(
                LabelTaxonomy,
                LabelTaxonomy.id == EventObservation.label_taxonomy_id,
            )
            .filter(EventObservation.event_id.in_(chunk))
            .all()
        ):
            count = obs.effective_count
            if count <= 0:
                continue
            ctx = events_in_scope[obs.event_id]
            sci_name = (
                taxonomy.scientific_name if taxonomy else None
            ) or (obs.label or "")
            species_name = (taxonomy.name if taxonomy else None) or obs.label
            if obs.category == "animal" and species_name:
                observed_taxa.setdefault(species_name, (taxonomy, species_name))
            observations_rows.append(
                _camtrap_event_row(obs, ctx, count, sci_name, classified_by)
            )

    if files_without_date:
        logger.info(
            f"CamtrapDP export: left out {files_without_date} file(s) with no "
            f"capture date (schema requires a timestamp per record)"
        )

    datapackage = _build_datapackage(
        project,
        sites_seen.values(),
        first_captured,
        last_captured,
        observed_taxa,
    )

    return (
        deployments_rows,
        media_rows,
        observations_rows,
        datapackage,
        camtrap_skipped_deployment_ids,
    )


def _camtrap_event_row(
    obs: EventObservation,
    ctx: dict[str, str],
    count: int,
    sci_name: str,
    classified_by: str,
) -> list[Any]:
    """Event-level observation row (per species per event, no bbox).

    Order must match _CAMTRAP_OBS_HEADERS exactly. classificationMethod is
    "human" when the count was set by a person, else "machine" (MaxN).
    """
    method = "human" if obs.human_count is not None else "machine"
    return [
        f"obs-event-{obs.id}",            # observationID
        ctx["deployment_id"],             # deploymentID
        "",                               # mediaID (event-level, no media)
        obs.event_id,                     # eventID
        ctx["event_start"],               # eventStart
        ctx["event_end"],                 # eventEnd
        "event",                          # observationLevel
        _obs_type_from_category(obs.category),  # observationType
        "",                               # cameraSetupType
        sci_name,                         # scientificName
        count,                            # count
        "",                               # lifeStage
        "",                               # sex
        "",                               # behavior
        "",                               # individualID
        "",                               # individualPositionRadius
        "",                               # individualPositionAngle
        "",                               # individualSpeed
        "",                               # bboxX
        "",                               # bboxY
        "",                               # bboxWidth
        "",                               # bboxHeight
        method,                           # classificationMethod
        classified_by,                    # classifiedBy
        "",                               # classificationTimestamp
        "",                               # classificationProbability
        "",                               # observationTags
        "",                               # observationComments
    ]


# The Camtrap DP `observationType` controlled vocabulary. Camtrap DP is
# an external standard (https://camtrap-dp.tdwg.org/data/) and a value
# outside this set fails validation in the camtrapdp R package and in
# GBIF ingestion.
CAMTRAP_OBSERVATION_TYPES = frozenset(
    {"animal", "human", "vehicle", "blank", "unknown", "unclassified"}
)


def _obs_type_from_category(category: str) -> str:
    """Translate our detector category into Camtrap DP's vocabulary.

    This is the **only** place a raw category is translated. Everywhere
    else in the app a category is passed through verbatim, so a marine
    detector's `shark` reaches the folder tree and the generic CSV
    intact. Camtrap DP cannot take it: the standard has no marine
    categories and expects all wildlife under `animal`, with the species
    carried separately in `scientificName`.

    So anything that is not a person, a vehicle or a blank is wildlife.
    That holds for `animal` today and for `shark` / `fish` / `turtle`
    when such a detector lands, without needing a list of them.
    """
    if category == "person":
        return "human"
    if category in ("vehicle", "blank"):
        return category
    return "animal"


def _camtrap_blank_row(
    file_obj: File,
    deployment_id: str,
    event_id: str,
    event_start: str,
    event_end: str,
    not_reviewed: str,
) -> list[Any]:
    """Observation row for a file with no in-scope detections (blank).

    Order must match _CAMTRAP_OBS_HEADERS exactly.
    """
    method = "human" if file_obj.verified else "machine"
    comments = "Human identification" if file_obj.verified else not_reviewed
    return [
        f"obs-blank-{file_obj.id}",  # observationID
        deployment_id,                # deploymentID
        file_obj.id,                  # mediaID
        event_id,                     # eventID
        event_start,                  # eventStart
        event_end,                    # eventEnd
        "media",                      # observationLevel
        "blank",                      # observationType
        "",                           # cameraSetupType
        "",                           # scientificName
        "",                           # count
        "",                           # lifeStage
        "",                           # sex
        "",                           # behavior
        "",                           # individualID
        "",                           # individualPositionRadius
        "",                           # individualPositionAngle
        "",                           # individualSpeed
        "",                           # bboxX
        "",                           # bboxY
        "",                           # bboxWidth
        "",                           # bboxHeight
        method,                       # classificationMethod
        "",                           # classifiedBy
        "",                           # classificationTimestamp
        "",                           # classificationProbability
        "",                           # observationTags
        comments,                     # observationComments
    ]


def _build_datapackage(
    project: Project,
    sites: Iterable[Site],
    first_captured: datetime | None,
    last_captured: datetime | None,
    observed_taxa: dict[str, tuple[LabelTaxonomy | None, str]],
) -> dict[str, Any]:
    sites_list = list(sites)
    lats = [s.latitude for s in sites_list]
    lons = [s.longitude for s in sites_list]

    spatial: dict[str, Any] = {}
    if lats and lons:
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        spatial = {
            "type": "Polygon",
            "bbox": [min_lon, min_lat, max_lon, max_lat],
            "coordinates": [
                [
                    [min_lon, min_lat],
                    [max_lon, min_lat],
                    [max_lon, max_lat],
                    [min_lon, max_lat],
                    [min_lon, min_lat],
                ]
            ],
        }

    temporal: dict[str, str] = {}
    if first_captured is not None:
        temporal["start"] = first_captured.date().isoformat()
    if last_captured is not None:
        temporal["end"] = last_captured.date().isoformat()

    taxonomic: list[dict[str, Any]] = []
    for species_name in sorted(observed_taxa):
        taxonomy, _ = observed_taxa[species_name]
        entry: dict[str, Any] = {}
        if taxonomy and taxonomy.scientific_name:
            entry["scientificName"] = taxonomy.scientific_name
        else:
            entry["scientificName"] = species_name
        if taxonomy and taxonomy.level:
            entry["taxonRank"] = taxonomy.level
        entry["vernacularNames"] = {"en": species_name.replace("_", " ")}
        taxonomic.append(entry)

    return {
        "profile": CAMTRAP_DP_PROFILE,
        "name": f"addaxai-{slugify(project.name)}",
        "id": str(uuid.uuid4()),
        "created": datetime.now(UTC).isoformat(),
        "title": project.name,
        "description": project.description or "",
        "version": "1.0.0",
        "contributors": [
            {"title": "AddaxAI WebUI", "role": "publisher"},
        ],
        "licenses": [
            {"name": "CC-BY-4.0", "scope": "data"},
            {"name": "CC-BY-4.0", "scope": "media"},
        ],
        "project": {
            "id": project.id,
            "title": project.name,
            "samplingDesign": "opportunistic",
            "captureMethod": ["activityDetection"],
            "individualAnimals": False,
            "observationLevel": ["media", "event"],
        },
        "spatial": spatial,
        "temporal": temporal,
        "taxonomic": taxonomic,
        "resources": [
            {
                "name": "deployments",
                "path": "deployments.csv",
                "profile": "tabular-data-resource",
                "format": "csv",
                "mediatype": "text/csv",
                "encoding": "utf-8",
                "schema": DEPLOYMENTS_SCHEMA,
            },
            {
                "name": "media",
                "path": "media.csv",
                "profile": "tabular-data-resource",
                "format": "csv",
                "mediatype": "text/csv",
                "encoding": "utf-8",
                "schema": MEDIA_SCHEMA,
            },
            {
                "name": "observations",
                "path": "observations.csv",
                "profile": "tabular-data-resource",
                "format": "csv",
                "mediatype": "text/csv",
                "encoding": "utf-8",
                "schema": OBSERVATIONS_SCHEMA,
            },
        ],
    }


def camtrap_dp_headers() -> tuple[list[str], list[str], list[str]]:
    return (
        _CAMTRAP_DEPLOYMENTS_HEADERS,
        _CAMTRAP_MEDIA_HEADERS,
        _CAMTRAP_OBS_HEADERS,
    )
