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
from collections.abc import Iterable, Iterator, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, NamedTuple

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session, defer
from sqlalchemy.sql.elements import ColumnElement

from app.api.crud.export_formats import slugify
from app.api.crud.ordering import detection_order_in_file, file_time_order
from app.core.logging_config import get_logger
from app.core.observation_attributes import LIFE_STAGES, SEXES
from app.db.sql_params import iter_id_chunks
from app.ml.detection_visibility import on_visible_frame
from app.ml.label_exclusion import is_non_label, threshold_or_verified
from app.ml.observation_type import strongest_passing_detection
from app.ml.taxonomic_rollup import format_scientific_name_from_taxonomy_row
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

# ---------------------------------------------------------------------------
# Context block: where and when a row came from
# ---------------------------------------------------------------------------

# Every table carries the identity of its parents, broadest first: site,
# deployment, event, file. The same columns in the same order in counts,
# detections and files, built by ``_site_cells`` and friends below, so a
# person can read one table in Excel without joining another. The ids
# stay, so a join is still possible for those who want one.
#
# The tables were tidy once (time and place only in files.csv and
# deployments.csv, everything else joined on an id), and four users in
# one month wrote in with the same problem: the column they needed was
# in the other file and the join key was a UUID. The rule that replaced
# it, written out in DEVELOPERS.md "Export tables carry their context":
# parent context travels down to every row; a child aggregate is added
# only when its definition is fixed and bounded. Deployments and Summary
# are the parent tables themselves and keep their own shape.
_SITE_CONTEXT_HEADERS = ["site_name", "latitude", "longitude", "site_tags"]
_DEPLOYMENT_CONTEXT_HEADERS = [
    "deployment_id",
    "deployment_start",
    "deployment_end",
    "deployment_tags",
]
_EVENT_CONTEXT_HEADERS = ["event_id", "event_start", "event_end"]
# What a row says about its file, after the file's own id: the path inside
# the deployment folder, the capture time and the pixel size. The size is
# what turns the fractional box columns into pixels without opening the
# image. The absolute path is files.csv's own: repeated per box it nearly
# doubled detections.csv (2.6 to 4.8 MB on the ENA24 test project) for a
# value the relative path plus files.csv already gives.
_FILE_CONTEXT_HEADERS = [
    "relative_path",
    "datetime",
    "image_width",
    "image_height",
]

# Flat detections CSV. One row per detection (the box grain). Carries the
# context block, so it links to every parent table and reads on its own.
# Mirrors Camtrap-DP, whose observations table carries deploymentID /
# mediaID / eventID together.
#
# Column order: the row's own id, the context block broad to specific,
# then the pipeline: the detector stage (category + score), the
# classifier stage (label + score), the label's taxonomy / names, then
# geometry.
#
#   detection_id    — the detection row id.
#   site_*, deployment_*, event_*, file_id and the file columns: the
#                     context block, same values as files.csv.
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
    *_SITE_CONTEXT_HEADERS,
    *_DEPLOYMENT_CONTEXT_HEADERS,
    *_EVENT_CONTEXT_HEADERS,
    "file_id",
    *_FILE_CONTEXT_HEADERS,
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
    "taxon_variant",
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


def _excluded_box(project: Project) -> ColumnElement[bool] | None:
    """True for a box the project's species exclusion removes from the
    exports, or None when nothing is excluded.

    An unverified animal box whose taxonomy name OR raw label matches
    ``Project.excluded_classes`` (case-insensitive). The taxonomy is
    matched through a subquery, so the clause needs no LabelTaxonomy join.
    ``coalesce`` keeps a NULL taxonomy id or label from turning the test
    NULL. Verified boxes are never removed: a human relabel to an excluded
    species (possible when the species selection hid the true class from
    the classifier) outranks the exclusion config, same as the threshold
    override.
    """
    excluded = [s.lower() for s in (project.excluded_classes or [])]
    if not excluded:
        return None
    excluded_taxonomy_ids = select(LabelTaxonomy.id).where(
        func.lower(LabelTaxonomy.name).in_(excluded)
    )
    return and_(
        Detection.category == "animal",
        Detection.verified == False,  # noqa: E712
        or_(
            func.coalesce(Detection.label_taxonomy_id, "").in_(excluded_taxonomy_ids),
            func.lower(func.coalesce(Detection.label, "")).in_(excluded),
        ),
    )


def _files_with_excluded_boxes(
    db: Session, project: Project, file_ids: Iterable[str]
) -> set[str]:
    """Which of ``file_ids`` hold a passing box the species exclusion
    removed. Such a file arrives with no detections, the same as a file
    the detector found empty, and Camtrap DP must not call it blank: a
    blank observation claims there is no animal. Chunked like
    ``_events_by_file``."""
    removed = _excluded_box(project)
    if removed is None:
        return set()
    found: set[str] = set()
    for chunk in iter_id_chunks(file_ids):
        stmt = (
            select(Detection.file_id)
            .distinct()
            .where(Detection.file_id.in_(chunk))
            .where(threshold_or_verified(project.counting_threshold))
            .where(removed)
        )
        found.update(db.scalars(stmt))
    return found


class ScopedFile(NamedTuple):
    """One media file in an export's scope, with the boxes that count."""

    file: File
    deployment: Deployment
    site: Site | None
    detections: list[tuple[Detection, LabelTaxonomy | None]]


# How many rows the scoped query fetches at a time, and about how many rows
# (boxes, or the file itself when it has none) one batch of files holds
# for the lookups of their events and camera cells. Memory per batch, not
# per project; small enough to stay flat, large enough that the per-batch
# lookups cost nothing next to the rows.
STREAM_BATCH = 1000


def iter_scoped_files(
    db: Session,
    project: Project,
    *,
    deployment_ids: list[str] | None = None,
    visible_only: bool = True,
) -> Iterator[ScopedFile]:
    """Every media file in scope for ``project``, one at a time, with its
    boxes that count. The one source of every table export.

    **Streamed, never collected.** The query is read ``STREAM_BATCH`` rows
    at a time and regrouped as it goes, so an export holds one batch, not
    the project. Holding it all is what ran a 3.2-million-box export out
    of memory (2026-10-14). The regrouping relies on the ORDER BY ending
    in ``File.id`` (``file_time_order``) and then the detection order, so
    a file's rows are adjacent. The session's identity map holds its
    objects weakly, so a file is freed once its consumer moves on.

    **Which boxes count.** A LEFT JOIN on Detection, and every rule sits in
    that join's condition, never in WHERE: a file whose boxes all fail a
    rule still arrives (with no detections), so the Files table lists it
    as the empty file it is for this export.

    - Threshold + verified override (DEVELOPERS.md), always. There used to
      be an ``apply_threshold=False`` escape hatch for the folder-run
      tables; it produced sheets that disagreed with each other and with
      the Labels step. The complete record is ``addaxai-recognitions.json``.
    - ``visible_only`` keeps a video's boxes to its saved frame, plus
      verified boxes on any frame (``on_visible_frame``): the boxes a user
      can see, open and correct, and the only ones every table but Camtrap
      DP shows. Reading the rest only to drop them cost 50 to 100 times the
      rows on a video project. Camtrap DP passes False: it is the archival
      export and keeps every frame.
    - The species exclusion (``_excluded_box``).
    """
    detection_scope = [
        Detection.file_id == File.id,
        threshold_or_verified(project.counting_threshold),
    ]
    if visible_only:
        detection_scope.append(on_visible_frame())
    removed_by_exclusion = _excluded_box(project)
    if removed_by_exclusion is not None:
        detection_scope.append(~removed_by_exclusion)

    # Outer join on Site so deployments without an assigned site still
    # appear in the export; the CSV serializer emits blank lat/lon cells
    # for those rows, and CamtrapDP / GeoJSON skip them with a separate
    # skipped_deployment_ids list.
    query = (
        select(File, Detection, Deployment, Site, LabelTaxonomy)
        # Two File columns no export builder reads, so they stay out of
        # every row. exif_data: without this, SQLite drags ~70k EXIF blobs
        # through the ORDER BY sorter's temp file and hits SQLITE_FULL
        # ("database or disk is full") on large projects; the camera cells
        # read it in their own query (``_camera_cells_by_file``).
        # frames_processed: a JSON list per video, decoded again on every
        # joined row.
        .options(defer(File.exif_data), defer(File.frames_processed))
        .select_from(File)
        .join(Deployment, File.deployment_id == Deployment.id)
        .outerjoin(Site, Deployment.site_id == Site.id)
        .outerjoin(Detection, and_(*detection_scope))
        .outerjoin(
            LabelTaxonomy, LabelTaxonomy.id == Detection.label_taxonomy_id
        )
        .where(Deployment.project_id == project.id)
        .where(File.file_type.in_(("image", "video")))
        .order_by(
            *file_time_order(newest_first=False), *detection_order_in_file()
        )
        .execution_options(yield_per=STREAM_BATCH)
    )
    # Optional export scope: narrow to a subset of the project's
    # deployments (see resolve_scope_deployment_ids). None = whole project.
    if deployment_ids is not None:
        query = query.where(Deployment.id.in_(deployment_ids))

    result = db.execute(query)
    try:
        current: ScopedFile | None = None
        for file_obj, detection, deployment, site, taxonomy in result:
            if current is None or current.file.id != file_obj.id:
                if current is not None:
                    yield current
                current = ScopedFile(file_obj, deployment, site, [])
            if detection is not None:
                current.detections.append((detection, taxonomy))
        if current is not None:
            yield current
    finally:
        # A consumer that stops early (the XLSX row limit) must not leave
        # the cursor open on the connection.
        result.close()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _batched(files: Iterable[ScopedFile]) -> Iterator[list[ScopedFile]]:
    """``files`` in lists of about ``STREAM_BATCH`` rows, for the lookups
    made once per batch rather than once per project.

    Counted in rows, not files: a batch holds its files' boxes too, and a
    video with every frame kept (Camtrap DP) carries hundreds. A thousand
    such files per batch was a gigabyte of box objects at once (measured
    on 6,000 videos x 425 boxes, 2026-10-15). So memory stays bounded by
    the batch plus the single largest file, whatever the project looks
    like.
    """
    batch: list[ScopedFile] = []
    rows = 0
    for scoped in files:
        batch.append(scoped)
        rows += max(1, len(scoped.detections))
        if rows >= STREAM_BATCH:
            yield batch
            batch, rows = [], 0
    if batch:
        yield batch


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
    """The six formal ranks (variant included), empty string where unknown."""
    if taxonomy is None:
        return ["", "", "", "", "", ""]
    return [
        taxonomy.taxon_class or "",
        taxonomy.taxon_order or "",
        taxonomy.taxon_family or "",
        taxonomy.taxon_genus or "",
        taxonomy.taxon_species or "",
        taxonomy.taxon_variant or "",
    ]


def _full_binomial(
    taxonomy: LabelTaxonomy | None, *, with_variant: bool = True
) -> str | None:
    """"Isoodon macrourus" rebuilt from the rank columns, or None when the
    row has no genus + species to build it from."""
    if taxonomy is None or not (taxonomy.taxon_genus and taxonomy.taxon_species):
        return None
    return format_scientific_name_from_taxonomy_row(
        taxonomy.name,
        taxonomy.taxon_genus,
        taxonomy.taxon_species,
        taxon_variant=taxonomy.taxon_variant if with_variant else None,
        abbreviate=False,
    )


def _export_scientific_name(
    taxonomy: LabelTaxonomy | None, stored: str | None = None
) -> str:
    """The Latin name as every export writes it: "Isoodon macrourus", where
    the app shows "I. macrourus".

    Only a row with both genus and species is ever abbreviated, so only
    that case is rebuilt. Every other row (a rollup to family, a custom
    label, the builtin Person) falls back to ``stored``, the name on the
    detection when the caller has one, and then to the taxonomy row's own
    name; both are already what they should be. The stored
    ``scientific_name`` stays the app's source of truth; the exports
    differ from it on purpose, see ``format_scientific_name_from_taxonomy_row``.
    """
    return (
        _full_binomial(taxonomy)
        or stored
        or (taxonomy.scientific_name if taxonomy else None)
        or ""
    )


def _scientific_name(
    detection: Detection, taxonomy: LabelTaxonomy | None
) -> str:
    """Latin name for the spatial layer: blank for a person or a vehicle."""
    if detection.category != "animal":
        return ""
    return _export_scientific_name(taxonomy)


def _camtrap_taxonomy_name(taxonomy: LabelTaxonomy | None) -> str | None:
    """The taxonomy row's name for Camtrap DP's ``scientificName``.

    A variant row answers the plain binomial: the standard wants a real
    scientific name there, and the variant travels in lifeStage / sex /
    observationComments instead (see ``_camtrap_variant_fields``).
    """
    if taxonomy is None:
        return None
    if taxonomy.taxon_variant:
        return _full_binomial(taxonomy, with_variant=False)
    return _export_scientific_name(taxonomy) or None


# Camtrap DP's enums for the two observation columns a variant can fill.
# External standard (https://camtrap-dp.tdwg.org/data/); values outside
# them fail validation, so anything else goes to observationComments.
# The same vocabularies a person picks from on the Counts page.
_CAMTRAP_LIFE_STAGES = frozenset(LIFE_STAGES)
_CAMTRAP_SEXES = frozenset(SEXES)


def _camtrap_variant_fields(
    taxonomy: LabelTaxonomy | None,
) -> tuple[str, str, str]:
    """Map a taxonomy row's variant onto Camtrap DP's observation columns.

    Returns ``(life_stage, sex, comment)``: the variant fills lifeStage
    or sex when it fits their vocabulary, else it rides along as a
    comment so it is never lost and never fails validation.
    """
    variant = (taxonomy.taxon_variant or "") if taxonomy else ""
    if not variant:
        return "", "", ""
    if variant in _CAMTRAP_LIFE_STAGES:
        return variant, "", ""
    if variant in _CAMTRAP_SEXES:
        return "", variant, ""
    return "", "", f"variant: {variant}"


def _camtrap_cohort_fields(
    taxonomy: LabelTaxonomy | None, obs: EventObservation, event_notes: str | None
) -> tuple[str, str, str, str]:
    """Camtrap DP's ``(lifeStage, sex, behavior, observationComments)`` for
    one cohort row.

    The model's variant wins over what a person set on the row when both
    speak (a variant is the taxon's own claim; that cannot happen for a
    model without variants). NULL exports blank, never "unknown": the
    enums have no such value. The event's note goes to observationComments
    of each of its rows (the standard has no event-level comment), after
    the variant comment when there is one.
    """
    life_stage, sex, variant_comment = _camtrap_variant_fields(taxonomy)
    comments = "; ".join(c for c in (variant_comment, event_notes or "") if c)
    return (
        life_stage or obs.life_stage or "",
        sex or obs.sex or "",
        obs.behavior or "",
        comments,
    )


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
    db: Session, project: Project, deployment_ids: list[str] | None = None
) -> tuple[list[str], Iterator[list[Any]]]:
    """
    Build `(headers, rows)` for the flat Detections export. ``rows`` is
    a generator: nothing is read until a writer pulls it, and it is read
    once.

    Grain: one row per detection (the box view). Carries the context block
    (site, deployment, event, file) so a row reads on its own, and the FK
    id set inside it for direct linkability. Files with no detections do
    not appear here; per-event counts live in counts.csv.

    `event_id` is blank when the file has no event, so every non-empty
    value resolves in the events table. See the note on `_FILES_HEADERS`.

    **Only boxes the user can reach.** A video stores detections on every
    sampled frame, but the app writes and shows exactly one frame per
    video, so a box on any other frame has no picture anywhere: it cannot
    be seen in the Labels grid, filtered to, or relabelled. Emitting them
    here handed users a species list they could not act on. The rows come
    from ``iter_scoped_files`` with ``visible_only`` on, so those boxes
    never arrive. The per-frame boxes are not lost:
    ``addaxai-recognitions.json`` carries all of them, with their frame
    numbers, which is the file built to be the complete record. Verified
    boxes pass on any frame, per the usual exception.
    """
    return _FLAT_DETECTION_HEADERS, _detection_rows(db, project, deployment_ids)


def _detection_rows(
    db: Session, project: Project, deployment_ids: list[str] | None
) -> Iterator[list[Any]]:
    tz_name = project.timezone
    files = iter_scoped_files(db, project, deployment_ids=deployment_ids)
    for batch in _batched(files):
        event_map = _events_by_file(db, [sf.file.id for sf in batch])
        for file_obj, deployment, site, detections in batch:
            event = event_map.get(file_obj.id)
            # Built once per file, not once per box: the same cells repeat on
            # every detection row of the file.
            context = [
                *_site_cells(site),
                *_deployment_cells(deployment),
                *_event_cells(event, tz_name),
                file_obj.id,
                *_file_cells(file_obj, deployment, tz_name),
            ]
            for detection, taxonomy in detections:
                yield [detection.id, *context] + _detection_cells(detection, taxonomy)


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
        _export_scientific_name(taxonomy, detection.scientific_name),
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
# (date span + trap-nights). The full site and deployment record; the
# other tables repeat its identity columns (the context block) and point
# back here with deployment_id for the rest. Mirrors the Camtrap-DP
# deployments table.
def _format_tags(tags: dict | None) -> str:
    """Serialize a tags dict to one cell as pipe-separated key:value pairs,
    matching the Camtrap-DP deploymentTags convention (e.g. "season:wet | access:4x4")."""
    if not tags:
        return ""
    return " | ".join(f"{k}:{v}" for k, v in tags.items())


def _site_cells(site: Site | None) -> list[Any]:
    """The site part of the context block; blanks for a deployment with no
    site, which is what the CSV already did for latitude / longitude."""
    if site is None:
        return ["", "", "", ""]
    return [site.name, site.latitude, site.longitude, _format_tags(site.tags)]


def _deployment_cells(deployment: Deployment | None) -> list[Any]:
    """The deployment part of the context block."""
    if deployment is None:
        return ["", "", "", ""]
    return [
        deployment.id,
        deployment.start_date_local.isoformat() if deployment.start_date_local else "",
        deployment.end_date_local.isoformat() if deployment.end_date_local else "",
        _format_tags(deployment.tags),
    ]


def _event_cells(event: Event | None, tz_name: str) -> list[Any]:
    """The event part of the context block. All blank when the file has no
    event, never a stand-in id; see the note on ``_FILES_HEADERS``."""
    if event is None:
        return ["", "", ""]
    return [
        event.id,
        _iso_datetime(event.event_start_local, tz_name),
        _iso_datetime(event.event_end_local, tz_name),
    ]


def _file_cells(
    file_obj: File, deployment: Deployment | None, tz_name: str
) -> list[Any]:
    """The file part of the context block, after the file's own id. The
    pixel size is blank, never 0, when it was not recorded (videos from
    before the best-frame size was stored)."""
    return [
        _relative_path(file_obj, deployment),
        _iso_datetime(file_obj.captured_at_local, tz_name),
        file_obj.width_px if file_obj.width_px is not None else "",
        file_obj.height_px if file_obj.height_px is not None else "",
    ]


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
    "paired_cameras",
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
    full location / effort record; the other tables carry its identity
    columns and point back here with deployment_id for the rest."""
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
                "true" if dep.paired_cameras else "false",
                dep.notes or "",
                _format_tags(dep.tags),
            ]
        )

    return _DEPLOYMENTS_HEADERS, rows


# ---------------------------------------------------------------------------
# Files rows (one row per media file, the media / membership table)
# ---------------------------------------------------------------------------

# One row per file, including empties. This is the home for "which files
# had no detections" (category=blank) and "which files are in which
# event" (event_id), instead of faking blank rows in the detections table.
# Mirrors the Camtrap-DP media table. Carries the context block, so the
# site and the deployment dates sit on the row.
#
# event_id is blank when a file has no event, never a stand-in id. Every
# image / video lands in exactly one cluster once events are generated
# (date-less files become singleton events with NULL bounds), so in
# practice the column is always populated. If it ever is blank, that means
# what it says: no event. Every non-empty value resolves in the events
# table.
_FILES_HEADERS = [
    "file_id",
    *_SITE_CONTEXT_HEADERS,
    *_DEPLOYMENT_CONTEXT_HEADERS,
    *_EVENT_CONTEXT_HEADERS,
    "file_type",
    # The file block, plus the absolute path that only this table carries.
    "relative_path",
    "absolute_path",
    "datetime",
    "image_width",
    "image_height",
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
    "taxon_variant",
    "scientific_name",
    "common_name",
    "is_verified",
    # The two marks a person can put on a file on the Labels page.
    "is_favorited",
    "is_flagged",
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
    ``iter_scoped_files``: that query defers ``File.exif_data`` so
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
) -> tuple[list[str], Iterator[list[Any]]]:
    """Build `(headers, rows)` for the Files export: one row per media file,
    including files with no detections (the effort table). `event_id`
    answers "which files are in this event". ``rows`` is a generator, read
    once, like ``build_detection_rows``'."""
    return _FILES_HEADERS, _files_rows(db, project, deployment_ids)


def _files_rows(
    db: Session, project: Project, deployment_ids: list[str] | None
) -> Iterator[list[Any]]:
    tz_name = project.timezone
    files = iter_scoped_files(db, project, deployment_ids=deployment_ids)
    for batch in _batched(files):
        file_ids = [sf.file.id for sf in batch]
        event_map = _events_by_file(db, file_ids)
        camera_map = _camera_cells_by_file(db, file_ids)
        for file_obj, deployment, site, detections in batch:
            event = event_map.get(file_obj.id)
            relative_path, *file_rest = _file_cells(file_obj, deployment, tz_name)
            yield [
                file_obj.id,
                *_site_cells(site),
                *_deployment_cells(deployment),
                *_event_cells(event, tz_name),
                file_obj.file_type or "",
                relative_path,
                file_obj.file_path,
                *file_rest,
                *camera_map.get(file_obj.id, _BLANK_CAMERA_CELLS),
                file_obj.observation_type or "",
                *_strongest_species_cells(project, file_obj, detections),
                "TRUE" if file_obj.verified else "FALSE",
                "TRUE" if file_obj.favorited else "FALSE",
                "TRUE" if file_obj.flagged else "FALSE",
                file_obj.notes or "",
            ]


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
    user can actually open. The gate is the ``visible_only`` condition in
    ``iter_scoped_files``, so ``detections`` holds only those boxes.

    The threshold is passed explicitly even though ``iter_scoped_files``
    already applied the same predicate in SQL. Re-applying it costs one pass
    over a handful of already-loaded objects and keeps these columns tied to
    the same threshold ``observation_type`` uses, so a later change to that
    query cannot silently desynchronise the two.

    Names come off the detection, not off the joined ``LabelTaxonomy``. That is
    the convention ``_detection_cells`` uses, so a Files row and its
    detections.csv rows read identically with no join. (The other convention in
    this module, ``_scientific_name``, blanks non-animals and reads the taxonomy
    row; it serves the Camtrap and spatial builders.) The practical effect is
    that a box with no species still names itself: ``Person``, ``Vehicle``,
    ``Animal``, per ``resolve_label_names``.
    """
    best = strongest_passing_detection(
        [det for det, _tax in detections], project.counting_threshold
    )
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
        _export_scientific_name(taxonomy, best.scientific_name),
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
    *_SITE_CONTEXT_HEADERS,
    *_DEPLOYMENT_CONTEXT_HEADERS,
    # The deployment's effort, so count per trap night is one formula on
    # the row. Counts only: on a per-box or per-file row it would invite
    # sums that mean nothing.
    "trap_nights",
    "event_start",
    "event_end",
    "category",
    "classification_label",
    "taxon_class",
    "taxon_order",
    "taxon_family",
    "taxon_genus",
    "taxon_species",
    "taxon_variant",
    "scientific_name",
    "common_name",
    # The human-confirmed count, falling back to the AI's count when the
    # event isn't confirmed. The two it is made of follow, so an export
    # taken after review still separates what the AI said from what a
    # person set (a user comparing the two asked for exactly that).
    "count",
    "ai_count",
    "human_count",
    # What a person recorded about the individuals on this row, blank
    # when unknown. One species can have several rows in an event (4 adult
    # males, 2 juveniles); the species total is their sum.
    "sex",
    "life_stage",
    "behavior",
    # The event's note, repeated on each of its rows.
    "event_notes",
    "is_confirmed",
]


def build_observation_rows(
    db: Session,
    project: Project,
    deployment_ids: list[str] | None = None,
) -> tuple[list[str], Iterator[list[Any]]]:
    """
    Build `(headers, rows)` for the event-level Observations export.
    ``rows`` is a generator, read once.

    Grain: one row per event x cohort (a species, or one demographic group
    of it), carrying the effective count (human-confirmed if set, else the
    AI count) and the event sign-off. Count-0 rows (a species a human
    removed) are skipped. This maps to the Counts page; the per-detection
    grain lives in the Detections export.
    """
    return _OBSERVATIONS_HEADERS, _observation_rows(db, project, deployment_ids)


def _observation_rows(
    db: Session, project: Project, deployment_ids: list[str] | None
) -> Iterator[list[Any]]:
    from app.api.crud.trap_nights import compute_trap_nights_for_deployments

    tz_name = project.timezone

    query = (
        select(EventObservation, Event, Deployment, Site, LabelTaxonomy)
        .join(Event, Event.id == EventObservation.event_id)
        .join(Deployment, Deployment.id == Event.deployment_id)
        .outerjoin(Site, Site.id == Deployment.site_id)
        .outerjoin(
            LabelTaxonomy,
            LabelTaxonomy.id == EventObservation.label_taxonomy_id,
        )
        .where(Deployment.project_id == project.id)
        .order_by(Event.event_start_local.asc(), Event.id)
    )
    if deployment_ids is not None:
        query = query.where(Deployment.id.in_(deployment_ids))

    # Trap nights for the deployments the rows will name, asked for before
    # the rows are streamed rather than collected from them.
    seen_deployment_ids = sorted(
        db.scalars(
            query.with_only_columns(Deployment.id).distinct().order_by(None)
        )
    )
    trap_nights = (
        compute_trap_nights_for_deployments(db, seen_deployment_ids)
        if seen_deployment_ids
        else {}
    )

    result = db.execute(query.execution_options(yield_per=STREAM_BATCH))
    try:
        for obs, event, deployment, site, taxonomy in result:
            count = obs.effective_count
            if count <= 0:
                continue
            yield [
                event.id,
                *_site_cells(site),
                *_deployment_cells(deployment),
                trap_nights.get(deployment.id, ""),
                _iso_datetime(event.event_start_local, tz_name),
                _iso_datetime(event.event_end_local, tz_name),
                obs.category,
                obs.label or "",
                *_taxon_ranks(taxonomy),
                _export_scientific_name(taxonomy),
                (taxonomy.common_name if taxonomy else "") or "",
                count,
                obs.max_n,
                obs.human_count if obs.human_count is not None else "",
                obs.sex or "",
                obs.life_stage or "",
                obs.behavior or "",
                event.notes or "",
                "TRUE" if event.confirmed else "FALSE",
            ]
    finally:
        result.close()


# ---------------------------------------------------------------------------
# Summary rows (one row per species / category, with counts)
# ---------------------------------------------------------------------------

# The overview table: what was found and how much of it. One row per
# species (or per person / vehicle / unclassified animal), the same name
# block as the other tables, then one count per other table. Built for the
# person who opens the workbook and wants the answer on the first sheet.
_SUMMARY_HEADERS = [
    "detection_category",
    "classification_label",
    "taxon_class",
    "taxon_order",
    "taxon_family",
    "taxon_genus",
    "taxon_species",
    "taxon_variant",
    "scientific_name",
    "common_name",
    "n_images",
    "n_videos",
    "n_detections",
    "n_events",
    "n_individuals",
]


def _summary_key(category: str, label: str | None) -> tuple[str, str]:
    """What identifies a Summary row: the detector category plus the species
    label, blank when nothing was classified (person, vehicle, unclassified
    animal).

    A label that only repeats the category is no species either. The
    builtin ``animal`` taxonomy row exists exactly for unclassified boxes,
    so a detection can carry it as a literal label, and the Counts table
    names every unclassified box by its category (``COALESCE(label,
    category)``). Folding both spellings onto the blank key is what keeps
    ``n_individuals`` on the same row as ``n_detections``.
    """
    return (category, "" if label in (None, "", category) else label)


def build_summary_rows(
    db: Session,
    project: Project,
    deployment_ids: list[str] | None = None,
) -> tuple[list[str], list[list[Any]]]:
    """
    Build `(headers, rows)` for the Summary export.

    One rule: a row is a species / category that appears in the Detections
    table, and every count column counts one other table for it.

    - ``n_detections``: its rows in the Detections table.
    - ``n_images`` / ``n_videos``: the files those rows sit on, by type.
    - ``n_events``: the events those files belong to.
    - ``n_individuals``: the sum of ``count`` in the Counts table (the
      confirmed number where one was set, else the AI's MaxN).

    The rows come from the same ``iter_scoped_files`` stream and scope as
    ``build_detection_rows``, so the two tables cannot disagree. That is
    also why a species that only exists in the Counts table (added by hand
    on the Counts page, or an excluded species with a leftover observation
    row) gets no row here: it has no detection to count.

    The one departure from the Detections table: a box a person rejected
    ("Mark false", label ``false detection``) keeps its Detections row as
    the honest record, but it was not found, so it gets no Summary row.
    The Counts table skips it the same way.

    Both sides key through ``_summary_key``, which treats a label that only
    repeats the category as "no species"; see there for why.

    Built in one pass over the stream, holding one counter per species.
    Each file arrives once, so the files per species are counted per file
    without keeping their ids; the events per species do need a set of
    ``(species, event)`` pairs, which grows with the events in scope, not
    with the boxes.
    """
    name_cells: dict[tuple[str, str], list[Any]] = {}
    n_images: dict[tuple[str, str], int] = defaultdict(int)
    n_videos: dict[tuple[str, str], int] = defaultdict(int)
    events: dict[tuple[str, str], set[str]] = defaultdict(set)
    n_detections: dict[tuple[str, str], int] = defaultdict(int)

    files = iter_scoped_files(db, project, deployment_ids=deployment_ids)
    for batch in _batched(files):
        event_map = _events_by_file(db, [sf.file.id for sf in batch])
        for file_obj, _deployment, _site, detections in batch:
            event = event_map.get(file_obj.id)
            keys_in_file: set[tuple[str, str]] = set()
            for detection, taxonomy in detections:
                if is_non_label(detection.label):
                    continue
                key = _summary_key(detection.category, detection.label)
                if key not in name_cells:
                    name_cells[key] = [
                        *_taxon_ranks(taxonomy),
                        _export_scientific_name(taxonomy, detection.scientific_name),
                        detection.common_name or "",
                    ]
                n_detections[key] += 1
                keys_in_file.add(key)
            per_type = n_videos if file_obj.file_type == "video" else n_images
            for key in keys_in_file:
                per_type[key] += 1
                if event is not None:
                    events[key].add(event.id)

    individuals: dict[tuple[str, str], int] = defaultdict(int)
    obs_headers, obs_rows = build_observation_rows(db, project, deployment_ids)
    cat_i = obs_headers.index("category")
    label_i = obs_headers.index("classification_label")
    count_i = obs_headers.index("count")
    for obs_row in obs_rows:
        key = _summary_key(obs_row[cat_i], obs_row[label_i])
        individuals[key] += obs_row[count_i]

    rows: list[list[Any]] = []
    for key, names in name_cells.items():
        category, label = key
        rows.append(
            [
                category,
                label,
                *names,
                n_images[key],
                n_videos[key],
                n_detections[key],
                len(events[key]),
                individuals[key],
            ]
        )
    # Most seen first; ties read in a stable alphabetical order.
    det_i = _SUMMARY_HEADERS.index("n_detections")
    rows.sort(key=lambda r: (-r[det_i], r[0], r[1]))
    return _SUMMARY_HEADERS, rows


def build_spreadsheet_sheets(
    db: Session,
    project: Project,
    deployment_ids: list[str] | None = None,
) -> list[tuple[str, list[str], Iterable[list[Any]]]]:
    """The tables that make up the project Export page's combined
    spreadsheet: Summary, Counts, Detections, Files, and Deployments. The
    folder-run Save step writes its own three-sheet workbook (Summary +
    Files + Detections) from the same row builders.

    **Sheet order is the docs order, broadest question first.** A
    workbook opens on its first sheet, so that sheet is what a user
    takes the file to contain. Deployments used to lead, which is one
    row per camera of locations and effort, and reading no further is
    how a beta tester concluded the export held nothing but deployment
    metadata. Summary leads because "which species, and how many" is the
    question the file is opened for; Counts follows as the analysis-ready
    table `docs/docs/reference/exports.md` tells people to start with.
    Keep the two in step. The CSV / TSV export writes these same five
    tables as five files, in this order.

    ``deployment_ids`` narrows every sheet to a subset of the project's
    deployments; None exports everything.

    The big sheets are generators: each runs its own query when its writer
    reaches it, one after the other, so memory holds one batch rather than
    the project. That reads the scoped files three times (Summary, then
    Detections, then Files), the price of never holding them."""
    summary_headers, summary_rows = build_summary_rows(db, project, deployment_ids)
    obs_headers, obs_rows = build_observation_rows(db, project, deployment_ids)
    det_headers, det_rows = build_detection_rows(db, project, deployment_ids)
    files_headers, files_rows = build_files_rows(db, project, deployment_ids)
    dep_headers, dep_rows = build_deployments_rows(db, project, deployment_ids)
    return [
        ("Summary", summary_headers, summary_rows),
        ("Counts", obs_headers, obs_rows),
        ("Detections", det_headers, det_rows),
        ("Files", files_headers, files_rows),
        ("Deployments", dep_headers, dep_rows),
    ]


# ---------------------------------------------------------------------------
# Spatial layers
# ---------------------------------------------------------------------------


def build_spatial_layers(
    db: Session,
    project: Project,
    deployment_ids: list[str] | None = None,
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
    # One pass over the scoped files for both layers' counts: boxes per
    # deployment, and boxes per (site, species). Both hold only what the
    # user can reach (``visible_only``), so a map bubble agrees with the
    # Labels grid and detections.csv, and the species points of a site add
    # up to its deployments.
    #
    # The species layer aggregates classified detections only -- person,
    # vehicle, unclassified, and blank rows have no label and don't belong
    # on a species map. It carries the common label, the latin name, and
    # the taxon ranks for readable map labelling and taxonomic filtering
    # in GIS.
    det_count_by_dep: dict[str, int] = defaultdict(int)
    summary_bucket: dict[tuple[str, str], dict[str, Any]] = {}
    for _file, deployment, site, detections in iter_scoped_files(
        db, project, deployment_ids=deployment_ids
    ):
        det_count_by_dep[deployment.id] += len(detections)
        if site is None:
            continue
        for detection, taxonomy in detections:
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

    # Include project deployments that had no in-scope files (so the
    # deployments layer lists every deployment, not only those with
    # detections). Outer join on Site so null-site deployments are not
    # filtered out here; we skip them below when building features
    # (GeoJSON needs coordinates) and record them separately.
    deployments_query = (
        db.query(Deployment, Site)
        .outerjoin(Site, Deployment.site_id == Site.id)
        .filter(Deployment.project_id == project.id)
    )
    if deployment_ids is not None:
        deployments_query = deployments_query.filter(Deployment.id.in_(deployment_ids))
    all_deployments = deployments_query.all()

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

    # Species summary: one point per (site, species), from the buckets
    # counted above.
    trap_days_by_site: dict[str, int] = defaultdict(int)
    site_coords: dict[str, tuple[float, float]] = {}
    for deployment, site in all_deployments:
        if site is None:
            continue
        trap_days_by_site[site.id] += trap_days_by_dep.get(deployment.id, 0)
        site_coords[site.id] = (site.longitude, site.latitude)

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
                    "taxon_variant": ranks[5],
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


class CamtrapDpBuilder:
    """The Camtrap DP tables, built while they are written.

    Camtrap DP is the archival export, so it reads every frame
    (``visible_only=False``) and on a video project it is the largest
    table export there is. Nothing is collected: the writer pulls

    1. ``deployments_rows``, one per camera, built up front,
    2. ``media()``, one ``(media_row, observation_rows)`` per file, from
       the scoped stream,
    3. ``event_observations()``, one per species per event, and
    4. ``datapackage()``,

    in that order. 3 and 4 summarise what 2 saw (events, taxa, dates), so
    they can only be asked for after 2 is exhausted.

    Deployments without a site are skipped because Camtrap DP requires
    lat/lon; ``skipped_deployment_ids`` lists them for the caller.
    """

    def __init__(
        self,
        db: Session,
        project: Project,
        deployment_ids: list[str] | None = None,
    ) -> None:
        self._db = db
        self._project = project
        self._deployment_ids = deployment_ids
        self._tz_name = project.timezone
        self._classified_by = _classifier_label(project)
        self._not_reviewed = f"{self._classified_by}, not reviewed"

        self._observed_taxa: dict[str, tuple[LabelTaxonomy | None, str]] = {}
        # Events in scope, for the event-level (per-species count) rows
        # emitted after the per-file media rows.
        self._events_in_scope: dict[str, dict[str, str]] = {}
        self._first_captured: datetime | None = None
        self._last_captured: datetime | None = None
        self._sites_seen: dict[str, Site] = {}
        self._media_done = False

        self.deployments_rows: list[list[Any]] = []
        self.skipped_deployment_ids: list[str] = []
        self._build_deployments()

    def _build_deployments(self) -> None:
        """deployments.csv: every deployment in scope with a site.
        Null-site deployments are skipped because Camtrap DP requires
        deploymentLocation (lat/lon)."""
        tz_name = self._tz_name
        deployments_query = (
            self._db.query(Deployment, Site)
            .outerjoin(Site, Deployment.site_id == Site.id)
            .filter(Deployment.project_id == self._project.id)
        )
        if self._deployment_ids is not None:
            deployments_query = deployments_query.filter(
                Deployment.id.in_(self._deployment_ids)
            )
        for deployment, site in deployments_query.order_by(
            Deployment.start_date_local
        ).all():
            if site is None:
                self.skipped_deployment_ids.append(deployment.id)
                continue
            self._sites_seen[site.id] = site
            camera_model = deployment.camera_model or ""
            camera_id = deployment.camera_serial or deployment.camera_model or deployment.id
            # Camtrap DP defines a deployment as one camera. A paired-cameras
            # deployment holds several, so say so in deploymentTags.
            deployment_tags = dict(deployment.tags or {})
            if deployment.paired_cameras:
                deployment_tags["paired_cameras"] = "true"
            # Row order must match _CAMTRAP_DEPLOYMENTS_HEADERS exactly.
            self.deployments_rows.append(
                [
                    deployment.id,                                                # deploymentID
                    "",                                                           # locationID
                    site.name or "",                                              # locationName
                    site.latitude,                                                # latitude
                    site.longitude,                                               # longitude
                    "",                                                      # coordinateUncertainty
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
                    "",                                                          # detectionDistance
                    "",                                                           # timestampIssues
                    "",                                                           # baitUse
                    "",                                                           # featureType
                    site.habitat_type or "",                                      # habitat
                    "",                                                           # deploymentGroups
                    _format_tags(deployment_tags),                                # deploymentTags
                    deployment.notes or "",                                     # deploymentComments
                ]
            )

    def media_count(self) -> int:
        """How many rows ``media()`` will yield, for a progress bar: files
        in scope with a capture date, in a deployment with a site."""
        query = (
            select(func.count(File.id))
            .join(Deployment, File.deployment_id == Deployment.id)
            .where(Deployment.project_id == self._project.id)
            .where(Deployment.site_id.isnot(None))
            .where(File.file_type.in_(("image", "video")))
            .where(File.captured_at_local.isnot(None))
        )
        if self._deployment_ids is not None:
            query = query.where(Deployment.id.in_(self._deployment_ids))
        return self._db.scalar(query) or 0

    def media(self) -> Iterator[tuple[list[Any], list[list[Any]]]]:
        """One ``(media_row, observation_rows)`` per file in scope.

        Files whose deployment has no site were already skipped at the
        deployments.csv stage (Camtrap DP requires lat/lon), so their media
        and observations are skipped here too and the package stays
        consistent.
        """
        tz_name = self._tz_name
        skipped_dep_set = set(self.skipped_deployment_ids)
        files_without_date = 0
        files = iter_scoped_files(
            self._db,
            self._project,
            deployment_ids=self._deployment_ids,
            visible_only=False,
        )
        for batch in _batched(files):
            file_ids = [sf.file.id for sf in batch]
            event_map = _events_by_file(self._db, file_ids)
            excluded_only = _files_with_excluded_boxes(self._db, self._project, file_ids)
            for file_obj, deployment, _site, detections in batch:
                if deployment.id in skipped_dep_set:
                    continue
                # media.timestamp and observations.eventStart / eventEnd are
                # required datetime fields in the Camtrap DP schema, so a file
                # with no capture date cannot be represented: an empty string
                # there fails validation in the camtrapdp R package and GBIF
                # ingestion, poisoning the whole package. The export dialog
                # warns about these via /files-without-date.
                if file_obj.captured_at_local is None:
                    files_without_date += 1
                    continue
                if (
                    self._first_captured is None
                    or file_obj.captured_at_local < self._first_captured
                ):
                    self._first_captured = file_obj.captured_at_local
                if (
                    self._last_captured is None
                    or file_obj.captured_at_local > self._last_captured
                ):
                    self._last_captured = file_obj.captured_at_local

                # Row order must match _CAMTRAP_MEDIA_HEADERS exactly.
                media_row = [
                    file_obj.id,                                        # mediaID
                    deployment.id,                                      # deploymentID
                    "activityDetection",                                # captureMethod
                    _iso_datetime(file_obj.captured_at_local, tz_name), # timestamp
                    file_obj.file_path,                                 # filePath
                    "false",                                            # filePublic
                    "",                                                 # fileName
                    _media_type_for(file_obj.file_format),              # fileMediatype
                    "",                                                 # exifData
                    "true" if file_obj.favorited else "",               # favorite
                    "",                                                 # mediaComments
                ]
                yield media_row, self._file_observations(
                    file_obj,
                    deployment,
                    detections,
                    event_map.get(file_obj.id),
                    excluded_only=file_obj.id in excluded_only,
                )

        if files_without_date:
            logger.info(
                f"CamtrapDP export: left out {files_without_date} file(s) with no "
                f"capture date (schema requires a timestamp per record)"
            )
        self._media_done = True

    def _file_observations(
        self,
        file_obj: File,
        deployment: Deployment,
        detections: list[tuple[Detection, LabelTaxonomy | None]],
        event: Event | None,
        *,
        excluded_only: bool,
    ) -> list[list[Any]]:
        """The media-level observation rows of one file."""
        tz_name = self._tz_name
        # Unlike files.csv / detections.csv, a missing event here falls back
        # to the file id on purpose. Camtrap DP has no events resource:
        # eventID is a grouping key inside observations.csv, paired with the
        # eventStart / eventEnd written just below (the file's own capture
        # time). So an unclustered file is its own single-media event and the
        # row stays self-consistent. A blank eventID would instead merge every
        # unclustered file into one event under a GROUP BY.
        event_id = event.id if event else file_obj.id
        event_start = _iso_datetime(
            event.event_start_local if event else file_obj.captured_at_local, tz_name
        )
        event_end = _iso_datetime(
            event.event_end_local if event else file_obj.captured_at_local, tz_name
        )
        captured_iso = _iso_datetime(file_obj.captured_at_local, tz_name)
        if event is not None:
            self._events_in_scope.setdefault(
                event_id,
                {
                    "deployment_id": deployment.id,
                    "event_start": event_start or captured_iso,
                    "event_end": event_end or captured_iso,
                    "notes": event.notes or "",
                },
            )

        # A box a person rejected (X, or a relabel to a non-label class such
        # as a model's "non-animal") is not an observation. Camtrap DP has
        # no place for it: the row would read observationType "animal" with
        # "false detection" as its scientificName. Dropping it here also
        # lets a file whose only box was rejected take the blank row below,
        # which is what its observation_type already says.
        detections = [(d, t) for d, t in detections if not is_non_label(d.label)]

        # The boxes about to be written are the honest test for "is this
        # file blank". This used to also short-circuit on the stored
        # `observation_type == "blank"`, which was near-equivalent while
        # that column was derived over every frame. It is not equivalent
        # now: a video whose best frame is empty but which still has
        # passing boxes on other frames would take the blank branch and
        # lose every per-box row from an archival export.
        #
        # A file left without boxes by the species exclusion is not blank
        # either: it shows an animal the user chose not to export. It keeps
        # its media row and gets no observation, which in Camtrap DP reads
        # as "not annotated" rather than "nothing there".
        if not detections and excluded_only:
            return []
        if not detections:
            return [
                _camtrap_blank_row(
                    file_obj,
                    deployment.id,
                    event_id,
                    event_start or captured_iso,
                    event_end or captured_iso,
                    self._not_reviewed,
                )
            ]

        rows: list[list[Any]] = []
        for detection, taxonomy in detections:
            # Media-level rows are the per-box detections (one row per
            # bounding box). Box-less species are carried by the
            # event-level rows, so skip any here.
            if detection.bbox_x is None:
                continue
            obs_type = _obs_type_from_category(detection.category)
            sci_name = (
                (_camtrap_taxonomy_name(taxonomy) or "")
                if detection.category == "animal"
                else ""
            )
            species_name = _species_label(detection, taxonomy)
            if detection.category == "animal" and species_name:
                self._observed_taxa.setdefault(species_name, (taxonomy, species_name))

            obs_id_prefix = "obs-human" if detection.verified else "obs-ai"
            method = "human" if detection.verified else "machine"
            comments = "Human identification" if detection.verified else self._not_reviewed
            life_stage, sex, variant_comment = _camtrap_variant_fields(taxonomy)
            if variant_comment:
                comments = (
                    f"{comments}; {variant_comment}"
                    if comments
                    else variant_comment
                )
            prob = (
                round(detection.label_confidence, 6)
                if detection.category == "animal" and detection.label_confidence is not None
                else ""
            )

            # Row order must match _CAMTRAP_OBS_HEADERS exactly. One
            # media-level row per bounding box (observationLevel="media").
            rows.append(
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
                    life_stage,                            # lifeStage (enum)
                    sex,                                   # sex (enum)
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
                    self._classified_by,                   # classifiedBy
                    "",                                    # classificationTimestamp
                    prob,                                  # classificationProbability
                    "",                                    # observationTags
                    comments,                              # observationComments
                ]
            )
        return rows

    def event_observations(self) -> Iterator[list[Any]]:
        """Event-level observations: one row per species per event carrying
        the effective count (human override, else MaxN). Mutually exclusive
        and summable, the Camtrap DP shape for "N individuals in this
        sequence". The per-box media rows remain for spatial detail.
        Chunked over the event ids to stay under SQLite's bound-parameter
        limit (a large project can have >32k events in scope)."""
        if not self._media_done:
            raise RuntimeError("event_observations() needs media() to be read first")
        for chunk in iter_id_chunks(self._events_in_scope.keys()):
            for obs, taxonomy in (
                self._db.query(EventObservation, LabelTaxonomy)
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
                ctx = self._events_in_scope[obs.event_id]
                sci_name = _camtrap_taxonomy_name(taxonomy) or (obs.label or "")
                species_name = (taxonomy.name if taxonomy else None) or obs.label
                if obs.category == "animal" and species_name:
                    self._observed_taxa.setdefault(species_name, (taxonomy, species_name))
                yield _camtrap_event_row(
                    obs, ctx, count, sci_name, self._classified_by, taxonomy
                )

    def datapackage(self) -> dict[str, Any]:
        """``datapackage.json``: coverage and taxa of everything written."""
        if not self._media_done:
            raise RuntimeError("datapackage() needs media() to be read first")
        return _build_datapackage(
            self._project,
            self._sites_seen.values(),
            self._first_captured,
            self._last_captured,
            self._observed_taxa,
        )


def _camtrap_event_row(
    obs: EventObservation,
    ctx: dict[str, str],
    count: int,
    sci_name: str,
    classified_by: str,
    taxonomy: LabelTaxonomy | None,
) -> list[Any]:
    """Event-level observation row (per species per event, no bbox).

    Order must match _CAMTRAP_OBS_HEADERS exactly. classificationMethod is
    "human" when the count was set by a person, else "machine" (MaxN).
    """
    method = "human" if obs.human_count is not None else "machine"
    life_stage, sex, behavior, comments = _camtrap_cohort_fields(
        taxonomy, obs, ctx["notes"]
    )
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
        life_stage,                       # lifeStage
        sex,                              # sex
        behavior,                         # behavior
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
        comments,                         # observationComments
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

    # Variant rows contribute their plain binomial at species rank, so
    # adult and juvenile of one species produce a single taxon entry
    # (deduplicated by scientificName; first vernacular name wins).
    taxonomic: list[dict[str, Any]] = []
    seen_scientific: set[str] = set()
    for species_name in sorted(observed_taxa):
        taxonomy, _ = observed_taxa[species_name]
        sci = _camtrap_taxonomy_name(taxonomy) or species_name
        if sci in seen_scientific:
            continue
        seen_scientific.add(sci)
        entry: dict[str, Any] = {"scientificName": sci}
        if taxonomy and taxonomy.level:
            entry["taxonRank"] = (
                "species" if taxonomy.level == "variant" else taxonomy.level
            )
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
