"""
Export CRUD — scoping query plus row / layer / CamTrap DP table builders.

The serializers in ``export_formats.py`` don't touch the DB; everything
here does. Row shapes line up 1:1 with the column orders declared in
``export_formats.py``.

Conventions
-----------

* ``Detection.confidence >= project.detection_threshold OR Detection.verified``
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

from sqlalchemy import Row, and_, func, or_, select
from sqlalchemy.orm import Session

from app.api.crud.export_formats import slugify
from app.models import (
    Deployment,
    Detection,
    Event,
    File,
    LabelTaxonomy,
    Project,
    Site,
    event_files,
)
from app.utils.datetime_serialization import to_local_iso_with_offset

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

# Flat observations CSV. One row per detection (the fine-grained grain:
# nothing is aggregated away — a user can re-aggregate by groupby, but
# can't un-aggregate a summary). Columns left-to-right: ids, where/when,
# the detector stage, the classifier stage, checked.
#
#   detection_id    — the detection row id; empty on a blank-file row.
#   file_id, event_id — foreign keys for joining back to files / events.
#                     A file with no event falls back to its own id, so
#                     event_id is never empty. Event bounds aren't stored;
#                     they're min/max of `datetime` per event_id.
#   relative_path   — the file's path relative to its deployment's source
#                     folder (forward slashes). Disambiguates duplicate
#                     filenames across cameras and stays portable. Falls
#                     back to the bare filename when the deployment folder
#                     is unknown or the file sits outside it.
#   detection_category    — detector class: animal / person / vehicle (or
#                     "blank" for an empty file).
#   detection_confidence  — detector (MegaDetector) score for the box.
#   bbox_*          — normalized [0,1] box; empty for event-level / blank.
#   frame_number    — video frame index; empty for images.
#   classification_label — the assigned species identification, by model
#                     or human (provenance-neutral; see is_verified).
#                     Empty when nothing was classified (person, vehicle,
#                     unclassified animal, blank).
#   classification_confidence — score for that label: the classifier's
#                     score, or 1.0 when a human assigned it.
#   taxon_*         — formal ranks from label_taxonomy; empty where the
#                     label has no (or partial) taxonomy. The full Latin
#                     name is `taxon_genus + taxon_species`.
#   is_verified     — this detection is human-verified.
#
# Empty files get one sentinel row (detection_category="blank") so survey
# visits / empty cameras stay visible for effort accounting.
_FLAT_OBS_HEADERS = [
    "detection_id",
    "file_id",
    "event_id",
    "relative_path",
    "datetime",
    "site_name",
    "latitude",
    "longitude",
    "detection_category",
    "detection_confidence",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "frame_number",
    "classification_label",
    "classification_confidence",
    "taxon_class",
    "taxon_order",
    "taxon_family",
    "taxon_genus",
    "taxon_species",
    "is_verified",
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


def get_scoped_detection_rows(
    db: Session,
    project: Project,
    *,
    extra_excluded: list[str] | None = None,
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
    """
    threshold_clause = or_(
        Detection.confidence >= project.detection_threshold,
        Detection.verified.is_(True),
    )

    # Outer join on Site so deployments without an assigned site still
    # appear in the export; the CSV serializer emits blank lat/lon cells
    # for those rows, and CamtrapDP / GeoJSON skip them with a separate
    # skipped_deployment_ids list.
    query = (
        select(File, Detection, Deployment, Site, LabelTaxonomy)
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

    excluded = [s.lower() for s in (project.excluded_classes or [])]
    if extra_excluded:
        excluded = list({*excluded, *(s.lower() for s in extra_excluded)})
    if excluded:
        # Drop animal detections whose taxonomy name OR raw label matches
        # the excluded list (case-insensitive). ``coalesce`` turns null
        # taxonomy/label into an empty string so ``in_`` behaves predictably
        # instead of returning SQL NULL.
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
    if detection.display_name:
        return detection.display_name
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
    Latin / scientific name. ``label_taxonomy.display_name`` is the single
    source of truth (see MEMORY.md project_taxonomy_display_name).
    """
    if detection.category != "animal":
        return ""
    if taxonomy and taxonomy.display_name:
        return taxonomy.display_name
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
    """Map each file id to its event in one query (avoids N+1)."""
    if not file_ids:
        return {}
    stmt = (
        select(event_files.c.file_id, Event)
        .join(event_files, event_files.c.event_id == Event.id)
        .where(event_files.c.file_id.in_(list(file_ids)))
    )
    return {fid: event for fid, event in db.execute(stmt).all()}


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
# Flat observations rows (one row per species per image)
# ---------------------------------------------------------------------------


def build_observation_rows(
    db: Session, project: Project, scoped_rows: Sequence[Row[Any]]
) -> tuple[list[str], list[list[Any]]]:
    """
    Build `(headers, rows)` for the flat Observations export.

    Grain: one row per detection. Files with no in-scope detections get
    a single sentinel ``category="blank"`` row so empty cameras stay
    visible for effort accounting.
    """
    tz_name = project.timezone

    grouped = list(_group_rows_by_file(scoped_rows))
    event_map = _events_by_file(db, [f.id for f, _d, _s, _dets in grouped])

    rows: list[list[Any]] = []
    for file_obj, deployment, site, detections in grouped:
        captured = _iso_datetime(file_obj.captured_at_local, tz_name)
        site_name = site.name if site is not None else ""
        latitude = site.latitude if site is not None else ""
        longitude = site.longitude if site is not None else ""
        relative_path = _relative_path(file_obj, deployment)

        # Event grouping. A file with no event falls back to its own id
        # so the column is never empty.
        event = event_map.get(file_obj.id)
        event_id = event.id if event else file_obj.id

        # Shared file/where/when block for every row on this file. The
        # row's own id (detection_id) leads; this is everything after it.
        where_when = [
            file_obj.id,
            event_id,
            relative_path,
            captured,
            site_name,
            latitude,
            longitude,
        ]

        if not detections:
            rows.append([""] + where_when + _blank_detection_cells(file_obj))
            continue

        for detection, taxonomy in detections:
            rows.append(
                [detection.id] + where_when + _detection_cells(detection, taxonomy)
            )

    return _FLAT_OBS_HEADERS, rows


def _round_or_blank(value: float | None, ndigits: int) -> float | str:
    return round(value, ndigits) if value is not None else ""


def _detection_cells(
    detection: Detection, taxonomy: LabelTaxonomy | None
) -> list[Any]:
    """The detector + classifier tail of a flat observations row.

    ``classification_label`` is the species label only — empty for
    person, vehicle, or an unclassified animal, so the detector and
    classifier stages stay cleanly separated.
    """
    return [
        detection.category,
        round(detection.confidence, 6),
        _round_or_blank(detection.bbox_x, 6),
        _round_or_blank(detection.bbox_y, 6),
        _round_or_blank(detection.bbox_width, 6),
        _round_or_blank(detection.bbox_height, 6),
        detection.frame_number if detection.frame_number is not None else "",
        detection.label or "",
        _round_or_blank(detection.label_confidence, 6),
        *_taxon_ranks(taxonomy),
        "TRUE" if detection.verified else "FALSE",
    ]


def _blank_detection_cells(file_obj: File) -> list[Any]:
    """Sentinel tail for an empty file (no in-scope detections)."""
    return [
        "blank",      # detection_category
        "",           # detection_confidence
        "", "", "", "",  # bbox
        "",           # frame_number
        "",           # classification
        "",           # classification_confidence
        "", "", "", "", "",  # taxon ranks
        "TRUE" if file_obj.verified else "FALSE",
    ]


# ---------------------------------------------------------------------------
# Spatial layers
# ---------------------------------------------------------------------------


def build_spatial_layers(
    db: Session, project: Project, scoped_rows: Sequence[Row[Any]]
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """Build the three spatial layers: deployments, observations, species_summary.

    Returns ``(layers, skipped_deployment_ids)``. The skipped list
    contains deployments that were excluded from the GeoJSON /
    Shapefile / GeoPackage because they have no site coordinates.
    """
    headers, flat_rows = build_observation_rows(db, project, scoped_rows)

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

    # Observations layer: one feature per flat row (blank rows included so
    # downstream consumers know "this camera had a survey visit").
    # Site coordinates come from the first scoped row touching that file.
    site_by_file: dict[str, Site] = {}
    for row in scoped_rows:
        file_obj, _detection, _deployment, site, _taxonomy = row
        site_by_file.setdefault(file_obj.id, site)

    observations_features: list[dict[str, Any]] = []
    for row in flat_rows:
        # Read by header name, not position, so column reorders here don't
        # silently shift the spatial properties.
        rec = dict(zip(headers, row, strict=True))
        site = site_by_file.get(rec["file_id"])
        if site is None:
            continue
        observations_features.append(
            {
                "lon": site.longitude,
                "lat": site.latitude,
                "properties": {
                    "file_id": rec["file_id"],
                    "relative_path": rec["relative_path"],
                    "datetime": rec["datetime"],
                    "site_name": rec["site_name"],
                    "event_id": rec["event_id"],
                    "detection_category": rec["detection_category"],
                    "detection_confidence": rec["detection_confidence"],
                    "classification_label": rec["classification_label"],
                    "classification_confidence": rec["classification_confidence"],
                    "is_verified": rec["is_verified"],
                },
            }
        )

    # Species summary: aggregate observations per (site, species). Only
    # rows with a species classification count; person, vehicle,
    # unclassified animal, and blank rows have an empty classification.
    summary_bucket: dict[tuple[str, str], dict[str, Any]] = {}
    trap_days_by_site: dict[str, int] = defaultdict(int)
    site_coords: dict[str, tuple[float, float]] = {}
    for deployment, site in all_deployments:
        if site is None:
            continue
        trap_days_by_site[site.id] += trap_days_by_dep.get(deployment.id, 0)
        site_coords[site.id] = (site.longitude, site.latitude)

    for feat in observations_features:
        props = feat["properties"]
        species = props["classification_label"]
        if not species:
            continue
        site = site_by_file.get(props["file_id"])
        if site is None:
            continue
        key = (site.id, species)
        bucket = summary_bucket.setdefault(
            key,
            {
                "site_name": site.name,
                "species": species,
                "total_count": 0,
            },
        )
        # One flat row per detection now, so each row is one count.
        bucket["total_count"] += 1

    species_summary_features: list[dict[str, Any]] = []
    for (site_id, _species), data in summary_bucket.items():
        trap_days = trap_days_by_site.get(site_id, 0)
        rate = (data["total_count"] / trap_days * 100) if trap_days > 0 else 0.0
        lon, lat = site_coords[site_id]
        species_summary_features.append(
            {
                "lon": lon,
                "lat": lat,
                "properties": {
                    "site_name": data["site_name"],
                    "species": data["species"],
                    "total_count": data["total_count"],
                    "detection_rate_per_100": round(rate, 2),
                },
            }
        )

    return (
        {
            "deployments": deployments_features,
            "observations": observations_features,
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
                "",                                                           # habitat
                "",                                                           # deploymentGroups
                "",                                                           # deploymentTags
                deployment.notes or "",                                       # deploymentComments
            ]
        )

    # media.csv + observations.csv: iterate scoped rows grouped by file.
    # Files whose deployment has no site were already skipped at the
    # deployments.csv stage (CamtrapDP requires lat/lon). Skip their
    # media and observations rows here so the package stays consistent.
    skipped_dep_set = set(camtrap_skipped_deployment_ids)
    for file_obj, deployment, _site, detections in _group_rows_by_file(scoped_rows):
        if deployment.id in skipped_dep_set:
            continue
        if file_obj.captured_at_local is not None:
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

        event = _file_event(db, file_obj.id)
        event_id = event.id if event else file_obj.id
        event_start = _iso_datetime(
            event.event_start_local if event else file_obj.captured_at_local, tz_name
        )
        event_end = _iso_datetime(
            event.event_end_local if event else file_obj.captured_at_local, tz_name
        )
        captured_iso = _iso_datetime(file_obj.captured_at_local, tz_name)

        if not detections or file_obj.observation_type == "blank":
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

            # Row order must match _CAMTRAP_OBS_HEADERS exactly.
            # Event-level observations (no bbox) emit at
            # observationLevel="event" with empty bbox columns; this is
            # the canonical Camtrap-DP shape for "species was seen in
            # this clip without a frame-anchored ROI."
            has_bbox = detection.bbox_x is not None
            observations_rows.append(
                [
                    f"{obs_id_prefix}-{detection.id}",    # observationID
                    deployment.id,                         # deploymentID
                    file_obj.id,                           # mediaID
                    event_id,                              # eventID
                    event_start or captured_iso,           # eventStart
                    event_end or captured_iso,             # eventEnd
                    "media" if has_bbox else "event",      # observationLevel
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
                    round(detection.bbox_x, 6) if has_bbox else "",     # bboxX
                    round(detection.bbox_y, 6) if has_bbox else "",     # bboxY
                    round(detection.bbox_width, 6) if has_bbox else "", # bboxWidth
                    round(detection.bbox_height, 6) if has_bbox else "",# bboxHeight
                    method,                                # classificationMethod
                    classified_by,                         # classifiedBy
                    "",                                    # classificationTimestamp
                    prob,                                  # classificationProbability
                    "",                                    # observationTags
                    comments,                              # observationComments
                ]
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


def _obs_type_from_category(category: str) -> str:
    if category == "animal":
        return "animal"
    if category == "person":
        return "human"
    if category == "vehicle":
        return "vehicle"
    return "unknown"


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
        if taxonomy and taxonomy.display_name:
            entry["scientificName"] = taxonomy.display_name
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
