"""Reconstruct the canonical AddaxAI / Timelapse recognition JSON from the DB.

The same JSON shape that `app.ml.json_pipeline.merge_json_files`
produces is the de facto AddaxAI recognition format. Users have
downstream scripts that parse it; the Timelapse Analyser imports it
directly.

For a folder run, the pipeline writes the full rich JSON during
analysis but then merges it into the DB and discards the working
copy. To produce a recognition file for the user we serialise the
current DB state back into the same shape. This is slightly lossy
on the classifications list — the DB only stores the top-1
classification per detection — but the shape is identical:
`detection_categories`, `classification_categories`,
`classification_category_descriptions` (the 7-token taxonomy strings,
rebuilt from `label_taxonomy` exactly as results mode emits them),
the `info` block (`format_version` 1.6 + the `addaxai` sub-block),
per-image `exif_metadata` (DateTimeOriginal and GPSInfo, as
MegaDetector writes them) and `width`/`height`, per-video
`frame_rate` and `frames_processed` (required for videos by MD format
1.6), and per-image detections with `category`, `conf`, `bbox`, and
optional `classifications` and `frame_number` keys.

The `info.addaxai` block additionally carries the app version and a
`settings` sub-dict (smoothing, rollup, geofence, independence
interval, video fps) so the run is reproducible from the JSON alone.
The file is the complete record of the run: every stored detection is
included, nothing is threshold-filtered.

File paths in the output are relative to the source folder (the
deployment's `folder_path`). The save step defaults the output dir to
the source folder itself, so the file lands where those paths
resolve — required by the Timelapse Analyser, which matches the
relative paths against the folder the JSON sits in.

Filename is `addaxai-recognitions.json`: one canonical name so
scripts (and the Timelapse Analyser) can rely on it, with the shared
`addaxai-` prefix so the run's outputs sort together between the
user's own files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import __version__ as APP_VERSION
from app.core.logging_config import get_logger
from app.models import Deployment, Detection, File, LabelTaxonomy, Project

logger = get_logger(__name__)


# Detection category id mapping, identical to MegaDetector's convention
# and to how `json_pipeline._load_to_database` (and the rest of the
# code) interprets the field.
_CATEGORY_TO_ID = {"animal": "1", "person": "2", "vehicle": "3"}
_DETECTION_CATEGORIES = {v: k for k, v in _CATEGORY_TO_ID.items()}

# Output filename. Stays the same across runs so downstream tools that
# look for one canonical filename keep working.
RECOGNITION_JSON_FILENAME = "addaxai-recognitions.json"


@dataclass
class RecognitionJsonResult:
    """Summary of a recognition-JSON write."""

    output_path: str = ""
    image_count: int = 0
    detection_count: int = 0
    classification_count: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "output_path": self.output_path,
            "image_count": self.image_count,
            "detection_count": self.detection_count,
            "classification_count": self.classification_count,
            "errors": list(self.errors),
        }


def _relative_path(file_path: str, base: str | None) -> str:
    """Return file_path made relative to the deployment base when possible.

    If the file is not under the base (rare; shouldn't happen for a
    folder run) we fall back to the absolute path so the consumer
    still has a valid identifier.
    """
    if not base:
        return file_path
    try:
        return str(Path(file_path).resolve().relative_to(Path(base).resolve()))
    except ValueError:
        return file_path


def _failure_entries(db: Session, project_id: str) -> list[dict]:
    """MegaDetector-format entries for files that could not be read.

    Sourced from ``Deployment.warnings``, the durable record the analysis
    worker writes (the queue entry carrying the same list is ephemeral).
    Only decode failures are emitted: a file with no capture date was read
    perfectly well and has a real File row already in ``images``.
    """
    entries: list[dict] = []
    deployments = db.execute(
        select(Deployment).where(Deployment.project_id == project_id)
    ).scalars().all()
    for deployment in deployments:
        for warning in deployment.warnings or []:
            if warning.get("type") != "video_processing_failure":
                continue
            path = warning.get("path")
            if not path:
                continue
            entries.append(
                {
                    "file": _relative_path(path, deployment.folder_path),
                    "failure": warning.get("reason") or "Could not be read",
                }
            )
    return entries


def _bbox_for_detection(det: Detection) -> list[float] | None:
    """Return the bbox in the canonical [x, y, w, h] order, or None
    for event-level observations where every bbox coordinate is null.

    All four fields are constrained to be set-or-null together (see
    DEVELOPERS.md and the bbox-nullable migration); we still defend
    against partial nulls so a corrupt row doesn't break the export.
    """
    if (
        det.bbox_x is None
        or det.bbox_y is None
        or det.bbox_width is None
        or det.bbox_height is None
    ):
        return None
    return [
        float(det.bbox_x),
        float(det.bbox_y),
        float(det.bbox_width),
        float(det.bbox_height),
    ]


def _taxonomy_description(label: str, tax: LabelTaxonomy | None) -> str | None:
    """Build the 7-token classification description for a label.

    Matches results mode exactly (see
    ``json_utils.build_classification_category_descriptions``):
    ``name;class;order;family;genus;species;name`` (all lowercase, with an
    empty token for any missing rank). MegaDetector's smoothing reads token
    0 as an identifier, tokens 1-5 as the taxonomy, and token 6 as the
    display name.

    Returns ``None`` when there's no taxonomy to describe (a custom or
    unknown label with no ranks), so that category id is omitted entirely
    -- exactly as results mode omits labels absent from the model taxonomy.
    """
    if tax is None:
        return None
    ranks = [
        tax.taxon_class,
        tax.taxon_order,
        tax.taxon_family,
        tax.taxon_genus,
        tax.taxon_species,
    ]
    if not any(r for r in ranks):
        return None
    name = label.strip().lower()
    tokens = [name, *[(r or "").strip().lower() for r in ranks], name]
    return ";".join(tokens)


def write_recognition_json(
    db: Session,
    project_id: str,
    target_dir: Path,
    *,
    excluded_species: list[str] | None = None,
) -> RecognitionJsonResult:
    """Serialise the project's analysis results to the canonical JSON shape.

    The output is written to `target_dir/addaxai-recognitions.json`.
    The directory is created if it does not exist. An existing file
    at that path is overwritten — the recognition file represents the
    current DB state, so a re-export replaces the previous snapshot.

    ``excluded_species`` filters classification entries out of the
    per-detection ``classifications`` list. The detection itself
    stays (it's still a real detection) but its species attribution
    is suppressed.
    """
    project = db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id!r} not found")

    excluded = (
        frozenset(excluded_species)
        if excluded_species
        else frozenset()
    )

    target_dir.mkdir(parents=True, exist_ok=True)

    # Use the first deployment's folder_path as the base for relative
    # file paths. A folder run has exactly one deployment by
    # construction; research projects may have several, in which case
    # the JSON will end up with a mix of relative + absolute paths,
    # which is acceptable for an interoperability export.
    deployment_row = db.execute(
        select(Deployment).where(Deployment.project_id == project_id).limit(1)
    ).scalar_one_or_none()
    base_folder = (
        deployment_row.folder_path if deployment_row is not None else None
    )

    files = db.execute(
        select(File)
        .join(Deployment, File.deployment_id == Deployment.id)
        .where(Deployment.project_id == project_id)
        .order_by(File.captured_at_local.asc())
    ).scalars().all()

    # Build a stable label -> id map as we walk detections, mirroring
    # the unified mapping merge_json_files produces. Alongside it, remember
    # each label's taxonomy row so we can rebuild the canonical
    # classification_category_descriptions (the 7-token taxonomy strings).
    label_to_id: dict[str, str] = {}
    next_label_id = 1
    taxonomy_id_by_label: dict[str, str] = {}

    images_out: list[dict] = []
    detection_total = 0
    classification_total = 0

    for file in files:
        # Frame order first so a video's detections read sequentially
        # (frame_number is NULL for images, so they all tie and fall back
        # to the confidence order MegaDetector's image format expects).
        detections = db.execute(
            select(Detection)
            .where(Detection.file_id == file.id)
            .order_by(
                Detection.frame_number.asc(),
                Detection.confidence.desc(),
            )
        ).scalars().all()

        det_objs: list[dict] = []
        for det in detections:
            category_id = _CATEGORY_TO_ID.get(det.category)
            if category_id is None:
                # Unknown category — log and skip rather than emit a
                # row downstream tools cannot interpret.
                logger.warning(
                    f"recognition_json: dropping detection with unknown "
                    f"category {det.category!r} on file {file.id}"
                )
                continue

            bbox = _bbox_for_detection(det)
            if bbox is None:
                # Event-level observation — no spatial annotation to
                # write. The canonical Timelapse / AddaxAI JSON has
                # no representation for this, so we skip.
                continue

            det_entry: dict = {
                "category": category_id,
                "conf": round(float(det.confidence), 4),
                "bbox": [round(v, 6) for v in bbox],
                # Whether a human has verified this specific detection.
                "verified": bool(det.verified),
            }

            label_excluded = bool(excluded) and (
                (det.label_taxonomy_id and det.label_taxonomy_id in excluded)
                or (det.label and det.label in excluded)
            )
            if (
                det.label
                and det.label_confidence is not None
                and not label_excluded
            ):
                if det.label not in label_to_id:
                    label_to_id[det.label] = str(next_label_id)
                    next_label_id += 1
                if det.label_taxonomy_id and det.label not in taxonomy_id_by_label:
                    taxonomy_id_by_label[det.label] = det.label_taxonomy_id
                det_entry["classifications"] = [
                    [
                        label_to_id[det.label],
                        round(float(det.label_confidence), 4),
                    ]
                ]
                classification_total += 1

            if det.frame_number is not None:
                det_entry["frame_number"] = int(det.frame_number)

            det_objs.append(det_entry)
            detection_total += 1

        # Restore the per-image metadata MegaDetector writes (it runs with
        # --include_exif_tags datetimeoriginal,gpsinfo) and merge_json_files
        # passes through verbatim: image dimensions and the EXIF block
        # (DateTimeOriginal, GPSInfo). Stored on the File row at ingestion.
        image_entry: dict = {
            "file": _relative_path(file.file_path, base_folder),
        }
        # Video fields, required by the MD output format 1.6 (Timelapse
        # needs frame_rate to resolve frame numbers to timestamps).
        # frames_processed is NULL for videos ingested before the column
        # existed; omitted then, restored on re-analysis.
        if file.frame_rate is not None:
            image_entry["frame_rate"] = float(file.frame_rate)
        if file.frames_processed is not None:
            image_entry["frames_processed"] = [
                int(n) for n in file.frames_processed
            ]
        image_entry["detections"] = det_objs
        if file.width_px is not None:
            image_entry["width"] = file.width_px
        if file.height_px is not None:
            image_entry["height"] = file.height_px
        if file.exif_data:
            image_entry["exif_metadata"] = file.exif_data
        images_out.append(image_entry)

    # Files the detector could not read have no File row, so the loop above
    # cannot emit them and the JSON would quietly describe a smaller folder
    # than the one analysed. MegaDetector's own format carries them as
    # `{"file": ..., "failure": ...}`, which is what the internal
    # results.json holds; re-emit them so this file is a complete account of
    # the folder and a downstream tool can tell "nothing found" apart from
    # "never looked at".
    images_out.extend(_failure_entries(db, project_id))

    classification_categories = {
        cid: name for name, cid in label_to_id.items()
    }

    # Rebuild classification_category_descriptions from label_taxonomy, the
    # same 7-token taxonomy strings results mode emits. Custom / unknown
    # labels with no ranks get no entry, exactly as results mode omits
    # labels missing from the model taxonomy.
    tax_ids = set(taxonomy_id_by_label.values())
    tax_by_id: dict[str, LabelTaxonomy] = {}
    if tax_ids:
        tax_by_id = {
            t.id: t
            for t in db.execute(
                select(LabelTaxonomy).where(LabelTaxonomy.id.in_(tax_ids))
            ).scalars().all()
        }
    classification_category_descriptions: dict[str, str] = {}
    for label, cid in label_to_id.items():
        tax = tax_by_id.get(taxonomy_id_by_label.get(label, ""))
        desc = _taxonomy_description(label, tax)
        if desc is not None:
            classification_category_descriptions[cid] = desc

    output_payload: dict = {
        "images": images_out,
        "detection_categories": dict(_DETECTION_CATEGORIES),
        "classification_categories": classification_categories,
    }
    # Mirror results mode: only present when there's taxonomy to describe.
    if classification_category_descriptions:
        output_payload["classification_category_descriptions"] = (
            classification_category_descriptions
        )
    output_payload["info"] = {
        # MegaDetector output format version this file conforms to.
        # 1.6 requires frame_rate + frames_processed on video entries
        # and frame_number on their detections, all emitted above.
        "format_version": "1.6",
        "addaxai": {
            "version": APP_VERSION,
            "export_source": "folder-run",
            "classification_completion_time": (
                datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
            ),
            "detection_model": project.detection_model_id,
            **(
                {"classification_model": project.classification_model_id}
                if project.classification_model_id
                else {}
            ),
            **(
                {"embedding_model": project.embedding_model_id}
                if project.embedding_model_id
                else {}
            ),
            # The result-affecting settings that produced this export, so a
            # run is reproducible from the JSON alone. These are the
            # project's current settings, which for a folder run are the
            # settings that produced the outputs. No detection threshold:
            # this file is the complete record down to the inference
            # floor, nothing was filtered out.
            "settings": {
                # Detection confidence above which crops were classified
                # and embedded. MD itself ran at its output cap (0.01).
                "classification_gate": project.classification_gate,
                "country_code": project.country_code,
                "state_code": project.state_code,
                "event_smoothing": project.event_smoothing,
                "smoothing_strength": project.smoothing_strength,
                "taxonomic_rollup": project.taxonomic_rollup,
                "independence_interval_seconds": project.independence_interval,
                "video_fps": project.video_fps,
            },
        }
    }

    output_path = target_dir / RECOGNITION_JSON_FILENAME
    with open(output_path, "w") as f:
        json.dump(output_payload, f, indent=2)

    logger.info(
        f"recognition_json: project={project_id} "
        f"images={len(images_out)} "
        f"detections={detection_total} "
        f"classifications={classification_total} "
        f"path={output_path}"
    )

    return RecognitionJsonResult(
        output_path=str(output_path),
        image_count=len(images_out),
        detection_count=detection_total,
        classification_count=classification_total,
    )
