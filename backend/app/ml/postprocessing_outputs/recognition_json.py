"""Reconstruct the canonical AddaxAI / Timelapse recognition JSON from the DB.

The same JSON shape that `app.ml.json_pipeline.merge_json_files`
produces (and that the Timelapse integration writes to
`<folder>/timelapse_recognition_file.json`) is the de facto AddaxAI
recognition format. Users have downstream scripts that parse it; the
Timelapse Analyser imports it directly.

For a folder run, the pipeline writes the full rich JSON during
analysis but then merges it into the DB and discards the working
copy. To produce a recognition file for the user we serialise the
current DB state back into the same shape. This is slightly lossy
on the classifications list — the DB only stores the top-1
classification per detection — but the shape is identical:
`detection_categories`, `classification_categories`, the
`info.addaxai` block, per-image detections with `category`, `conf`,
`bbox`, and optional `classifications` and `frame_number` keys.

File paths in the output are relative to the source folder (the
deployment's `folder_path`). This matches how the Timelapse runner
writes them and keeps the file portable: copy or share the source
folder along with the JSON and downstream tools still work.

Filename is `timelapse_recognition_file.json`, identical to the
Timelapse mode output. The user explicitly asked for one canonical
name across all modes so scripts that look for one filename keep
working in the folder-run flow.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.models import Deployment, Detection, File, Project

logger = get_logger(__name__)


# Detection category id mapping, identical to MegaDetector's convention
# and to how `json_pipeline._load_to_database` (and the rest of the
# code) interprets the field.
_CATEGORY_TO_ID = {"animal": "1", "person": "2", "vehicle": "3"}
_DETECTION_CATEGORIES = {v: k for k, v in _CATEGORY_TO_ID.items()}

# Output filename. Stays the same across folder runs and Timelapse so
# downstream tools that look for one canonical filename keep working.
RECOGNITION_JSON_FILENAME = "timelapse_recognition_file.json"


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


def write_recognition_json(
    db: Session,
    project_id: str,
    target_dir: Path,
    *,
    excluded_species: list[str] | None = None,
) -> RecognitionJsonResult:
    """Serialise the project's analysis results to the canonical JSON shape.

    The output is written to `target_dir/timelapse_recognition_file.json`.
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
    deployment_id = deployment_row.id if deployment_row is not None else ""

    files = db.execute(
        select(File)
        .join(Deployment, File.deployment_id == Deployment.id)
        .where(Deployment.project_id == project_id)
        .order_by(File.captured_at_local.asc())
    ).scalars().all()

    # Build a stable label -> id map as we walk detections, mirroring
    # the unified mapping merge_json_files produces.
    label_to_id: dict[str, str] = {}
    next_label_id = 1

    images_out: list[dict] = []
    detection_total = 0
    classification_total = 0

    for file in files:
        detections = db.execute(
            select(Detection)
            .where(Detection.file_id == file.id)
            .order_by(Detection.confidence.desc())
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

        images_out.append(
            {
                "file": _relative_path(file.file_path, base_folder),
                "detections": det_objs,
            }
        )

    classification_categories = {
        cid: name for name, cid in label_to_id.items()
    }

    output_payload: dict = {
        "images": images_out,
        "detection_categories": dict(_DETECTION_CATEGORIES),
        "classification_categories": classification_categories,
        "info": {
            "addaxai": {
                "version": "folder-run-export",
                "deployment_id": deployment_id,
                "classification_completion_time": (
                    datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
                ),
                "detection_model": project.detection_model_id,
                **(
                    {"classification_model": project.classification_model_id}
                    if project.classification_model_id
                    else {}
                ),
            }
        },
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
