"""
Classification postprocessing service.

Applies event smoothing and taxonomic rollup to classification results
using MegaDetector's classification_postprocessing module. Reads raw
predictions from JSON files and writes smoothed results back to the DB.

The actual smoothing runs as a subprocess in the ML environment because
megadetector is only installed there, not in the backend environment.

Created by Claude Code on 2026-02-14
"""

import hashlib
import json
import subprocess
import tempfile
import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.job_cancellation import (
    JobCancelledError,
    is_cancel_requested,
    track_subprocess,
)
from app.core.logging_config import get_logger
from app.core.subprocess_group import popen_group
from app.ml.observation_type import derive_observation_type
from app.models import Deployment, Detection, File, Project
from app.utils.subprocess_env import clean_python_env

logger = get_logger(__name__)


def compute_postprocessing_settings_hash(project) -> str:
    """
    Compute SHA-256 hash of the project's postprocessing-relevant settings.

    Used to detect when settings have changed and reprocessing is needed.

    Args:
        project: Project ORM object

    Returns:
        64-character hex SHA-256 hash string
    """
    canonical = json.dumps(
        {
            "event_smoothing": project.event_smoothing,
            "smoothing_strength": project.smoothing_strength,
            "taxonomic_rollup": project.taxonomic_rollup,
            "independence_interval": project.independence_interval,
            "excluded_classes": sorted(project.excluded_classes or []),
            "country_code": project.country_code,
            "state_code": project.state_code,
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def build_smoother_input(
    deployment_id: str,
    independence_interval: int,
    db: Session,
) -> list[dict]:
    """
    Package a deployment's events for MegaDetector's sequence-level smoother.

    Clustering uses `app.services.event_clustering.cluster_files_into_events`
    — the same primitive that produces `Event` rows in the UI — so the
    smoother operates on identical groupings to what the user sees.

    The returned format is the COCO Camera Traps "images" shape that
    `smooth_classification_results_sequence_level` expects. Each dict
    carries `seq_id` (MegaDetector's contract for event membership),
    `file_name` (relative to the deployment folder), and `datetime`.

    Args:
        deployment_id: Deployment UUID
        independence_interval: Gap in seconds that starts a new event
        db: Database session

    Returns:
        List of dicts with keys: file_name, seq_id, datetime. Empty list
        when the deployment has no files to cluster.
    """
    from sqlalchemy.orm import joinedload

    from app.services.event_clustering import cluster_files_into_events

    # Use image + video (not frame) because the smoothing script matches
    # file_name against the raw JSON which has video-level entries, not
    # frame-level JPEG paths.
    files = (
        db.query(File)
        .options(joinedload(File.source_video))
        .filter(File.deployment_id == deployment_id)
        .filter(File.file_type.in_(["image", "video"]))
        .all()
    )
    if not files:
        return []

    deployment = (
        db.query(Deployment).filter(Deployment.id == deployment_id).first()
    )
    deployment_folder = deployment.folder_path if deployment else None

    smoother_input: list[dict] = []
    for cluster in cluster_files_into_events(files, independence_interval):
        seq_id = str(uuid.uuid4())
        for file_record in cluster:
            # Date-less files form single-file events and can't be
            # temporally smoothed; exclude them so the smoother never
            # sees a null datetime (they keep their raw classification).
            if file_record.captured_at_local is None:
                continue
            if deployment_folder:
                try:
                    rel_path = str(
                        Path(file_record.file_path).relative_to(deployment_folder)
                    )
                except ValueError:
                    rel_path = file_record.file_path
            else:
                rel_path = file_record.file_path
            smoother_input.append(
                {
                    "file_name": rel_path,
                    "seq_id": seq_id,
                    "datetime": file_record.captured_at_local.isoformat(),
                }
            )
    return smoother_input


def _find_classification_model_dir(project, db: Session) -> Path | None:
    """
    Find the classification model directory for taxonomy.csv lookup.

    Args:
        project: Project ORM object
        db: Database session

    Returns:
        Path to classification model directory, or None
    """
    if not project.classification_model_id:
        return None

    try:
        from app.ml.manifest_manager import ManifestManager
        from app.ml.model_storage import ModelStorage

        manifest_manager = ManifestManager()
        model_storage = ModelStorage()
        cls_manifest = manifest_manager.get_model(project.classification_model_id)
        return model_storage.get_model_path(cls_manifest)
    except Exception as e:
        logger.warning(f"Could not find classification model dir: {e}")
        return None


def _get_ml_python_path() -> Path:
    """
    Get the Python executable path for the ML environment.

    Returns:
        Path to the ML environment's Python executable
    """
    from app.ml.environment_manager import EnvironmentManager

    env_manager = EnvironmentManager()
    return env_manager.get_python("env-addaxai-base")


def _ensure_7_token_descriptions(md_results: dict, project, db: Session) -> bool:
    """
    Ensure classification_category_descriptions are in 7-token format.

    Modifies md_results in-place if rebuild is needed.

    Returns:
        True if descriptions were rebuilt (JSON file should be updated)
    """
    descriptions = md_results.get("classification_category_descriptions", {})
    needs_rebuild = not descriptions
    if not needs_rebuild and descriptions:
        sample = next(iter(descriptions.values()), "")
        if len(sample.split(";")) != 7:
            needs_rebuild = True

    if not needs_rebuild:
        return False

    cls_model_dir = _find_classification_model_dir(project, db)
    if cls_model_dir:
        taxonomy_csv = cls_model_dir / "taxonomy.csv"
        if taxonomy_csv.exists():
            from app.ml.json_utils import build_classification_category_descriptions

            class_cats = md_results.get("classification_categories", {})
            new_descriptions = build_classification_category_descriptions(
                class_cats, taxonomy_csv
            )
            if new_descriptions:
                md_results["classification_category_descriptions"] = new_descriptions
                return True

    return False


def run_postprocessing_for_deployment(
    deployment_id: str,
    json_path: Path,
    deployment_folder: Path,
    project,
    db: Session,
    job_id: str | None = None,
) -> dict:
    """
    Run classification postprocessing (smoothing + taxonomic rollup) on a deployment.

    Loads the raw JSON, applies MegaDetector's smoothing via a subprocess
    (since megadetector is only installed in the ML environment), and returns
    the smoothed results dict.

    Args:
        deployment_id: Deployment UUID
        json_path: Path to results.json
        deployment_folder: Path to deployment folder
        project: Project ORM object
        db: Database session

    Returns:
        Smoothed MegaDetector-format results dict
    """
    # Load raw JSON
    with open(json_path) as f:
        md_results = json.load(f)

    # Ensure 7-token descriptions exist; update JSON file if rebuilt
    if _ensure_7_token_descriptions(md_results, project, db):
        with open(json_path, "w") as f:
            json.dump(md_results, f, indent=2)
        logger.info("Rebuilt classification_category_descriptions to 7-token format")

    # Apply label exclusion in memory (JSON on disk stays as ground truth).
    # When taxonomy is available, this is a no-op: excluded species are
    # handled by the geofence-aware rollup below instead.
    from app.ml.label_exclusion import apply_label_exclusion_to_results

    cls_model_dir = _find_classification_model_dir(project, db)
    exclusion_taxonomy = None
    if cls_model_dir:
        _tax = cls_model_dir / "taxonomy.csv"
        if _tax.exists():
            from app.ml.taxonomic_rollup import load_taxonomy_lookup

            exclusion_taxonomy = load_taxonomy_lookup(_tax)

    apply_label_exclusion_to_results(
        md_results, project.excluded_classes, exclusion_taxonomy
    )

    # Build excluded_names and allowed_taxonomy_keys for geofence-aware
    # rollup (matching official SpeciesNet API behavior).
    excluded_names: frozenset[str] | None = None
    if project.excluded_classes:
        excluded_names = frozenset(
            name.lower() for name in project.excluded_classes
        )

    allowed_taxonomy_keys: frozenset[str] | None = None
    if cls_model_dir and project.country_code:
        try:
            from app.ml.geofence import get_allowed_taxonomy_keys

            allowed_taxonomy_keys = get_allowed_taxonomy_keys(
                cls_model_dir, project.country_code, project.state_code
            )
        except FileNotFoundError:
            pass

    # --- Taxonomic rollup (geofence-aware) ---
    # Two paths: (A) excluded top-1 rolls up to allowed ancestor,
    # (B) low-confidence allowed top-1 rolls up at 0.65 threshold.
    if project.taxonomic_rollup:
        if not cls_model_dir:
            cls_model_dir = _find_classification_model_dir(project, db)
        if cls_model_dir:
            taxonomy_csv = cls_model_dir / "taxonomy.csv"
            if taxonomy_csv.exists():
                from app.ml.taxonomic_rollup import (
                    apply_taxonomic_rollup_to_results,
                    load_taxonomy_lookup,
                )

                rollup_result = apply_taxonomic_rollup_to_results(
                    md_results,
                    taxonomy_csv,
                    excluded_names=excluded_names,
                    allowed_taxonomy_keys=allowed_taxonomy_keys,
                )
                md_results = rollup_result.md_results

                # Persist rolled-up entries to label_taxonomy table
                if rollup_result.new_entries and project.classification_model_id:
                    try:
                        from app.ml.taxonomy_db import add_rollup_taxonomy_entry

                        taxonomy_lookup = load_taxonomy_lookup(taxonomy_csv)
                        for entry in rollup_result.new_entries:
                            add_rollup_taxonomy_entry(
                                project.classification_model_id,
                                entry["name"],
                                entry["level"],
                                taxonomy_lookup,
                                db,
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to persist rollup taxonomy entries: {e}"
                        )

    # --- Event smoothing (independent of rollup) ---
    # Rollup section above is complete. If smoothing is off, return the
    # (possibly rolled-up) results as-is — no subprocess needed.
    if not project.event_smoothing:
        return md_results

    # Package events for the smoother. Uses the same clustering primitive
    # as the UI's Event rows, so the smoother sees the same boundaries.
    smoother_input = build_smoother_input(
        deployment_id, project.independence_interval, db
    )
    if smoother_input:
        logger.info(
            f"Running event-level smoothing on {len(smoother_input)} file "
            f"rows (independence_interval={project.independence_interval}s)"
        )

    # Build options for the subprocess script. `smoother_input` is the
    # CCT-format dict list; `smoothing_script.py` passes it through to
    # MegaDetector's `cct_sequence_information=` parameter unchanged.
    smoothing_options = {
        "event_smoothing": project.event_smoothing,
        "smoothing_strength": project.smoothing_strength,
        "detection_threshold": project.detection_threshold,
        "smoother_input": smoother_input,
    }

    # Run smoothing as subprocess in the ML environment
    python_path = _get_ml_python_path()
    script_path = Path(__file__).parent / "smoothing_script.py"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as opts_f:
        json.dump(smoothing_options, opts_f)
        opts_path = opts_f.name

    # Write rollup-modified JSON to a temp file (don't pass the on-disk raw file)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as input_f:
        json.dump(md_results, input_f)
        input_path = input_f.name

    output_path = tempfile.mktemp(suffix=".json")

    try:
        logger.info(f"Running smoothing subprocess for deployment {deployment_id}")
        # Popen + wait (instead of subprocess.run) so cancel can kill
        # the process group mid-smoothing.
        process = popen_group(
            [str(python_path), str(script_path), input_path, opts_path, output_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=clean_python_env(),
        )
        with track_subprocess(job_id, process):
            try:
                stdout, stderr = process.communicate(timeout=300)
            except subprocess.TimeoutExpired as e:
                process.kill()
                stdout, stderr = process.communicate()
                raise RuntimeError("Smoothing script timed out after 300s") from e

        if job_id and is_cancel_requested(job_id):
            raise JobCancelledError()

        if process.returncode != 0:
            error_detail = stderr or stdout or "(no output)"
            raise RuntimeError(f"Smoothing script failed: {error_detail}")

        with open(output_path) as f:
            smoothed = json.load(f)

        return smoothed

    except JobCancelledError:
        raise
    except Exception as e:
        logger.error(
            f"Event smoothing failed for deployment {deployment_id}, "
            f"returning rollup-only results: {e}",
            exc_info=True,
        )
        return md_results

    finally:
        Path(input_path).unlink(missing_ok=True)
        Path(opts_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


def update_database_from_smoothed_results(
    deployment_id: str,
    smoothed_results: dict,
    deployment_folder: Path,
    db: Session,
    taxonomy_lookup: dict[str, dict[str, str]] | None = None,
    excluded_classes: list[str] | None = None,
    excluded_taxonomy_ids: set[str] | None = None,
    taxonomy_name_to_id: (
        dict[str, tuple[str, str | None]] | None
    ) = None,
) -> dict:
    """
    Update Detection records in the database from smoothed JSON results.

    Matches JSON detections to DB records using file_path + bbox coordinates
    (rounded to 4 decimal places) + frame_number.

    After updating labels, sweeps for any non-verified detection whose
    label_taxonomy_id is still in excluded_taxonomy_ids and clears it.

    Args:
        deployment_id: Deployment UUID
        smoothed_results: Smoothed MegaDetector-format dict
        deployment_folder: Path to deployment folder
        db: Database session
        taxonomy_lookup: Optional taxonomy for scientific_name lookup
        excluded_classes: Optional list of excluded label names
            (legacy, used as fallback when excluded_taxonomy_ids
            is not provided).
        excluded_taxonomy_ids: Set of excluded taxonomy UUIDs.
            Preferred over excluded_classes for the final sweep.
        taxonomy_name_to_id: Pre-resolved {lowercase_name:
            (taxonomy_id, scientific_name)} mapping for setting
            label_taxonomy_id on updated detections.

    Returns:
        Dict with counts: {updated, unchanged, errors}
    """
    class_names = smoothed_results.get("classification_categories", {})

    # Build lookup: (file_path, bbox_key, frame_number) -> Detection record.
    # Post-2026-05 every detection points directly at its parent file row
    # (video or image), so `det.file.file_path` matches the JSON's
    # `img["file"]` directly. Legacy frame rows are removed by the
    # one-shot migration that runs on startup, so we never see them
    # here.
    # Event-level observations have no bbox and are never produced by
    # the AI pipeline, so the smoothing/rollup matcher (which keys on
    # bbox geometry) has nothing to match them against. Excluding them
    # here keeps the lookup keys total and avoids ever overwriting a
    # deliberate user observation.
    detections = (
        db.query(Detection)
        .join(File)
        .filter(File.deployment_id == deployment_id)
        .filter(Detection.bbox_x.isnot(None))
        .all()
    )

    detection_lookup: dict[tuple, Detection] = {}
    for det in detections:
        bbox_key = (
            round(det.bbox_x, 4),
            round(det.bbox_y, 4),
            round(det.bbox_width, 4),
            round(det.bbox_height, 4),
        )
        key = (det.file.file_path, bbox_key, det.frame_number)
        detection_lookup[key] = det

    updated = 0
    unchanged = 0
    errors = 0
    skipped_verified = 0

    # Track which files had label changes (for observation_type recomputation)
    changed_file_ids: set[str] = set()

    # `images or []` and the `failure` skip below tolerate failed-video
    # entries from process_video (their `detections` field is None).
    for img in smoothed_results.get("images") or []:
        if img.get("failure"):
            continue
        relative_file = img["file"]
        absolute_path = str((deployment_folder / relative_file).resolve())

        for det in img.get("detections") or []:
            try:
                bbox = det["bbox"]
                bbox_key = (
                    round(float(bbox[0]), 4),
                    round(float(bbox[1]), 4),
                    round(float(bbox[2]), 4),
                    round(float(bbox[3]), 4),
                )
                frame_number = det.get("frame_number")
                key = (absolute_path, bbox_key, frame_number)

                db_det = detection_lookup.get(key)
                if db_det is None:
                    errors += 1
                    continue

                # Preserve human-verified detections
                if db_det.verified:
                    skipped_verified += 1
                    unchanged += 1
                    continue

                # Get smoothed top-1 classification
                classifications = det.get("classifications", [])
                if classifications:
                    top_class_id, top_conf = classifications[0]
                    new_label = class_names.get(str(top_class_id))
                    new_confidence = float(top_conf)
                else:
                    new_label = None
                    new_confidence = None

                # Resolve taxonomy ID + both display names
                new_taxonomy_id = None
                new_scientific = None
                new_common = None
                if new_label and taxonomy_name_to_id:
                    resolved = taxonomy_name_to_id.get(
                        new_label.lower()
                    )
                    if resolved:
                        new_taxonomy_id = resolved[0]
                        new_scientific = resolved[1]
                        new_common = resolved[2]
                if new_label and not new_scientific:
                    from app.ml.taxonomic_rollup import format_common_name
                    from app.models.label_taxonomy import LabelTaxonomy

                    tax_row = (
                        db.query(
                            LabelTaxonomy.id,
                            LabelTaxonomy.scientific_name,
                            LabelTaxonomy.common_name,
                        )
                        .filter(LabelTaxonomy.name == new_label)
                        .first()
                    )
                    if tax_row:
                        new_taxonomy_id = tax_row[0]
                        new_scientific = tax_row[1]
                        new_common = tax_row[2]
                    else:
                        new_scientific = (
                            new_label[0].upper() + new_label[1:]
                        )
                        new_common = format_common_name(new_label)

                if db_det.label != new_label or db_det.label_confidence != new_confidence:
                    db_det.label = new_label
                    db_det.label_confidence = new_confidence
                    db_det.scientific_name = new_scientific
                    db_det.common_name = new_common
                    db_det.label_taxonomy_id = new_taxonomy_id
                    updated += 1
                    changed_file_ids.add(db_det.file_id)
                else:
                    unchanged += 1

            except Exception as e:
                logger.warning(f"Error updating detection: {e}")
                errors += 1

    # Final sweep: clear any non-verified detection whose label is
    # still excluded. Catches edge cases where smoothing re-introduces
    # an excluded label or rollup couldn't find an included ancestor.
    if excluded_taxonomy_ids:
        for det in detections:
            if det.verified:
                continue
            if (
                det.label_taxonomy_id
                and det.label_taxonomy_id in excluded_taxonomy_ids
            ):
                det.label = None
                det.label_confidence = None
                det.scientific_name = None
                det.common_name = None
                det.label_taxonomy_id = None
                changed_file_ids.add(det.file_id)
                updated += 1
    elif excluded_classes:
        excluded_lower = {name.lower() for name in excluded_classes}
        for det in detections:
            if det.verified:
                continue
            if det.label and det.label.lower() in excluded_lower:
                det.label = None
                det.label_confidence = None
                det.scientific_name = None
                det.common_name = None
                det.label_taxonomy_id = None
                changed_file_ids.add(det.file_id)
                updated += 1

    # Recompute observation_type for files with changed detections, from
    # the file's passing detections (over the project threshold or
    # verified), so a file left with only sub-threshold boxes reads "blank".
    threshold = 0.0
    _dep = db.get(Deployment, deployment_id)
    if _dep is not None:
        _proj = db.get(Project, _dep.project_id)
        if _proj is not None:
            threshold = _proj.detection_threshold
    for file_id in changed_file_ids:
        file_record = db.query(File).filter(File.id == file_id).first()
        if not file_record:
            continue

        file_detections = (
            db.query(Detection).filter(Detection.file_id == file_id).all()
        )
        file_record.observation_type = derive_observation_type(
            file_detections, threshold
        )

    db.commit()

    logger.info(
        f"Database update complete: {updated} updated, {unchanged} unchanged, "
        f"{errors} errors, {skipped_verified} skipped (verified)"
    )

    return {
        "updated": updated,
        "unchanged": unchanged,
        "errors": errors,
        "skipped_verified": skipped_verified,
    }


def reload_raw_classifications_from_json(
    deployment_id: str,
    json_path: Path,
    deployment_folder: Path,
    db: Session,
    excluded_classes: list[str] | None = None,
    taxonomy_csv_path: Path | None = None,
    excluded_names: frozenset[str] | None = None,
    allowed_taxonomy_keys: frozenset[str] | None = None,
) -> dict:
    """
    Reload raw (unsmoothed) classifications from JSON back to database.

    Effectively reverts to the original predictions by reading the raw JSON
    and updating DB records. Applies geofence-aware rollup when taxonomy
    is available.

    Args:
        deployment_id: Deployment UUID
        json_path: Path to results.json
        deployment_folder: Path to deployment folder
        db: Database session
        excluded_classes: Optional list of label names to exclude
        taxonomy_csv_path: Optional path to taxonomy.csv for rollup
        excluded_names: Lowercase excluded species names for rollup
        allowed_taxonomy_keys: Geofence taxonomy keys for rollup

    Returns:
        Dict with counts: {updated, unchanged, errors}
    """
    with open(json_path) as f:
        raw_results = json.load(f)

    from app.ml.label_exclusion import apply_label_exclusion_to_results

    taxonomy_lookup = None
    if taxonomy_csv_path and taxonomy_csv_path.exists():
        from app.ml.taxonomic_rollup import load_taxonomy_lookup

        taxonomy_lookup = load_taxonomy_lookup(taxonomy_csv_path)

    apply_label_exclusion_to_results(
        raw_results, excluded_classes, taxonomy_lookup
    )

    # Apply geofence-aware rollup (same logic as main postprocessing)
    if taxonomy_csv_path and taxonomy_csv_path.exists():
        from app.ml.taxonomic_rollup import (
            apply_taxonomic_rollup_to_results,
        )

        rollup_result = apply_taxonomic_rollup_to_results(
            raw_results,
            taxonomy_csv_path,
            excluded_names=excluded_names,
            allowed_taxonomy_keys=allowed_taxonomy_keys,
        )
        raw_results = rollup_result.md_results

    return update_database_from_smoothed_results(
        deployment_id, raw_results, deployment_folder, db, taxonomy_lookup,
        excluded_classes=excluded_classes,
    )
