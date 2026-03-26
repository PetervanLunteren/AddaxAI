"""
Postprocessing worker for reprocessing classification results.

Applies event smoothing and taxonomic rollup to all deployments in a project
by reading raw predictions from JSON files and writing smoothed results to DB.

Created by Claude Code on 2026-02-14
"""

from pathlib import Path

from sqlalchemy import func, text
from sqlalchemy.orm import Session

from app.api.crud import event as event_crud
from app.api.crud import job as job_crud
from app.api.crud import project as project_crud
from app.core.logging_config import get_logger
from app.core.websocket_manager import ws_manager
from app.db.base import get_db
from app.ml.postprocessing import (
    compute_postprocessing_settings_hash,
    reload_raw_classifications_from_json,
    run_postprocessing_for_deployment,
    update_database_from_smoothed_results,
)
from app.models import Deployment, Detection, File, Site

logger = get_logger(__name__)


def _get_label_counts(db: Session, project_id: str) -> dict[str, int]:
    """Query label detection counts for a project."""
    rows = (
        db.query(Detection.label, func.count(Detection.id))
        .join(File)
        .join(Deployment)
        .join(Site)
        .filter(Site.project_id == project_id)
        .filter(Detection.label.isnot(None))
        .group_by(Detection.label)
        .all()
    )
    return {label: count for label, count in rows}


def _build_label_diff(
    before: dict[str, int], after: dict[str, int]
) -> list[dict]:
    """Build list of labels where counts changed, sorted by absolute change."""
    all_labels = set(before) | set(after)
    diff = []
    for lbl in all_labels:
        b = before.get(lbl, 0)
        a = after.get(lbl, 0)
        if b != a:
            diff.append({"label": lbl, "before": b, "after": a})
    diff.sort(key=lambda d: abs(d["after"] - d["before"]), reverse=True)
    return diff


async def process_postprocessing_job(job_id: str) -> None:
    """
    Process a postprocessing (reprocess) job for a project.

    For each deployment with classifications:
    - If smoothing enabled: run smoothing and update DB
    - If smoothing disabled: reload raw predictions from JSON

    Args:
        job_id: Job ID to process
    """
    db = next(get_db())

    try:
        # Get job and payload
        job = job_crud.get_job(db, job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        payload = job.payload or {}
        project_id = payload.get("project_id")
        if not project_id:
            raise ValueError("Missing project_id in job payload")

        project = project_crud.get_project(db, project_id)
        if not project:
            raise ValueError(f"Project not found: {project_id}")

        job_crud.update_job_status(db, job_id, "running")
        await ws_manager.send_progress(job_id, "Starting postprocessing...", 0.0)

        smoothing_enabled = project.event_smoothing or project.taxonomic_rollup

        # Find all deployments with classifications
        deployments_with_cls = (
            db.query(Deployment)
            .join(Site)
            .filter(Site.project_id == project_id)
            .join(File, File.deployment_id == Deployment.id)
            .join(Detection, Detection.file_id == File.id)
            .filter(Detection.label.isnot(None))
            .distinct()
            .all()
        )

        total = len(deployments_with_cls)
        if total == 0:
            logger.info("No deployments with classifications found")
            await ws_manager.send_progress(job_id, "No classifications to process", 1.0)
            job_crud.update_job_status(db, job_id, "completed")
            await ws_manager.send_complete(
                job_id=job_id,
                success=True,
                message="No deployments with classifications found",
            )
            db.close()
            return

        logger.info(f"Processing {total} deployments for project {project.name}")

        # Resolve taxonomy CSV for exclusion rollup
        taxonomy_csv = None
        if project.classification_model_id:
            from app.core.config import get_settings

            settings = get_settings()
            _cls_dir = (
                settings.user_data_dir / "models" / "cls"
                / project.classification_model_id
            )
            if _cls_dir.exists():
                _tax = _cls_dir / "taxonomy.csv"
                if _tax.exists():
                    taxonomy_csv = _tax

        # Snapshot label counts before processing
        before_counts = _get_label_counts(db, project_id)

        total_updated = 0
        total_errors = 0

        for idx, deployment in enumerate(deployments_with_cls, start=1):
            folder_path = Path(deployment.folder_path)
            json_path = folder_path / ".addaxai" / "projects" / project_id / "results.json"

            progress = (idx - 1) / total
            await ws_manager.send_progress(
                job_id,
                f"Processing deployment {idx}/{total}...",
                progress,
            )

            if not json_path.exists():
                logger.warning(
                    f"JSON file not found for deployment {deployment.id}: {json_path}"
                )
                continue

            try:
                if smoothing_enabled:
                    smoothed = run_postprocessing_for_deployment(
                        deployment.id, json_path, folder_path, project, db
                    )
                    result = update_database_from_smoothed_results(
                        deployment.id, smoothed, folder_path, db
                    )
                else:
                    result = reload_raw_classifications_from_json(
                        deployment.id, json_path, folder_path, db,
                        excluded_classes=project.excluded_classes,
                        taxonomy_csv_path=taxonomy_csv,
                    )

                total_updated += result.get("updated", 0)
                total_errors += result.get("errors", 0)
                logger.info(
                    f"Deployment {idx}/{total}: "
                    f"{result.get('updated', 0)} updated, "
                    f"{result.get('unchanged', 0)} unchanged"
                )

            except Exception as e:
                logger.error(
                    f"Postprocessing failed for deployment {deployment.id}: {e}",
                    exc_info=True,
                )
                total_errors += 1

        # Ensure base taxonomy is populated (handles reprocessing with new model)
        if project.classification_model_id:
            try:
                from app.ml.postprocessing import _find_classification_model_dir
                from app.ml.taxonomy_db import populate_taxonomy_from_csv

                cls_model_dir = _find_classification_model_dir(project, db)
                if cls_model_dir:
                    taxonomy_csv = cls_model_dir / "taxonomy.csv"
                    if taxonomy_csv.exists():
                        populate_taxonomy_from_csv(
                            project.classification_model_id, taxonomy_csv, db
                        )
            except Exception as e:
                logger.warning(f"Failed to populate taxonomy DB: {e}")

        # Link detections to taxonomy rows via FK
        try:
            from app.ml.taxonomy_db import link_detections_to_taxonomy

            link_detections_to_taxonomy(project_id, db)
        except Exception as e:
            logger.warning(f"Failed to link detections to taxonomy: {e}")

        # Update project hash
        project.postprocessing_settings_hash = compute_postprocessing_settings_hash(
            project
        )
        db.commit()

        # Expire cached state so the count query hits the DB fresh
        db.expire_all()

        # Snapshot label counts after processing and compute diff
        after_counts = _get_label_counts(db, project_id)
        label_diff = _build_label_diff(before_counts, after_counts)

        # Report completion
        action = "Smoothing applied" if smoothing_enabled else "Raw predictions restored"
        message = f"{action} across {total} deployments ({total_updated} detections updated)"

        # Auto-regenerate events (independence_interval may have changed)
        event_count = event_crud.generate_events_for_project(db, project_id)
        logger.info(f"Postprocessing job {job_id}: Regenerated {event_count} events")

        db.execute(text("ANALYZE"))
        db.commit()

        job_crud.update_job_status(db, job_id, "completed")
        await ws_manager.send_progress(job_id, message, 1.0)
        await ws_manager.send_complete(
            job_id=job_id,
            success=True,
            message=message,
            data={
                "deployments_processed": total,
                "detections_updated": total_updated,
                "errors": total_errors,
                "label_diff": label_diff,
            },
        )

        logger.info(f"Postprocessing job {job_id} completed: {message}")

    except Exception as e:
        logger.error(f"Postprocessing job {job_id} failed: {e}", exc_info=True)

        try:
            job_crud.update_job_status(db, job_id, "failed")
        except Exception:
            pass

        await ws_manager.send_error(job_id, str(e))

    finally:
        db.close()
