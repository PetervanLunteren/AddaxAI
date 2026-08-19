"""
Postprocessing worker for reprocessing classification results.

Applies event smoothing and taxonomic rollup to all deployments in a project
by reading raw predictions from JSON files and writing smoothed results to DB.

Created by Claude Code on 2026-02-14
"""

import asyncio
from pathlib import Path

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.crud import event as event_crud
from app.api.crud import job as job_crud
from app.api.crud import project as project_crud
from app.core.logging_config import get_logger
from app.core.websocket_manager import ws_manager
from app.db.base import get_db, refresh_query_statistics
from app.ml.postprocessing import (
    compute_postprocessing_settings_hash,
    reload_raw_classifications_from_json,
    run_postprocessing_for_deployment,
    update_database_from_smoothed_results,
)
from app.models import Deployment, Detection, File

logger = get_logger(__name__)


def _get_label_counts(db: Session, project_id: str) -> dict[str, int]:
    """Query label detection counts for a project."""
    rows = (
        db.query(Detection.label, func.count(Detection.id))
        .join(File)
        .join(Deployment)
        .filter(Deployment.project_id == project_id)
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
            .filter(Deployment.project_id == project_id)
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

        # Resolve classification model dir and taxonomy CSV
        taxonomy_csv = None
        cls_model_dir = None
        if project.classification_model_id:
            from app.core.config import get_settings

            settings = get_settings()
            cls_model_dir = (
                settings.models_dir / "cls"
                / project.classification_model_id
            )
            if cls_model_dir.exists():
                _tax = cls_model_dir / "taxonomy.csv"
                if _tax.exists():
                    taxonomy_csv = _tax
            else:
                cls_model_dir = None

        # Snapshot label counts before processing
        before_counts = _get_label_counts(db, project_id)

        total_updated = 0
        total_errors = 0
        # Deployments the reprocess cannot touch. Their labels keep the
        # settings from the run that wrote them, so the job has to own up to
        # it rather than report a clean success over every deployment.
        #
        # Split by cause, because each needs a different fix: a folder that
        # is gone needs reconnecting, a folder that lost its hidden .addaxai
        # artifacts needs analysing again, an unreadable one needs unlocking.
        # One example path per cause is all the message shows, which also
        # keeps this payload flat on a project holding hundreds of folders.
        skipped: dict[str, dict] = {}

        def _skip(cause: str, path: str) -> None:
            entry = skipped.setdefault(cause, {"count": 0, "path": path})
            entry["count"] += 1

        loop = asyncio.get_event_loop()

        for idx, deployment in enumerate(deployments_with_cls, start=1):
            # Emit progress BEFORE the heavy work for this deployment and
            # include metrics so the modal can render a real counter.
            progress = (idx - 1) / total
            await ws_manager.send_progress(
                job_id,
                f"Processing deployment {idx}/{total}...",
                progress,
                phase="postprocessing",
                phase_progress=progress,
                data={
                    "metrics": {
                        "current": idx,
                        "total": total,
                        "unit": "deployment",
                    }
                },
            )

            # A deployment can legitimately carry no folder (imported or
            # hand-made rows); crud/deployment.py handles that everywhere
            # else. Path(None) here would take the whole job down with a
            # TypeError, taking every other deployment with it.
            if not deployment.folder_path:
                logger.warning(f"Deployment {deployment.id} has no folder path")
                _skip("no_folder", str(deployment.id))
                continue

            folder_path = Path(deployment.folder_path)
            json_path = folder_path / ".addaxai" / "projects" / project_id / "results.json"

            # Path.exists() re-raises EACCES (only ENOENT and its family are
            # swallowed), so one unreadable folder — a locked share, a disk
            # owned by another user — used to abort the whole project's
            # reprocess from here. Report it as one skipped folder instead.
            try:
                results_missing = not json_path.exists()
                folder_gone = not folder_path.exists()
            except OSError as e:
                logger.warning(f"Cannot read {folder_path}: {e}")
                _skip("unreadable", str(folder_path))
                continue

            if results_missing:
                logger.warning(
                    f"JSON file not found for deployment {deployment.id}: {json_path}"
                )
                _skip(
                    "folder_missing" if folder_gone else "no_results",
                    str(folder_path),
                )
                continue

            try:
                # Resolve excluded_classes to taxonomy UUIDs
                excluded_tax_ids: set[str] | None = None
                if project.excluded_classes:
                    from app.models.label_taxonomy import LabelTaxonomy

                    exc_rows = (
                        db.query(LabelTaxonomy.id)
                        .filter(
                            LabelTaxonomy.name.in_(
                                project.excluded_classes
                            ),
                        )
                        .all()
                    )
                    excluded_tax_ids = {r[0] for r in exc_rows}

                if smoothing_enabled:
                    # Heavy: subprocess + rollup. Run off-loop so the
                    # progress frame we just emitted actually flushes
                    # to the client.
                    smoothed = await loop.run_in_executor(
                        None,
                        run_postprocessing_for_deployment,
                        deployment.id,
                        json_path,
                        folder_path,
                        project,
                        db,
                    )
                    # Load taxonomy for scientific_name formatting
                    pp_taxonomy = None
                    if taxonomy_csv and taxonomy_csv.exists():
                        from app.ml.taxonomic_rollup import load_taxonomy_lookup

                        pp_taxonomy = load_taxonomy_lookup(taxonomy_csv)

                    # Resolve label names to taxonomy IDs
                    from app.ml.taxonomy_db import batch_resolve_taxonomy_ids

                    pp_name_to_id = batch_resolve_taxonomy_ids(
                        list(
                            smoothed.get(
                                "classification_categories", {}
                            ).values()
                        ),
                        project.classification_model_id,
                        project_id,
                        db,
                    ) if project.classification_model_id else None

                    def _apply_smoothed(
                        _dep_id=deployment.id,
                        _sm=smoothed,
                        _fp=folder_path,
                        _ptax=pp_taxonomy,
                        _exc=project.excluded_classes,
                        _exc_tax=excluded_tax_ids,
                        _n2i=pp_name_to_id,
                    ):
                        return update_database_from_smoothed_results(
                            _dep_id, _sm, _fp, db, _ptax,
                            excluded_classes=_exc,
                            excluded_taxonomy_ids=_exc_tax,
                            taxonomy_name_to_id=_n2i,
                        )

                    result = await loop.run_in_executor(None, _apply_smoothed)
                else:
                    # Build excluded_names and geofence keys for rollup
                    excluded_names = frozenset(
                        n.lower()
                        for n in (project.excluded_classes or [])
                    )
                    geo_keys = None
                    if cls_model_dir and project.country_code:
                        try:
                            from app.ml.geofence import (
                                get_allowed_taxonomy_keys,
                            )

                            geo_keys = get_allowed_taxonomy_keys(
                                cls_model_dir,
                                project.country_code,
                                project.state_code,
                            )
                        except FileNotFoundError:
                            pass
                    def _reload_raw(
                        _dep_id=deployment.id,
                        _jp=json_path,
                        _fp=folder_path,
                        _exc=project.excluded_classes,
                        _tax_csv=taxonomy_csv,
                        _exc_names=excluded_names,
                        _geo=geo_keys,
                    ):
                        return reload_raw_classifications_from_json(
                            _dep_id, _jp, _fp, db,
                            excluded_classes=_exc,
                            taxonomy_csv_path=_tax_csv,
                            excluded_names=_exc_names,
                            allowed_taxonomy_keys=_geo,
                        )

                    result = await loop.run_in_executor(None, _reload_raw)

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
                # The settings did not reach this folder either, so it counts
                # as skipped. Reported as unreadable because the first thing
                # this block does is open and parse results.json, which is
                # what a damaged or locked file fails on.
                _skip("unreadable", str(folder_path))

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

        # Update project hash, but only when the settings really did reach
        # every deployment. The hash is what /postprocessing-status compares
        # to decide whether the project still needs a reprocess; stamping it
        # after a skip makes the app claim the labels match settings they
        # were never built with.
        if not skipped:
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
        # "folders", not "deployments": the same modal renders this line for
        # the folder-run flow, which has no deployments. And "Settings
        # applied" rather than naming smoothing: `smoothing_enabled` is true
        # whenever rollup is on, so the old line claimed smoothing on
        # projects that have it switched off.
        n_skipped = sum(entry["count"] for entry in skipped.values())
        done = total - n_skipped
        if done == 0:
            message = "Could not apply settings to any folder"
        else:
            message = (
                f"Settings applied to {done} of {total} folders "
                f"({total_updated} detections updated)"
            )
            if n_skipped:
                message += f", {n_skipped} skipped"

        # Auto-regenerate events (independence_interval may have changed)
        event_count = event_crud.generate_events_for_project(db, project_id)
        logger.info(f"Postprocessing job {job_id}: Regenerated {event_count} events")

        refresh_query_statistics(db)
        db.commit()

        job_crud.update_job_status(db, job_id, "completed")
        await ws_manager.send_progress(job_id, message, 1.0)
        await ws_manager.send_complete(
            job_id=job_id,
            success=True,
            message=message,
            data={
                "deployments_processed": total,
                "skipped": skipped,
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
