"""
Embedding worker — incrementally embeds detections missing embeddings for the current model.

Iterates all deployments with detections, skips already-embedded ones,
and saves new embeddings to DB. Existing embeddings for the same
(detection_id, model) pair are replaced by save_embeddings_to_db.
"""

import asyncio
import json as _json
from pathlib import Path

from sqlalchemy import text

from app.api.crud import job as job_crud
from app.api.crud import project as project_crud
from app.core.logging_config import get_logger
from app.core.websocket_manager import ws_manager
from app.db.base import get_db
from app.ml.embedding_utils import build_embedding_input, save_embeddings_to_db
from app.models import Deployment, Detection, File, Site
from app.models.detection_embedding import DetectionEmbedding

logger = get_logger(__name__)


async def process_re_embedding_job(job_id: str) -> None:
    """
    Re-embed all detections in a project with a new embedding model.

    For each deployment with detections:
    - Build embedding input from detection crops
    - Run embedding model subprocess
    - Save embeddings to DB (replaces existing for same model)

    Args:
        job_id: Job ID to process
    """
    db = next(get_db())

    try:
        job = job_crud.get_job(db, job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        payload = job.payload or {}
        project_id = payload.get("project_id")
        embedding_model_id = payload.get("embedding_model_id")
        if not project_id:
            raise ValueError("Missing project_id in job payload")
        if not embedding_model_id:
            raise ValueError("Missing embedding_model_id in job payload")

        project = project_crud.get_project(db, project_id)
        if not project:
            raise ValueError(f"Project not found: {project_id}")

        job_crud.update_job_status(db, job_id, "running")
        await ws_manager.send_progress(job_id, "Starting embedding...", 0.0)

        # Find all deployments with detections
        deployments = (
            db.query(Deployment)
            .join(Site)
            .filter(Site.project_id == project_id)
            .join(File, File.deployment_id == Deployment.id)
            .join(Detection, Detection.file_id == File.id)
            .distinct()
            .all()
        )

        total = len(deployments)
        if total == 0:
            logger.info("No deployments with detections found")
            await ws_manager.send_progress(job_id, "No detections to embed", 1.0)
            job_crud.update_job_status(db, job_id, "completed")
            await ws_manager.send_complete(
                job_id=job_id,
                success=True,
                message="No detections to embed",
            )
            db.close()
            return

        logger.info(f"Re-embedding {total} deployments for project {project.name}")

        # Initialize embedding model once
        from app.ml.environment_manager import EnvironmentManager
        from app.ml.inference.embedding_model import EmbeddingModel
        from app.ml.manifest_manager import ManifestManager
        from app.ml.model_storage import ModelStorage

        manifest_manager = ManifestManager()
        env_manager = EnvironmentManager()
        model_storage = ModelStorage()

        emb_manifest = manifest_manager.get_model(embedding_model_id)
        emb_model_path = model_storage.get_model_file(emb_manifest)
        embedding_model = EmbeddingModel(emb_model_path, emb_manifest, env_manager)

        total_embedded = 0
        total_errors = 0
        loop = asyncio.get_event_loop()

        for idx, deployment in enumerate(deployments, start=1):
            progress = (idx - 1) / total

            # Send deployment context message (matches useTaskProgress protocol)
            await ws_manager.send_progress(
                job_id,
                "Computing embeddings...",
                progress,
                phase="embedding",
                phase_progress=0.0,
                data={
                    "deployment_index": idx,
                    "total_deployments": total,
                    "video_count": 0,
                    "image_count": 0,
                    "has_classifier": False,
                    "has_embedding": True,
                },
            )

            try:
                # Find detections already embedded with the current model
                already_embedded = set(
                    row[0]
                    for row in db.query(DetectionEmbedding.detection_id)
                    .join(Detection, Detection.id == DetectionEmbedding.detection_id)
                    .join(File, File.id == Detection.file_id)
                    .filter(
                        File.deployment_id == deployment.id,
                        DetectionEmbedding.embedding_model_id == embedding_model_id,
                    )
                    .all()
                )

                input_data = build_embedding_input(
                    deployment.id,
                    db,
                    skip_detection_ids=already_embedded,
                )
                if not input_data["detections"]:
                    logger.info(f"Deployment {deployment.id}: no valid detections, skipping")
                    continue

                # Write temp input JSON
                artifacts_folder = Path(deployment.folder_path) / ".addaxai"
                artifacts_folder.mkdir(parents=True, exist_ok=True)
                embedding_input_json = artifacts_folder / "re_embedding_input.json"
                embedding_output_npz = artifacts_folder / "re_embeddings.npz"

                with open(embedding_input_json, "w") as f:
                    _json.dump(input_data, f)

                # Sync progress callback (same pattern as detection_worker.py)
                def sync_embedding_progress(
                    message: str,
                    phase_progress: float,
                    metrics: dict | None = None,
                    _idx=idx,
                    _progress=progress,
                ) -> None:
                    """Sync wrapper that schedules async callback from executor thread."""
                    data: dict = {
                        "deployment_index": _idx,
                        "total_deployments": total,
                        "video_count": 0,
                        "image_count": 0,
                        "has_classifier": False,
                        "has_embedding": True,
                    }
                    if metrics:
                        metrics["unit"] = "crop"
                        if "compute_device" in metrics:
                            data["compute_device"] = metrics.pop("compute_device")
                        data["metrics"] = metrics
                    asyncio.run_coroutine_threadsafe(
                        ws_manager.send_progress(
                            job_id,
                            message,
                            _progress,
                            "embedding",
                            phase_progress,
                            data,
                        ),
                        loop,
                    )

                # Run embedding in executor (blocking subprocess)
                embedded_count = await loop.run_in_executor(
                    None,
                    lambda cb=sync_embedding_progress,
                    _eij=embedding_input_json,
                    _eon=embedding_output_npz,
                    _bs=project.embedding_batch_size: embedding_model.compute_embeddings(
                        _eij, _eon, _bs, cb
                    ),
                )

                # Save to DB
                if embedding_output_npz.exists():
                    save_embeddings_to_db(
                        embedding_output_npz,
                        job_id,
                        embedding_model_id,
                        emb_manifest.embedding_dim,
                        db,
                    )
                    total_embedded += embedded_count

                # Clean up temp files
                embedding_input_json.unlink(missing_ok=True)
                embedding_output_npz.unlink(missing_ok=True)

                logger.info(f"Deployment {idx}/{total}: {embedded_count} detections embedded")

            except Exception as e:
                logger.error(
                    f"Re-embedding failed for deployment {deployment.id}: {e}",
                    exc_info=True,
                )
                total_errors += 1

        # Report completion
        message = f"Re-embedded {total_embedded} detections across {total} deployments"
        if total_errors:
            message += f" ({total_errors} errors)"

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
                "detections_embedded": total_embedded,
                "errors": total_errors,
            },
        )

        logger.info(f"Re-embedding job {job_id} completed: {message}")

    except Exception as e:
        logger.error(f"Re-embedding job {job_id} failed: {e}", exc_info=True)

        try:
            job_crud.update_job_status(db, job_id, "failed")
        except Exception:
            pass

        await ws_manager.send_error(job_id, str(e))

    finally:
        db.close()
