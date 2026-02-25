"""
Embedding utilities — build input for embedding script, save results to database.

Following CONVENTIONS.md: crash early and loudly, no silent failures.
"""

import uuid
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.models import Detection, File
from app.models.detection_embedding import DetectionEmbedding

logger = get_logger(__name__)


def build_embedding_input(
    deployment_id: str,
    deployment_folder: Path,
    artifacts_folder: Path,
    db: Session,
) -> dict:
    """
    Query all detections for a deployment and build input for embedding_script.py.

    For video detections (frame_number is not None): resolve frame path from artifacts.
    For image detections: use File.file_path directly.

    Args:
        deployment_id: Deployment ID to query detections for
        deployment_folder: Root folder of the deployment
        artifacts_folder: Project-scoped artifacts folder (contains video_frames/)
        db: Database session

    Returns:
        JSON-serializable dict: {"detections": [{"detection_id", "image_path", "bbox"}]}
    """
    detections = (
        db.query(Detection, File)
        .join(File, Detection.file_id == File.id)
        .filter(File.deployment_id == deployment_id)
        .all()
    )

    video_frames_dir = artifacts_folder / "video_frames"
    entries = []

    for det, file in detections:
        # Resolve image path
        if det.frame_number is not None:
            # Video detection — resolve extracted frame
            frame_path = video_frames_dir / Path(file.file_path).name / f"frame{det.frame_number:06d}.jpg"
            if not frame_path.exists():
                logger.warning(
                    f"Frame not found for detection {det.id}: {frame_path}, skipping"
                )
                continue
            image_path = str(frame_path)
        else:
            # Image detection — use file path directly
            image_path = file.file_path
            if not Path(image_path).exists():
                logger.warning(
                    f"Image not found for detection {det.id}: {image_path}, skipping"
                )
                continue

        entries.append({
            "detection_id": det.id,
            "image_path": image_path,
            "bbox": [det.bbox_x, det.bbox_y, det.bbox_width, det.bbox_height],
        })

    logger.info(
        f"Built embedding input: {len(entries)} detections "
        f"({len(detections)} total, {len(detections) - len(entries)} skipped)"
    )

    return {"detections": entries}


def save_embeddings_to_db(
    npz_path: Path,
    job_id: str,
    embedding_model_id: str,
    embedding_dim: int,
    db: Session,
) -> int:
    """
    Load .npz file and bulk-insert DetectionEmbedding rows.

    Computes l2_norm per vector during insertion.
    Deletes existing embeddings for the same (detection_id, embedding_model_id) first.
    Flushes every 500 records to avoid excessive WAL growth.

    Args:
        npz_path: Path to .npz file (keys=detection_ids, values=float16 arrays)
        job_id: Job ID for tracking
        embedding_model_id: Model ID used for embedding
        embedding_dim: Expected embedding dimension
        db: Database session

    Returns:
        Count of inserted records

    Raises:
        RuntimeError: If .npz file is invalid or dimensions don't match
    """
    data = np.load(npz_path)
    detection_ids = list(data.files)

    if not detection_ids:
        logger.warning("No embeddings in .npz file")
        return 0

    # Validate first embedding dimension
    sample = data[detection_ids[0]]
    if sample.shape[0] != embedding_dim:
        raise RuntimeError(
            f"Dimension mismatch: expected {embedding_dim}, got {sample.shape[0]}"
        )

    # Delete existing embeddings for these detections with this model
    existing_count = (
        db.query(DetectionEmbedding)
        .filter(
            DetectionEmbedding.detection_id.in_(detection_ids),
            DetectionEmbedding.embedding_model_id == embedding_model_id,
        )
        .delete(synchronize_session=False)
    )
    if existing_count:
        logger.info(f"Deleted {existing_count} existing embeddings for re-embedding")

    # Bulk insert
    inserted = 0
    flush_interval = 500

    for det_id in detection_ids:
        vector = data[det_id]

        # Ensure float16
        if vector.dtype != np.float16:
            vector = vector.astype(np.float16)

        # Compute L2 norm (use float32 for precision)
        l2_norm = float(np.linalg.norm(vector.astype(np.float32)))

        embedding = DetectionEmbedding(
            id=str(uuid.uuid4()),
            detection_id=det_id,
            job_id=job_id,
            embedding_model_id=embedding_model_id,
            vector=vector.tobytes(),
            dimension=embedding_dim,
            l2_norm=l2_norm,
        )
        db.add(embedding)
        inserted += 1

        if inserted % flush_interval == 0:
            db.flush()

    db.commit()
    logger.info(f"Saved {inserted} embeddings to database")

    return inserted
