"""
Embedding utilities — build input for embedding script, save results to database.

Following CONVENTIONS.md: crash early and loudly, no silent failures.
"""

import uuid
from pathlib import Path

import numpy as np
from sqlalchemy import or_
from sqlalchemy.orm import Session, contains_eager, load_only

from app.core.logging_config import get_logger
from app.db.sql_params import iter_id_chunks
from app.ml.label_exclusion import threshold_or_verified
from app.models import Detection, File, Track
from app.models.detection_embedding import DetectionEmbedding

logger = get_logger(__name__)


def video_pixels_for(det: Detection, file: File) -> tuple[str, list[float]] | None:
    """The JPEG on disk a video box has pixels in, and the box within it.

    A track's card is the crop the tracking script wrote: the whole
    file, so the box is the full frame. A card without a crop (a drawn
    box, a legacy one-frame track, a crop that failed to write) that sits
    on the video's cover frame is cut from the cover at its own box. Any
    other card has no picture on disk: the crop service decodes its
    frame on request, the embedder (a subprocess that cannot) skips it.
    """
    if det.frame_number is None:
        return None
    track = det.track
    if track is None or det.frame_number != track.representative_frame_number:
        return None
    if track.crop_path:
        return track.crop_path, [0.0, 0.0, 1.0, 1.0]
    if file.best_frame_number == det.frame_number and file.best_frame_path:
        return file.best_frame_path, [det.bbox_x, det.bbox_y, det.bbox_width, det.bbox_height]
    return None


def build_embedding_input(
    deployment_id: str,
    db: Session,
    *,
    min_confidence: float,
    skip_detection_ids: set[str] | None = None,
) -> dict:
    """
    Query a deployment's detections and build input for embedding_script.py.

    ``min_confidence`` is the project's classification gate: detections
    below it are neither classified nor embedded, so the near-noise
    tail MegaDetector emits at its 0.01 output cap never
    multiplies the per-crop model work. Verified detections always
    embed — a human said the box is real.

    Pure database work plus a dict; the heavy call sites run it off the
    event loop through `build_embedding_input_owned_session`.

    Image detections embed against `File.file_path`. A video detection
    embeds only when it is a track's card (`video_pixels_for`): the
    track's crop, or the cover frame for a card without one. The other
    boxes of a track have no card and no picture, and the embedder runs
    in a subprocess that cannot decode video, so the query leaves them
    in the database: a video box is loaded only on its track's
    representative frame. Until 2026-11-02 every box was loaded and
    sifted in Python, which on a BRUV deployment of 238,941 boxes (7
    clips, 11,696 tracks) grew the server past 3.5 GB and got it killed
    twice, once at the end of the run and once on "Embed now".

    Args:
        deployment_id: Deployment ID to query detections for
        db: Database session
        min_confidence: the project's classification gate
        skip_detection_ids: Detection IDs to skip (already embedded)

    Returns:
        JSON-serializable dict: {"detections": [{"detection_id", "image_path", "bbox"}]}
    """
    detections = (
        db.query(Detection, File)
        .join(File, Detection.file_id == File.id)
        .outerjoin(Track, Detection.track_id == Track.id)
        # Only the file columns the picture lookup reads. The join repeats
        # the file row on every card, and a file row carries its list of
        # sampled frames as JSON (about 100 KB for a 90 minute clip), so
        # the full row cost 1.25 GB of JSON across 11,696 cards, 4 GB once
        # parsed, and 30 s; without this the server was killed.
        .options(
            load_only(
                File.file_type, File.file_path, File.best_frame_number, File.best_frame_path
            ),
            contains_eager(Detection.track),
        )
        .filter(File.deployment_id == deployment_id)
        .filter(threshold_or_verified(min_confidence))
        # A clip's boxes come back only on their track's card frame; a
        # video box without a track has no card and stays behind too.
        .filter(
            or_(
                File.file_type != "video",
                Detection.frame_number == Track.representative_frame_number,
            )
        )
        .all()
    )

    entries = []
    skipped_no_pixels = 0
    skipped_no_bbox = 0

    for det, file in detections:
        if skip_detection_ids and det.id in skip_detection_ids:
            continue

        # Event-level observations carry no bbox and therefore no crop
        # for the embedder to read. Skip silently — they're a deliberate
        # part of the data, not a failure mode.
        if det.bbox_x is None:
            skipped_no_bbox += 1
            continue

        bbox = [det.bbox_x, det.bbox_y, det.bbox_width, det.bbox_height]
        if file.file_type == "video":
            pixels = video_pixels_for(det, file)
            if pixels is None:
                skipped_no_pixels += 1
                continue
            image_path, bbox = pixels
        else:
            image_path = file.file_path

        # No existence check here: the embedder skips a picture it cannot
        # open and says so on stderr. Checking first cost one stat per card
        # on the event loop, and on an ExFAT USB disk with thousands of
        # crops in one folder that froze the whole API for ten minutes.
        entries.append({
            "detection_id": det.id,
            "image_path": image_path,
            "bbox": bbox,
        })

    logger.info(
        f"Built embedding input: {len(entries)} detections "
        f"({len(detections)} total; "
        f"{skipped_no_pixels} video boxes without a picture on disk skipped; "
        f"{skipped_no_bbox} event-level observations skipped)"
    )

    return {"detections": entries}


def build_embedding_input_owned_session(
    deployment_id: str,
    *,
    min_confidence: float,
    skip_detection_ids: set[str] | None = None,
) -> dict:
    """Run `build_embedding_input` with a session created in THIS thread.

    The workers call this through ``asyncio.to_thread`` so the backend
    keeps answering while the cards of a big deployment are collected,
    the same reason Phase 6 and 7 have owned-session wrappers. A
    SQLAlchemy session belongs to one thread, so this makes its own
    rather than borrowing the worker's.
    """
    from app.db.base import get_session_factory

    db = get_session_factory()()
    try:
        return build_embedding_input(
            deployment_id,
            db,
            min_confidence=min_confidence,
            skip_detection_ids=skip_detection_ids,
        )
    finally:
        db.close()


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

    # Delete existing embeddings for these detections with this model. Chunk the
    # id list: one `IN (?, ?, ...)` over every id blows SQLite's bound-parameter
    # limit on large re-embeds (Simon's 50k+ detections crashed here with
    # "too many SQL variables"). See app/db/sql_params.
    existing_count = 0
    for chunk in iter_id_chunks(detection_ids):
        existing_count += (
            db.query(DetectionEmbedding)
            .filter(
                DetectionEmbedding.detection_id.in_(chunk),
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
