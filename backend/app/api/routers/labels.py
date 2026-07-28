"""
Labels API router.

Drives the Labels verify tab: embedding-based sort, nearest-neighbor
search, and embedding coverage stats. Heavy computation is delegated to
ml/inference/similarity_script.py via subprocess (no numpy/faiss needed here).
The "similarity" name on internal modules reflects the underlying technique;
user-facing surfaces are named after the unit of work, the label.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.schemas.label import (
    LabelStatsResponse,
    SearchRequest,
    SortRequest,
)
from app.core.confidence import DEFAULT_CLASSIFICATION_GATE
from app.db.base import get_db
from app.models import Deployment, Detection, DetectionEmbedding, File, Project
from app.services.label_service import (
    stream_cohorts_async,
    stream_search_async,
    stream_sort_async,
)
from app.utils.datetime_serialization import set_active_project_timezone

router = APIRouter(prefix="/api/projects", tags=["labels"])


def _set_project_tz(db: Session, project_id: str) -> None:
    """Activate the project's timezone in the request context."""
    tz = db.query(Project.timezone).filter(Project.id == project_id).scalar()
    if tz:
        set_active_project_timezone(tz)


@router.post("/{project_id}/labels/sort")
async def sort_detections(
    project_id: str,
    body: SortRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """Sort detections by visual similarity (greedy nearest-neighbor chain).

    Returns an `application/x-ndjson` event stream from the worker
    subprocess: progress lines while loading and computing, then a
    final `{"type":"result", ...}` line whose payload matches the
    legacy SortResponse shape. The frontend renders a progress bar
    from progress events and uses the result for the grid. When the
    client disconnects mid-stream (refresh, tab close, navigation)
    the subprocess is killed promptly, so the browser's per-host
    connection slot is freed and the next page load isn't queued.
    """
    _set_project_tz(db, project_id)
    try:
        stream = stream_sort_async(request, project_id, body, db)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from None
    return StreamingResponse(stream, media_type="application/x-ndjson")


@router.post("/{project_id}/labels/search")
async def search_similar(
    project_id: str,
    body: SearchRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """Find detections visually similar to an anchor detection.

    Same NDJSON event-stream shape as the sort endpoint; the final
    `result` payload matches SearchResponse.
    """
    _set_project_tz(db, project_id)
    try:
        stream = stream_search_async(request, project_id, body, db)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from None
    return StreamingResponse(stream, media_type="application/x-ndjson")


@router.get("/{project_id}/labels/cohorts")
async def cohorts(
    project_id: str,
    request: Request,
    min_count: int = Query(8, ge=1, le=1000),
    max_cohorts: int = Query(200, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """Group descendant-promotion suggestions for the Edit-step review panel.

    Returns an `application/x-ndjson` event stream identical in shape to
    the sort endpoint: progress lines during the FAISS + neighbour walk,
    then a final `{"type":"result", "cohorts":[...]}` line whose payload
    matches CohortsResponse. The panel renders a skeleton from progress
    events and uses the result to render its cards. Cohorts span the
    whole project, no filters apply.
    """
    _set_project_tz(db, project_id)
    try:
        stream = stream_cohorts_async(
            request, project_id, min_count, max_cohorts, db
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from None
    return StreamingResponse(stream, media_type="application/x-ndjson")


@router.get("/{project_id}/labels/unprocessed-count")
def get_unprocessed_count(
    project_id: str,
    min_confidence: float = 0.005,
    max_confidence: float = 1.0,
    db: Session = Depends(get_db),
) -> dict:
    """Count detections in a confidence range that could appear in the
    labels grid after an embedding backfill, but currently cannot: they
    are embeddable (bbox, on an embeddable surface) yet have no
    embedding for the project's current embedding model.

    Drives the grid's "unprocessed detections" banner. Purely
    data-driven, so it stays correct for projects whose deployments
    were analysed under different classification gates: whatever the
    historical gate was, an un-embedded detection shows up here.
    """
    from sqlalchemy import and_, exists, or_

    project = db.query(Project).filter(Project.id == project_id).first()
    embedding_model_id = project.embedding_model_id if project else None
    if not embedding_model_id:
        # No embedding model: the grid cannot show anything either way,
        # and the existing "Embed now" callout owns that situation.
        return {"count": 0}

    has_bbox = Detection.bbox_x.isnot(None)
    on_embeddable_surface = or_(
        File.file_type == "image",
        and_(
            File.file_type == "video",
            Detection.frame_number == File.best_frame_number,
        ),
    )
    has_embedding = exists().where(
        and_(
            DetectionEmbedding.detection_id == Detection.id,
            DetectionEmbedding.embedding_model_id == embedding_model_id,
        )
    )

    count = (
        db.query(func.count(Detection.id))
        .join(File, File.id == Detection.file_id)
        .join(Deployment, Deployment.id == File.deployment_id)
        .filter(Deployment.project_id == project_id)
        .filter(has_bbox)
        .filter(on_embeddable_surface)
        .filter(~has_embedding)
        .filter(Detection.confidence >= min_confidence)
        .filter(Detection.confidence <= max_confidence)
        .scalar()
    ) or 0

    return {"count": int(count)}


@router.get(
    "/{project_id}/labels/stats",
    response_model=LabelStatsResponse,
)
def get_label_stats(
    project_id: str,
    db: Session = Depends(get_db),
):
    """Get embedding coverage stats for a project."""
    # Get embedding model from project first (needed to filter embedded count)
    project = db.query(Project).filter(Project.id == project_id).first()
    embedding_model_id = project.embedding_model_id if project else None

    # Embeddability gate. A detection is embeddable when:
    #  - it has a bbox (event-level labels are bbox-less and have
    #    no crop to embed), AND
    #  - it sits on a pixel surface the embedding worker can read:
    #    images embed unconditionally; video detections only embed when
    #    they sit on the parent video's best frame (matches
    #    `build_embedding_input` in embedding_utils.py).
    # Non-embeddable detections are invisible to the Labels grid
    # and similarity search anyway, so we leave them out of the
    # "missing embeddings" count — otherwise the banner would chase a
    # population that `/embed-now` is deliberately designed to skip.
    from sqlalchemy import and_, exists, or_
    has_bbox = Detection.bbox_x.isnot(None)
    on_embeddable_surface = or_(
        File.file_type == "image",
        and_(
            File.file_type == "video",
            Detection.frame_number == File.best_frame_number,
        ),
    )
    embeddable_clause = and_(has_bbox, on_embeddable_surface)

    # Only detections at or above the classification gate (or verified)
    # are SUPPOSED to be embedded — MegaDetector runs untresholded, so
    # the below-gate tail is deliberately unprocessed and must not
    # count as "missing" here (the grid's separate unprocessed-range
    # banner owns that population, with its own backfill action).
    gate = (
        project.classification_gate
        if project
        else DEFAULT_CLASSIFICATION_GATE
    )
    above_gate_or_verified = or_(
        Detection.confidence >= gate,
        Detection.verified == True,  # noqa: E712
    )

    total = (
        db.query(func.count(Detection.id))
        .join(File, File.id == Detection.file_id)
        .join(Deployment, Deployment.id == File.deployment_id)
        .filter(Deployment.project_id == project_id)
        .filter(embeddable_clause)
        .filter(above_gate_or_verified)
        .scalar()
    ) or 0

    # Verification progress pill. This must count the population the
    # Labels view actually shows — detections that HAVE an
    # embedding AND pass the project's threshold-or-verified rule —
    # mirroring the similarity-sort query. Counting the raw embeddable
    # population instead (no threshold, embedding not required) would
    # report e.g. 95% while every on-screen label is verified,
    # because below-threshold / not-yet-embedded detections drag it
    # down even though they never appear here.
    threshold = project.counting_threshold if project else 0.0
    has_embedding = exists().where(
        DetectionEmbedding.detection_id == Detection.id
    )
    floor_or_verified = or_(
        Detection.confidence >= threshold,
        Detection.verified == True,  # noqa: E712
    )
    reviewable = (
        db.query(func.count(Detection.id))
        .join(File, File.id == Detection.file_id)
        .join(Deployment, Deployment.id == File.deployment_id)
        .filter(Deployment.project_id == project_id)
        .filter(has_embedding)
        .filter(floor_or_verified)
        .scalar()
    ) or 0
    verified = (
        db.query(func.count(Detection.id))
        .join(File, File.id == Detection.file_id)
        .join(Deployment, Deployment.id == File.deployment_id)
        .filter(Deployment.project_id == project_id)
        .filter(has_embedding)
        .filter(Detection.verified == True)  # noqa: E712
        .scalar()
    ) or 0

    # Detections with embeddings for the current model
    if embedding_model_id:
        embedded = (
            db.query(func.count(func.distinct(DetectionEmbedding.detection_id)))
            .join(Detection, Detection.id == DetectionEmbedding.detection_id)
            .join(File, File.id == Detection.file_id)
            .join(Deployment, Deployment.id == File.deployment_id)
            .filter(Deployment.project_id == project_id)
            .filter(embeddable_clause)
            .filter(DetectionEmbedding.embedding_model_id == embedding_model_id)
            .scalar()
        ) or 0
    else:
        embedded = 0

    # Get dimension from first embedding
    dim_row = (
        db.query(DetectionEmbedding.dimension)
        .join(Detection, Detection.id == DetectionEmbedding.detection_id)
        .join(File, File.id == Detection.file_id)
        .join(Deployment, Deployment.id == File.deployment_id)
        .filter(Deployment.project_id == project_id)
        .limit(1)
        .first()
    )

    return LabelStatsResponse(
        total_detections=total,
        reviewable_detections=reviewable,
        verified_detections=verified,
        embedded_detections=embedded,
        missing_embeddings=total - embedded,
        embedding_model_id=embedding_model_id,
        embedding_dimension=dim_row[0] if dim_row else None,
    )
