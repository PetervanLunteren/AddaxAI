"""
Observations API router.

Drives the Observations verify tab: embedding-based sort, nearest-neighbor
search, and embedding coverage stats. Heavy computation is delegated to
ml/inference/similarity_script.py via subprocess (no numpy/faiss needed here).
The "similarity" name on internal modules reflects the underlying technique;
user-facing surfaces are named after the unit of work, the observation.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.schemas.observation import (
    ObservationStatsResponse,
    SearchRequest,
    SearchResponse,
    SortRequest,
    SortResponse,
)
from app.db.base import get_db
from app.models import Deployment, Detection, DetectionEmbedding, File, Project
from app.services.observation_service import (
    search_similar as search_similar_service,
)
from app.services.observation_service import (
    sort_detections as sort_detections_service,
)
from app.utils.datetime_serialization import set_active_project_timezone

router = APIRouter(prefix="/api/projects", tags=["observations"])


def _set_project_tz(db: Session, project_id: str) -> None:
    """Activate the project's timezone in the request context."""
    tz = db.query(Project.timezone).filter(Project.id == project_id).scalar()
    if tz:
        set_active_project_timezone(tz)


@router.post("/{project_id}/observations/sort", response_model=SortResponse)
async def sort_detections(
    project_id: str,
    body: SortRequest,
    db: Session = Depends(get_db),
):
    """Sort detections by visual similarity using greedy nearest-neighbor chain."""
    _set_project_tz(db, project_id)
    try:
        return sort_detections_service(project_id, body, db)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from None


@router.post("/{project_id}/observations/search", response_model=SearchResponse)
async def search_similar(
    project_id: str,
    body: SearchRequest,
    db: Session = Depends(get_db),
):
    """Find detections visually similar to an anchor detection."""
    _set_project_tz(db, project_id)
    try:
        return search_similar_service(project_id, body, db)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from None


@router.get(
    "/{project_id}/observations/stats",
    response_model=ObservationStatsResponse,
)
def get_observation_stats(
    project_id: str,
    db: Session = Depends(get_db),
):
    """Get embedding coverage stats for a project."""
    # Get embedding model from project first (needed to filter embedded count)
    project = db.query(Project).filter(Project.id == project_id).first()
    embedding_model_id = project.embedding_model_id if project else None

    # Total detections in project
    total = (
        db.query(func.count(Detection.id))
        .join(File, File.id == Detection.file_id)
        .join(Deployment, Deployment.id == File.deployment_id)
        .filter(Deployment.project_id == project_id)
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

    return ObservationStatsResponse(
        total_detections=total,
        embedded_detections=embedded,
        missing_embeddings=total - embedded,
        embedding_model_id=embedding_model_id,
        embedding_dimension=dim_row[0] if dim_row else None,
    )
