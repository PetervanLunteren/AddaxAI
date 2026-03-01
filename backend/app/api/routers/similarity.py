"""
Similarity API router.

Provides similarity-sort and nearest-neighbor search endpoints
for detection embeddings. Delegates heavy computation to
similarity_script.py via subprocess (no numpy/faiss needed here).
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.schemas.similarity import (
    SearchRequest,
    SearchResponse,
    SimilarityStatsResponse,
    SortRequest,
    SortResponse,
)
from app.db.base import get_db
from app.models import Deployment, Detection, DetectionEmbedding, File, Project, Site
from app.services.similarity_service import (
    search_similar as search_similar_service,
    sort_detections as sort_detections_service,
)

router = APIRouter(prefix="/api/projects", tags=["similarity"])


@router.post("/{project_id}/similarity/sort", response_model=SortResponse)
def sort_detections(
    project_id: str,
    body: SortRequest,
    db: Session = Depends(get_db),
):
    """Sort detections by visual similarity using greedy nearest-neighbor chain."""
    try:
        return sort_detections_service(project_id, body, db)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{project_id}/similarity/search", response_model=SearchResponse)
def search_similar(
    project_id: str,
    body: SearchRequest,
    db: Session = Depends(get_db),
):
    """Find detections visually similar to an anchor detection."""
    try:
        return search_similar_service(project_id, body, db)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/{project_id}/similarity/stats",
    response_model=SimilarityStatsResponse,
)
def get_similarity_stats(
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
        .join(Site, Site.id == Deployment.site_id)
        .filter(Site.project_id == project_id)
        .scalar()
    ) or 0

    # Detections with embeddings for the current model
    if embedding_model_id:
        embedded = (
            db.query(func.count(func.distinct(DetectionEmbedding.detection_id)))
            .join(Detection, Detection.id == DetectionEmbedding.detection_id)
            .join(File, File.id == Detection.file_id)
            .join(Deployment, Deployment.id == File.deployment_id)
            .join(Site, Site.id == Deployment.site_id)
            .filter(Site.project_id == project_id)
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
        .join(Site, Site.id == Deployment.site_id)
        .filter(Site.project_id == project_id)
        .limit(1)
        .first()
    )

    return SimilarityStatsResponse(
        total_detections=total,
        embedded_detections=embedded,
        missing_embeddings=total - embedded,
        embedding_model_id=embedding_model_id,
        embedding_dimension=dim_row[0] if dim_row else None,
    )
