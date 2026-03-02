"""
Pydantic schemas for Similarity API.

Request/response models for similarity sort, search, and embedding stats.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class SimilarityFilters(BaseModel):
    """Filters for selecting detections to sort or search."""

    species: list[str] | None = None
    site_ids: list[str] | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    min_confidence: float | None = Field(None, ge=0.0, le=1.0)
    category: str | None = None
    verified: bool | None = None


class SortRequest(BaseModel):
    """Request body for similarity-sorting detections."""

    filters: SimilarityFilters = Field(default_factory=SimilarityFilters)
    reverse: bool = False


class SearchRequest(BaseModel):
    """Request body for finding similar detections."""

    anchor_detection_id: str
    filters: SimilarityFilters = Field(default_factory=SimilarityFilters)
    limit: int = Field(100, ge=1, le=500)
    threshold: float = Field(0.0, ge=-1.0, le=1.0)


class CropBbox(BaseModel):
    """Bbox position within the expanded crop (normalized 0-1)."""

    x: float
    y: float
    w: float
    h: float


class DetectionSummary(BaseModel):
    """Compact detection info for grid display."""

    detection_id: str
    file_id: str
    species: str | None
    species_confidence: float | None
    confidence: float
    category: str
    verified: bool
    classification_method: str | None
    distance_to_centroid: float | None = None
    similarity: float | None = None
    neighbor_agreement: float | None = None
    neighbor_top_label: str | None = None
    site_name: str | None = None
    deployment_id: str | None = None
    timestamp: datetime | None = None
    crop_url: str
    crop_bbox: CropBbox | None = None


class SortResponse(BaseModel):
    """Response for similarity sort endpoint."""

    detections: list[DetectionSummary]
    total_detections: int


class SearchResponse(BaseModel):
    """Response for search endpoint."""

    anchor: DetectionSummary
    results: list[DetectionSummary]
    total_results: int
    threshold_applied: float


class SimilarityStatsResponse(BaseModel):
    """Embedding coverage stats."""

    total_detections: int
    embedded_detections: int
    missing_embeddings: int
    embedding_model_id: str | None
    embedding_dimension: int | None
