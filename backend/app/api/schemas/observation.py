"""
Pydantic schemas for the Observations API.

Request/response models for the Observations verify tab: sort (greedy
nearest-neighbor chain), search (find similar), and embedding coverage
stats. The underlying technique is cosine similarity on DINOv2 crop
embeddings; the schemas here are named after the tab (observations),
not the algorithm.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_serializer

from app.utils.datetime_serialization import serialize_local_datetime

ObservationSort = Literal[
    "similarity",
    "similarity_reverse",
    "newest",
    "oldest",
    "cls_low",
]


class ObservationFilters(BaseModel):
    """Filters for selecting detections to sort or search.

    `project_floor` is server-injected (not user-facing) and applied as
    `(confidence >= floor OR verified)` — the global threshold + verified
    override rule. `min_confidence` / `max_confidence` are user knobs and
    apply LITERALLY without the OR-verified clause.
    """

    labels: list[str] | None = None
    site_ids: list[str] | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    min_confidence: float | None = Field(None, ge=0.0, le=1.0)
    max_confidence: float | None = Field(None, ge=0.0, le=1.0)
    min_label_confidence: float | None = Field(None, ge=0.0, le=1.0)
    max_label_confidence: float | None = Field(None, ge=0.0, le=1.0)
    project_floor: float | None = Field(None, ge=0.0, le=1.0)
    category: str | None = None
    verified: bool | None = None


_DEFAULT_MAX_DETECTIONS = 20000


class SortRequest(BaseModel):
    """Request body for the Observations sort endpoint.

    ``max_detections`` is the per-user memory budget for one sort,
    set in the Observations view-options popover (localStorage on
    the client). Defaults to 20000 when the client omits it.
    """

    filters: ObservationFilters = Field(default_factory=ObservationFilters)
    sort: ObservationSort = "similarity"
    max_detections: int = Field(
        default=_DEFAULT_MAX_DETECTIONS, ge=1000, le=50000
    )


class SearchRequest(BaseModel):
    """Request body for the Observations search endpoint.

    ``max_detections`` mirrors the sort endpoint — same per-user cap
    drives the underlying neighbour query.
    """

    anchor_detection_id: str
    filters: ObservationFilters = Field(default_factory=ObservationFilters)
    limit: int = Field(100, ge=1, le=500)
    threshold: float = Field(0.0, ge=-1.0, le=1.0)
    max_detections: int = Field(
        default=_DEFAULT_MAX_DETECTIONS, ge=1000, le=50000
    )


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
    label: str | None
    label_confidence: float | None
    display_name: str | None = None
    confidence: float
    category: str
    verified: bool
    classification_method: str | None
    distance_to_centroid: float | None = None
    similarity: float | None = None
    neighbor_agreement: float | None = None
    neighbor_top_label: str | None = None
    neighbor_top_display_name: str | None = None
    site_name: str | None = None
    deployment_id: str | None = None
    captured_at_local: datetime | None = None
    crop_url: str
    crop_bbox: CropBbox | None = None
    # Video detections carry the frame index they came from; image
    # detections expose None.
    frame_number: int | None = None

    @field_serializer("captured_at_local")
    def _serialize_captured_at_local(self, value: datetime | None) -> str | None:
        return serialize_local_datetime(value)


class SortResponse(BaseModel):
    """Response for the Observations sort endpoint."""

    detections: list[DetectionSummary]
    total_detections: int


class SearchResponse(BaseModel):
    """Response for the Observations search endpoint."""

    anchor: DetectionSummary
    results: list[DetectionSummary]
    total_results: int
    threshold_applied: float


class ObservationStatsResponse(BaseModel):
    """Embedding coverage stats."""

    total_detections: int
    # Verification progress pill, counted over the population the
    # Observations view shows (embedded + threshold-or-verified, i.e.
    # the similarity-sort population). `verified_detections /
    # reviewable_detections` is the % bar; counting the whole dataset
    # rather than the user's active filters, but matching what's on
    # screen so it can't read 95% while every visible item is verified.
    reviewable_detections: int
    verified_detections: int
    embedded_detections: int
    missing_embeddings: int
    embedding_model_id: str | None
    embedding_dimension: int | None
