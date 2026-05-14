"""
File schemas for API requests and responses.
"""

from datetime import datetime

from pydantic import BaseModel, field_serializer

from app.utils.datetime_serialization import serialize_local_datetime


class DetectionResponse(BaseModel):
    """Detection response schema.

    bbox fields are nullable because event-level observations (species
    seen in a video clip without a frame-anchored ROI) have no spatial
    annotation. All four fields are null-together for those rows; AI
    and user-drawn detections set all four.
    """

    id: str
    category: str
    confidence: float
    bbox_x: float | None
    bbox_y: float | None
    bbox_width: float | None
    bbox_height: float | None
    label: str | None
    label_confidence: float | None
    display_name: str | None = None
    label_taxonomy_id: str | None = None
    classification_method: str | None = None
    frame_number: int | None = None

    class Config:
        from_attributes = True


class FileResponse(BaseModel):
    """File response schema."""

    id: str
    deployment_id: str
    file_path: str
    file_type: str
    file_format: str
    size_bytes: int | None
    width_px: int | None
    height_px: int | None
    captured_at_local: datetime
    created_at_utc: datetime
    best_frame_number: int | None = None
    best_frame_path: str | None = None
    frame_rate: float | None = None
    observation_type: str = "unclassified"
    verified: bool = False
    verified_at_utc: datetime | None = None
    notes: str | None = None
    favorited: bool = False
    flagged: bool = False
    flagged_at_utc: datetime | None = None
    source_video_id: str | None = None
    source_frame_number: int | None = None

    # captured_at_local is naive wall-clock time at the camera. Rendered
    # with the offset that applies on the file's local date, read from
    # the active project timezone in the request context. See
    # DEVELOPERS.md "Datetime conventions".
    @field_serializer("captured_at_local")
    def _serialize_captured_at_local(self, value: datetime) -> str:
        return serialize_local_datetime(value)  # type: ignore[return-value]

    class Config:
        from_attributes = True


class FileWithDetections(FileResponse):
    """File with detections response schema."""

    detections: list[DetectionResponse]

    class Config:
        from_attributes = True


class FileUpdate(BaseModel):
    """Schema for updating a file (verification, notes, favorited, flagged)."""

    verified: bool | None = None
    notes: str | None = None
    favorited: bool | None = None
    flagged: bool | None = None


class FileSummaryDetection(BaseModel):
    """Minimal detection payload for Files-tab grid overlays.

    bbox fields nullable for event-level observations — no bbox to
    overlay on a tile, but the row still belongs to the file. """

    id: str
    category: str
    confidence: float
    bbox_x: float | None
    bbox_y: float | None
    bbox_width: float | None
    bbox_height: float | None
    label: str | None
    label_taxonomy_id: str | None = None
    # Video detections carry their frame index; image detections have None.
    # Used by the verify grid / filmstrip to filter to the best-frame
    # detections and to enumerate distinct frames per video.
    frame_number: int | None = None

    class Config:
        from_attributes = True


class FileSummary(BaseModel):
    """File summary for the Files verify tab card grid."""

    id: str
    deployment_id: str
    file_type: str
    file_format: str | None
    width_px: int | None
    height_px: int | None
    captured_at_local: datetime
    site_id: str | None
    site_name: str | None
    observation_type: str
    observation_types: list[str]
    labels: list[str]
    display_labels: dict[str, str]
    verified: bool
    favorited: bool
    flagged: bool
    source_video_id: str | None
    # Video rows expose `best_frame_number` so the grid overlay can
    # filter detections to that one frame (the thumbnail's frame).
    # Null for images and for any video whose best frame failed to
    # resolve.
    best_frame_number: int | None = None
    detections: list[FileSummaryDetection]

    @field_serializer("captured_at_local")
    def _serialize_captured_at_local(self, value: datetime) -> str:
        return serialize_local_datetime(value)  # type: ignore[return-value]


class FileVerificationStats(BaseModel):
    """Aggregate file verification stats across the filtered set."""

    total_files: int
    verified_files: int


class AdjacentFilesResponse(BaseModel):
    """Adjacent file IDs for file-to-file navigation in the Files tab."""

    previous_id: str | None
    next_id: str | None
    next_unverified_id: str | None
    current_index: int
    total_count: int
