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

    ``verified`` and ``job_id`` are what the drawing rule needs, and
    both are required rather than defaulted so a source row missing
    either fails loudly here instead of arriving as a plausible value.
    ``verified`` carries the threshold override (a box a human
    confirmed draws at any confidence); ``job_id`` is null exactly for
    a box a person drew, which is the only exact marker of one. The
    frontend has declared both on this payload for a long time while
    this schema dropped them, so the type checker could not see the
    hole: reading either gave ``undefined``, which reads as "not
    verified" and as "not human-drawn".
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
    common_name: str | None = None
    scientific_name: str | None = None
    label_taxonomy_id: str | None = None
    classification_method: str | None = None
    frame_number: int | None = None
    verified: bool
    job_id: str | None

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
    captured_at_local: datetime | None
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
    def _serialize_captured_at_local(self, value: datetime | None) -> str | None:
        return serialize_local_datetime(value) if value else None  # type: ignore[return-value]

    class Config:
        from_attributes = True


class FileWithDetections(FileResponse):
    """File with detections response schema."""

    detections: list[DetectionResponse]
    # The camera (subfolder name) this file came from, for files of a
    # paired-cameras deployment. None otherwise, and for root-level files.
    camera: str | None = None

    class Config:
        from_attributes = True


class FileUpdate(BaseModel):
    """Schema for updating a file (verification, notes, favorited, flagged)."""

    verified: bool | None = None
    notes: str | None = None
    favorited: bool | None = None
    flagged: bool | None = None


class FilmstripFrame(BaseModel):
    """One frame of a video's on-demand filmstrip preview."""

    frame_number: int
    # Seconds into the clip; null when the video has no known frame rate.
    time_seconds: float | None
    # Inline JPEG data URI ("data:image/jpeg;base64,...").
    image: str


class FilmstripResponse(BaseModel):
    """Evenly-spaced low-res frames for the counts-modal video gallery."""

    frames: list[FilmstripFrame]
