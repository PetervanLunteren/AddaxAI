"""
Sequence schemas for API requests and responses.
"""

from datetime import datetime

from pydantic import BaseModel, field_serializer

from app.api.schemas.file import FileWithDetections
from app.utils.datetime_serialization import serialize_local_datetime


class MaxNFrame(BaseModel):
    """Where and when a species' MaxN was counted, for the Counts modal."""

    file_id: str
    label: str | None = None
    label_taxonomy_id: str | None = None
    max_n: int
    # Human-authoritative count (`human_count` if set, else `max_n`).
    effective_count: int
    # The frame of the file the MaxN was counted on, and the wall-clock
    # time it stands for; both null for images and rows from before the
    # frame was recorded. Time is camera local, serialized like every
    # other observational datetime.
    max_n_frame_number: int | None = None
    max_n_time: datetime | None = None

    @field_serializer("max_n_time")
    def _serialize_max_n_time(self, value: datetime | None) -> str | None:
        return serialize_local_datetime(value)


class SequenceSummary(BaseModel):
    """Sequence summary for browse card display."""

    id: str
    deployment_id: str
    # Null when the sequence's files have no extractable capture time. Missing
    # capture time is tolerated, not fatal (see DEVELOPERS.md datetime conventions);
    # these sequences drop out of time-based features but must still serialize.
    sequence_start_local: datetime | None
    sequence_end_local: datetime | None
    file_count: int
    thumbnail_file_id: str | None
    # Up to four file IDs used for the sequence-card collage. First slots
    # come from `max_n_frames` (one per dominant species), remaining
    # slots are padded by max detection confidence. Empty for sequences
    # with no files.
    collage_file_ids: list[str] = []
    max_n_frames: list[MaxNFrame]
    site_name: str | None
    labels: list[str]
    scientific_labels: dict[str, str] | None = None
    common_labels: dict[str, str] | None = None
    observation_type: str
    observation_types: list[str]
    image_count: int
    frame_count: int
    video_count: int
    verified_count: int
    total_count: int
    verified_maxn_count: int
    total_maxn_count: int
    # Human confirmation of the sequence's species and counts, stored on
    # Sequence.confirmed (the "Confirm" action on the Observations page). The
    # verified_count / verified_maxn_count fields above are file-level
    # secondary detail.
    is_confirmed: bool
    # Aggregated file-level state for the card corner cluster.
    any_file_flagged: bool
    any_file_favorited: bool

    # sequence_start_local / sequence_end_local are naive wall-clock times in
    # the project's local camera timezone (see DEVELOPERS.md).
    @field_serializer("sequence_start_local", "sequence_end_local")
    def _serialize_sequence_local(self, value: datetime | None) -> str | None:
        return serialize_local_datetime(value)


class SequenceObservationItem(BaseModel):
    """One cohort row of a sequence's count list (the Counts-page count
    editor). `effective_count` is the human-authoritative count
    (`human_count` if set, else the AI-derived `max_n`). A human-only row
    (a species the AI missed, or a cohort split off by hand) has max_n=0.
    The demographics are None until a person sets them."""

    id: str
    category: str
    label: str | None = None
    label_taxonomy_id: str | None = None
    common_name: str | None = None
    scientific_name: str | None = None
    max_n: int
    effective_count: int
    sex: str | None
    life_stage: str | None
    behavior: str | None
    # Where and when the AI's MaxN was counted: the file, its frame (null
    # for a photo) and the wall-clock time; and when the species was
    # first seen in the sequence. All null on a human-only row, and the
    # times null when the files carry no capture time.
    max_n_file_id: str | None = None
    max_n_frame_number: int | None = None
    max_n_time: datetime | None = None
    first_arrival_time: datetime | None = None

    @field_serializer("max_n_time", "first_arrival_time")
    def _serialize_times(self, value: datetime | None) -> str | None:
        return serialize_local_datetime(value)


class SequenceWithFiles(BaseModel):
    """Sequence with all files and their detections."""

    id: str
    deployment_id: str
    # Null when the sequence's files have no extractable capture time. Missing
    # capture time is tolerated, not fatal (see DEVELOPERS.md datetime conventions);
    # these sequences drop out of time-based features but must still serialize.
    sequence_start_local: datetime | None
    sequence_end_local: datetime | None
    file_count: int
    max_n_frames: list[MaxNFrame]
    # Human confirmation of the species and counts ("Confirm" action).
    confirmed: bool = False
    # Per-species count list (AI + human-only), highest count first.
    observations: list[SequenceObservationItem] = []
    # A person's free text about the visit (Sequence.notes).
    notes: str | None
    created_at_utc: datetime
    site_name: str | None = None
    files: list[FileWithDetections]

    @field_serializer("sequence_start_local", "sequence_end_local")
    def _serialize_sequence_local(self, value: datetime | None) -> str | None:
        return serialize_local_datetime(value)

    class Config:
        from_attributes = True


class GenerateSequencesRequest(BaseModel):
    """Request body for sequence generation."""

    project_id: str


class GenerateSequencesResponse(BaseModel):
    """Response for sequence generation."""

    sequence_count: int
    message: str


class AdjacentSequencesResponse(BaseModel):
    """Response for adjacent sequence navigation."""

    previous_id: str | None
    next_id: str | None
    next_unconfirmed_id: str | None
    current_index: int
    total_count: int


class DateRange(BaseModel):
    """Date range with min and max."""

    min: datetime
    max: datetime


class SequenceVerificationStats(BaseModel):
    """What a reprocess must not lose, across the filtered sequences: the
    confirmed sequences and the verified cards on them.

    The Counts page and the dashboard read the sequence half from
    `/api/sequences/count`, which does not walk the detections.
    """

    sequences_confirmed: int
    sequences_total: int
    total_detections: int
    verified_detections: int


class LabelTreeNode(BaseModel):
    """A node in the label filter tree. Shape only: the counts arrive
    separately (`LabelCounts`) and the front end puts them on the leaves."""

    id: str
    name: str
    level: int
    children: list["LabelTreeNode"]
    selected: bool
    annotation: str | None = None
    # Leaves under a parent; None on a leaf.
    child_count: int | None = None


class LabelTree(BaseModel):
    """The label filter tree: every label the project has, nested by rank."""

    tree: list[LabelTreeNode]
    all_leaf_ids: list[str]


class SequenceFilterOptions(BaseModel):
    """Everything about the labels a project has, in one pass over its
    boxes. Changes only when labels change, so the pages share it."""

    labels: list[str]
    date_range: DateRange | None
    scientific_labels: dict[str, str] | None = None
    common_labels: dict[str, str] | None = None
    # Lowest classification score among the labels' boxes; the filter
    # bars' clamp for the classification range slider. None when nothing
    # is classified yet.
    min_label_confidence: float | None = None
    # Colour per label, keyed by taxonomy id and lowercased name
    # (`crud/label_colors.py`).
    label_colors: dict[str, str]
    # None when the project has no labels: the filter shows a flat list.
    label_tree: LabelTree | None


class LabelCounts(BaseModel):
    """How many of each label a page would show under its filters, keyed
    by taxonomy id. A label that is missing counts 0."""

    counts: dict[str, int]
