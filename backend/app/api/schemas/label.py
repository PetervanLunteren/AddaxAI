"""
Pydantic schemas for the Labels API.

Request/response models for the Labels verify tab: sort (greedy
nearest-neighbor chain), search (find similar), and embedding coverage
stats. The underlying technique is cosine similarity on DINOv2 crop
embeddings; the schemas here are named after the tab (labels),
not the algorithm.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_serializer

from app.utils.datetime_serialization import serialize_local_datetime

LabelSort = Literal[
    "similarity",
    # Group detections by their event, events newest first. Grouping is
    # the point, so the grid auto-shows event dividers in this mode.
    "events",
    # Cohort-grouped review mode. Reachable only via the toolbar's
    # review pill, never selectable from the sort dropdown.
    "suggestions",
]


class LabelFilters(BaseModel):
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
    # File-level triage marks, the Counts filters brought to Labels.
    # Omitted means "all"; the frontend never sends the "all" value.
    flagged: Literal["flagged", "not_flagged"] | None = None
    favorited: Literal["favorited", "not_favorited"] | None = None



class SortRequest(BaseModel):
    """Request body for the Labels sort endpoint.

    The result is capped to a fixed memory budget (the sort subprocess
    loads the newest slice and reports the uncapped total); there is no
    client-tunable limit.
    """

    filters: LabelFilters = Field(default_factory=LabelFilters)
    sort: LabelSort = "similarity"


class SearchRequest(BaseModel):
    """Request body for the Labels search endpoint."""

    anchor_detection_id: str
    filters: LabelFilters = Field(default_factory=LabelFilters)
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
    label: str | None
    # Taxonomy row id matching `label`. Carried so cohort dividers in
    # the suggestions sort mode can navigate to the existing label
    # filter (keyed on taxonomy id).
    label_taxonomy_id: str | None = None
    label_confidence: float | None
    common_name: str | None = None
    scientific_name: str | None = None
    confidence: float
    category: str
    verified: bool
    classification_method: str | None
    distance_to_centroid: float | None = None
    similarity: float | None = None
    neighbor_agreement: float | None = None
    neighbor_top_label: str | None = None
    neighbor_top_common_name: str | None = None
    neighbor_top_scientific_name: str | None = None
    site_name: str | None = None
    deployment_id: str | None = None
    captured_at_local: datetime | None = None
    # Event this detection's file belongs to. Drives the "By event" sort,
    # the event dividers, and the detail modal's sequence strip. Null when
    # event clustering has not run for the deployment.
    event_id: str | None = None
    # Start time of `event_id`, naive camera-local; the event divider's
    # header label. Serialized like captured_at_local.
    event_start_local: datetime | None = None
    crop_url: str
    crop_bbox: CropBbox | None = None
    # Video detections carry the frame index they came from; image
    # detections expose None.
    frame_number: int | None = None
    # File-level triage marks, for the card's corner badge cluster.
    # Defaulted so older worker rows (mid-deploy) stay parseable.
    file_flagged: bool = False
    file_favorited: bool = False

    @field_serializer("captured_at_local", "event_start_local")
    def _serialize_local_datetimes(self, value: datetime | None) -> str | None:
        return serialize_local_datetime(value)


class SortResponse(BaseModel):
    """Response for the Labels sort endpoint."""

    detections: list[DetectionSummary]
    total_detections: int


class SearchResponse(BaseModel):
    """Response for the Labels search endpoint."""

    anchor: DetectionSummary
    results: list[DetectionSummary]
    total_results: int
    threshold_applied: float


class CohortItem(BaseModel):
    """One row of the promotion review panel.

    A cohort groups unverified detections that currently carry
    `current_label` but whose 10 nearest embedding neighbours mostly
    look like `suggested_label` (which must be a strict taxonomic
    descendant of `current_label`, e.g. genus → species). One click on
    "Relabel all" promotes the whole group.
    """

    current_label: str | None
    # Taxonomy row id for `current_label`. Used by the panel's
    # "Review crops" navigation, which targets the existing label
    # filter (keyed on taxonomy id). May be null when the current label
    # has no taxonomy row.
    current_label_taxonomy_id: str | None
    current_common_name: str | None
    current_scientific_name: str | None
    suggested_label: str
    suggested_common_name: str | None
    suggested_scientific_name: str | None
    # Detection category ("animal", "person", "vehicle"). Carried so
    # the relabel call can keep the category fixed even when the model
    # category-output column drifts from the label-taxonomy category.
    category: str | None
    count: int
    # Sorted by ascending neighbour agreement so the thumbnail strip
    # leads with the crops that disagree most strongly with their
    # current label.
    detection_ids: list[str]


class CohortsResponse(BaseModel):
    """Response payload for the cohorts endpoint."""

    cohorts: list[CohortItem]


#: The Labels page's verification filter. A Literal so a typo is a 422
#: rather than a silently unfiltered 200, which is what a bare ``str``
#: gave: every other parameter on that endpoint already validates.
LabelsVerification = Literal["all", "verified", "unverified"]

#: Same three values the Counts page's empty filter uses: only the files
#: where nothing passes, only the files where something does, or both.
EmptyFilter = Literal["all", "show_only", "hide"]

LabelsFilesSort = Literal[
    # Default. Groups one camera's photos together, because file_path is
    # absolute and starts with the deployment folder. Reviewing files
    # means scanning the same scene repeatedly, and capture-time order
    # interleaves cameras.
    "path",
    "newest",
    "oldest",
    "random",
    # Bursts kept together: events newest first, each event's files in
    # shooting order. The grid shows divider rows in this mode, like the
    # Detections tab's event sort.
    "events",
]


class LabelsFileItem(BaseModel):
    """One file on the Files tab: enough to draw a tile, nothing more.

    No detections are carried. The tile fetches the file detail for its
    boxes, the same row the viewer opens, so the two share one cache
    entry.
    """

    id: str
    deployment_id: str
    file_path: str
    file_type: str
    captured_at_local: datetime | None
    verified: bool
    # Pixel size, so the grid can shape its tiles to the page's majority
    # aspect ratio instead of forcing 4:3 on a 16:9 camera.
    width_px: int | None
    height_px: int | None
    # The file's event, filled by the router only for sort=events, where
    # the grid draws divider rows from it. None in every other sort.
    event_id: str | None = None

    @field_serializer("captured_at_local")
    def _serialize_local_datetimes(self, value: datetime | None) -> str | None:
        return serialize_local_datetime(value)

    class Config:
        from_attributes = True


class LabelsFilesResponse(BaseModel):
    """Response for the Files tab list.

    `floor` is the confidence the empty filter was judged at, echoed so
    the page can name the number the user is actually looking at rather
    than re-deriving it in the frontend.

    `find_index` answers the `find` query param: the named file's
    0-based position in the full ordering, or None when it does not
    match the filters. Absent unless asked for.
    """

    total: int
    floor: float
    items: list[LabelsFileItem]
    find_index: int | None = None


class LabelsProgress(BaseModel):
    """How far through the Labels page the user is, counted in labels.

    A label is one call a person has to make: a detection above the
    threshold (one card in Detections), or a file where nothing passes
    (one "nothing here" call). The two do not overlap and together they
    cover the project, so one bar covers the page.

    `files` / `files_verified` count every file in scope and how many
    are signed off, for the Files tab's chip. Files with boxes are in
    both units on purpose: the Files tab lists them too.
    """

    total_labels: int
    verified_labels: int
    # The two halves, so each tab can say how much is waiting in the
    # other one without a second request.
    crop_labels: int
    crop_labels_verified: int
    empty_labels: int
    empty_labels_verified: int
    files: int
    files_verified: int


class LabelStatsResponse(BaseModel):
    """Embedding coverage stats."""

    total_detections: int
    verified_detections: int
    embedded_detections: int
    missing_embeddings: int
    embedding_model_id: str | None
    embedding_dimension: int | None
