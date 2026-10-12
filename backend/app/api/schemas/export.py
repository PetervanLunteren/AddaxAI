"""Schemas for the project export endpoints that answer with JSON rather
than a file: the recognition-file export, which writes into the
deployment folders and reports what it did per deployment, and the
image copies export, which shares its media options with the folder-run
Save step."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from app.core.confidence import DEFAULT_COUNTING_THRESHOLD
from app.ml.postprocessing_outputs.separate_folders import SeparateGroupBy


class RecognitionJsonWritten(BaseModel):
    deployment_id: str
    folder_path: str
    output_path: str
    image_count: int


class RecognitionJsonSkipped(BaseModel):
    deployment_id: str
    folder_path: str | None
    reason: str


class RecognitionJsonExportResult(BaseModel):
    """One entry per deployment in scope, written or skipped. A folder
    that is unmounted or read-only is a skipped entry with its reason,
    not a failed request: the other deployments' files are still worth
    writing, and a project's folders live on several drives."""

    written: list[RecognitionJsonWritten]
    skipped: list[RecognitionJsonSkipped]


class MediaOptions(BaseModel):
    """The options behind the media copies, shared by the folder-run Save
    step (``SaveOutputsRequest``, ``OutputPreviewRequest``) and the
    project image copies export. One definition, so the two surfaces
    cannot offer different knobs, and the field names are the keys the
    save-outputs worker reads off the job payload."""

    # Media-output confidence: detections below it (unless verified) are
    # left out of the separated copies, drawn boxes, blurs, and EXIF
    # tags. Data exports (CSV / XLSX / recognition JSON) ignore it: they
    # are always the complete record of the run.
    media_threshold: float = Field(
        DEFAULT_COUNTING_THRESHOLD, ge=0.0, le=1.0
    )
    # How media copies are grouped at the output root. ``taxonomic``
    # nests Class/Order/Family/Genus/species; ``flat`` is one folder
    # per species label; ``none`` copies everything flat at the root.
    separate_group_by: SeparateGroupBy = "flat"
    # Keep a burst together: every file in an event lands in one folder
    # (the event's main species) instead of being filed per file.
    group_events: bool = True
    # Flip the layering to ``<source subfolder>/<species>/`` instead of
    # ``<species>/<source subfolder>/``. Keeps the user's original folders
    # on top and species inside them (the layout camtrapR expects). No
    # effect with ``separate_group_by="none"`` or a flat source folder.
    separate_species_last: bool = False
    # Copy empty captures (no animal / person / vehicle) too. Off by
    # default so the media copies aren't padded with blank captures.
    include_empty: bool = False
    # Label identifiers to exclude from every output. Each entry is
    # either a LabelTaxonomy.id UUID (for taxonomy-mapped labels) or a
    # raw Detection.label string (for unmapped labels under the tree's
    # "Other" branch). Applied as a file-level filter for separate /
    # annotated_copies and as a row-level filter for CSV / XLSX /
    # recognition JSON.
    excluded_label_ids: list[str] = Field(default_factory=list)
    # Draw detection bounding boxes + pill labels on the annotated
    # copies.
    draw_bboxes: bool = False
    # Blur person / vehicle detections on the annotated copies — for
    # sharing datasets without identifying bystanders or vehicles.
    anonymise: bool = False
    # Which species name to burn into the visualised images: the common
    # name or the scientific name. Mirrors the UI display preference so
    # the saved images match what the user sees. EXIF metadata always
    # carries both names regardless of this choice.
    name_mode: Literal["common", "scientific"] = "common"


class MediaExportOptions(MediaOptions):
    """Body of ``POST .../export/media/preview``: the media options plus
    which files. Mirrored by ``MediaScope`` on the worker side."""

    # None = every deployment of the project. The Export page's scope
    # picker sends ids only; an empty list is refused by the prepare
    # endpoint (see there).
    deployment_ids: list[str] | None = None
    favorited_only: bool = False
    verified_only: bool = False
    # Inclusive on whole days. Files without a capture time drop out as
    # soon as either is set.
    date_from: date | None = None
    date_to: date | None = None


class MediaExportRequest(MediaExportOptions):
    """Body of ``POST .../export/media/prepare``. Copies go into the
    ``addaxai-media`` subfolder of ``output_dir``; a previous export's
    subfolder there is replaced."""

    output_dir: str = Field(..., min_length=1)


class OutputPreviewResponse(BaseModel):
    """Aggregate counts the Save step uses to render a live folder
    preview. Computed deterministically from the project's DB state;
    the numbers are exact, not estimates, because the placement
    rules are fixed.
    """

    total_files: int
    image_count: int
    video_count: int
    total_bytes: int
    files_with_known_size: int
    dropped_by_filter: int
    in_scope_files: int
    in_scope_image_count: int
    in_scope_video_count: int
    in_scope_bytes: int
    by_media_tree: dict[str, int]
    root_files: list[str]
