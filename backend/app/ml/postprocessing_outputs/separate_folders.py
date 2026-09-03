"""Reorganise a project's media files into ``<output_root>/<label>/``.

The legacy-AddaxAI mental model for postprocessing: browse the run's
output as ``output_root/dog/``, ``output_root/leopard/``,
``output_root/blank/``, …, in the file manager. This module re-creates
that experience on top of the modern DB pipeline.

Single-destination rule:

Every file lands in exactly one folder, named after its **strongest
passing detection**: that detection's species if it has one, otherwise
its detector category. An image with ``dog`` at 0.95 + ``wolf`` at 0.80
goes to ``output_root/dog/`` only, never both. An image whose best box
is a person goes to ``output_root/person/`` even if a weaker box
carries a species. The output stays a clean, predictable mirror of the
run with no duplication, images and videos alike. Multi-species
findability is intentionally out of scope here: every species is in the
CSV / recognition JSON, and projects mode is the place for richer
querying.

The category is the detector's own, never translated, so a detector
emitting ``shark`` / ``fish`` / ``turtle`` writes those folder names
with no change here. Only the Camtrap DP export translates, because
its ``observationType`` field has a fixed controlled vocabulary.

Original folder structure (suffix placement):

The user's source subfolders are preserved *under* the species folder,
so ``<source>/cam01/img.jpg`` of a dog lands at
``output_root/dog/cam01/img.jpg``. Species stays the top browsing axis
(the point of the feature) while the original structure, and the reason
the user organised it, is kept and collisions are avoided. Under
``group_by="none"`` there is no species folder, so the output simply
mirrors the source tree (``output_root/cam01/img.jpg``).

Event grouping (``group_events``):

With ``group_events`` on, every file in an event (a burst / sequence)
is placed in one shared folder, the species of the event's most
confident detection, instead of deciding per file. This keeps a
sequence together even when a stray frame's own top species differs.

Copies only, never a move. Legacy AddaxAI offered a move, and there
used to be an unreachable ``move`` mode here. It is gone on purpose,
not parked: the worker wipes ``addaxai-media`` before every re-save,
using the scan-skip marker as its proof of ownership, so a move would
hand the user's only copy to that wipe. The output folder also
defaults to the source folder itself, and the source tree is what the
recognition JSON's relative paths, the detection checkpoint and the
video-frame cache all hang off. The app promises in the FAQ and on the
Save step that originals are never moved. A user who wants the source
gone deletes it themselves, once, after checking the copies.

Grouping:

- ``taxonomic`` (default): a nested chain
  ``Class/Order/Family/Genus/species/`` derived from each label's
  ``LabelTaxonomy`` entry. Labels with no taxonomy mapping fall to
  ``Other/<label>/``.
- ``flat``: one folder per species label at the root (``dog/``,
  ``leopard/`` …). Best when only a few species are in play.

A file with no species on its deciding detection routes to a single
flat folder named after that detection's category (``person/``,
``vehicle/``, ``blank/``, ``animal/`` for an unclassified animal), in
both modes.

Collision handling: ``output_root/<label>/IMG_001.jpg`` already
exists → append ``_2``, ``_3``, … per destination folder until the
name is unique. Original files are never overwritten.

Videos: a video is copied as the file it is, the whole container under
its own name, like an image. That is what legacy AddaxAI did and what
three users asked for in the first week of v7 (2026-09): they sort
their media into species folders to keep the animal clips apart from
the junk they delete, and a still is no use for that. Until then a
video was written as its best-frame JPEG only, for the disk saving and
so boxes and blur had a picture to land on. Boxes still do: with
``draw_bboxes`` on, ``annotated_copies`` writes the annotated best
frame as ``<video_stem>_still.jpg`` beside the copied video (the
``_still`` suffix keeps it clear of a same-named photo, and the name is
allocated here with every other name so it cannot collide either). The
still shows the best frame, and the folder is decided by that same
frame, so the picture beside the video shows why it is there, except
for a clip filed by a verified box on another frame, which gets no
still because its best frame has nothing to draw.

Blur is the one exception, through ``videos_as_stills``: a blurred
still next to the unblurred video it came from is no anonymisation, so
with blur on a video is written as its blurred still only. The worker
passes ``videos_as_stills=anonymise``; nothing else sets it. In that
mode a video with no best frame on disk is skipped, since there is no
picture to blur; in normal mode such a video is still copied, and it
reads ``blank`` unless a verified box speaks for it (no visible
surface, see "What a file is about" in DEVELOPERS.md).

No EXIF tags are written into video containers (``is_image_path``),
same as legacy.

Each successful placement is recorded on the shared ``OutputContext``
so downstream modules (``annotated_copies``, the CSV / XLSX writers)
write into / reference the same path the separated file landed at.
"""

from __future__ import annotations

import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import __version__ as APP_VERSION
from app.core.logging_config import get_logger
from app.ml.detection_visibility import on_visible_frame, visible_detections
from app.ml.label_exclusion import is_a_real_detection, threshold_or_verified
from app.ml.observation_type import derive_observation_type
from app.models import Deployment, Detection, File, LabelTaxonomy, Project
from app.models.event import event_files

from ._exif_writer import ExifBatch, build_tag_set, is_image_path
from ._label_filter import (
    file_is_dropped_by_filter,
    strongest_label_for_file,
)
from ._output_context import OutputContext

logger = get_logger(__name__)


# ``taxonomic`` nests Class > Order > Family > Genus > species; labels
# with no taxonomy mapping fall to ``Other/<label>/``. ``flat`` collapses
# to one folder per species label at the root.
SeparateGroupBy = Literal["taxonomic", "flat", "none"]
# Common vs scientific naming for the species leaf folder, mirroring the
# UI's global species-name toggle. Only the leaf segment follows it; the
# Class/Order/Family/Genus ancestors are always the Latin taxon ranks.
NameMode = Literal["common", "scientific"]
# Bucket name for labels with no taxonomy mapping under ``taxonomic``.
UNRANKED_FOLDER = "Other"
# Order of taxonomic ranks. Path construction walks these in order and
# stops at the label's own rank; missing intermediate ancestors pad
# with ``UNRANKED_FOLDER`` so the depth stays consistent.
_TAXONOMIC_RANK_ORDER: tuple[str, ...] = (
    "class",
    "order",
    "family",
    "genus",
    "species",
    "variant",
)


@dataclass
class SeparateFoldersResult:
    """Summary of a separate-into-folders run.

    Every file lands in exactly one folder, so placement counts equal
    source-file counts. ``skipped_excluded`` counts files whose every
    passing identified detection was in the exclusion set, so the file
    disappears from the outputs entirely.
    """

    copied_count: int = 0
    skipped_missing_source: int = 0
    skipped_excluded: int = 0
    renamed_count: int = 0
    by_label: Counter = field(default_factory=Counter)
    errors: list[str] = field(default_factory=list)

    @property
    def written_count(self) -> int:
        """Total placements. Kept as its own name because the worker's
        result payload and the completion dialog read it."""
        return self.copied_count

    def to_dict(self) -> dict:
        return {
            "copied_count": self.copied_count,
            "written_count": self.written_count,
            "skipped_missing_source": self.skipped_missing_source,
            "skipped_excluded": self.skipped_excluded,
            "renamed_count": self.renamed_count,
            "by_label": dict(self.by_label),
            "errors": list(self.errors),
        }


# A file with no species is named by its detector category: `person/`,
# `vehicle/`, `blank/`, and `shark/` or `fish/` from a detector that
# emits those. Species-level taxonomy does not apply to a detector
# category, so these sit flat at the root of the output.
#
# There is no lookup table here any more. The category IS the folder
# name, slugged. The old table also renamed `human` to `person`, which
# is now unnecessary because `person` is what the detector called it and
# what `observation_type` carries.


def _build_taxonomy_map(
    db: Session, project: Project
) -> dict[str, LabelTaxonomy]:
    """Preload the ``LabelTaxonomy`` rows for the project's classifier
    so per-file plans resolve the taxonomic chain without a query per
    file. Empty dict when the project has no classifier — every label
    then falls back to ``Other/<label>``.
    """
    model_id = project.classification_model_id
    if not model_id:
        return {}
    rows = db.execute(
        select(LabelTaxonomy).where(
            LabelTaxonomy.classification_model_id == model_id
        )
    ).scalars().all()
    return {row.name: row for row in rows}


def build_event_primary_labels(
    db: Session,
    project_id: str,
    threshold: float,
    excluded_label_ids: frozenset[str] | None = None,
) -> dict[str, str]:
    """Map each animal file in an event to that event's primary species.

    Primary-species rule, in order:

    1. If the event has any human-verified labelled animal detections, the
       primary is the *most common* verified species (most verified
       detections; ties broken by highest detection confidence, then label).
       Human verification is the strongest signal, so a confirmed species
       owns the folder even when an unverified AI box scored higher.
    2. Otherwise it's the label of the single highest detection-confidence
       animal detection (the AI's best guess for the burst).

    **Step 1 counts, and that is deliberately not what a file does.**
    ``observation_type.strongest_passing_detection`` picks a *file's* subject
    by taking the single strongest box, with no tally at all. The difference is
    the grain, not drift, so do not unify them. A file is one photograph: one
    look at the animal, nothing to average. An event is dozens of looks at the
    same animal walking through, and per-frame classification is noisy enough
    that one raccoon read as raccoon / badger / badger / blank / opossum /
    raccoon / blank across seven consecutive frames. The mode cancels that;
    the strongest box would let one misread frame name the whole visit, on the
    folder the user actually sees. See DEVELOPERS.md "What a file is about".

    Only animal files in an event with at least one surviving species label
    appear in the map. A file in multiple events is assigned to the event of
    its own strongest detection (verified first, then confidence).

    Used by primary-only placement with ``group_events`` on to keep a whole
    burst in one folder. Mirrored by the Save-step preview (same function),
    so the two always agree.
    """
    rows = db.execute(
        select(
            event_files.c.event_id,
            Detection.file_id,
            Detection.label,
            Detection.label_taxonomy_id,
            Detection.confidence,
            Detection.verified,
        )
        .join(File, File.id == Detection.file_id)
        .join(event_files, event_files.c.file_id == File.id)
        .join(Deployment, Deployment.id == File.deployment_id)
        .where(Deployment.project_id == project_id)
        # Each video contributes only its best frame's boxes. This decides
        # the folder a burst is copied into, and a video is written to disk
        # as that one frame, so an off-frame box naming the folder files a
        # picture under a label the picture does not show. The mode across
        # the burst still denoises; it just counts boxes that exist as
        # pictures.
        .where(on_visible_frame())
        .where(threshold_or_verified(threshold))
        # A rejected box is not a species and must not vote: verified
        # first in the ordering below, one X press could otherwise name
        # a whole burst's folder "false detection". Same clause as
        # `passing_detections_for_file`, so the vote and the per-file
        # placement read the same boxes.
        .where(is_a_real_detection())
        # Verified first, then confidence: the first row per file is that
        # file's strongest detection, which decides its chosen event.
        .order_by(
            Detection.verified.desc(),
            Detection.confidence.desc(),
            event_files.c.event_id,
            Detection.file_id,
        )
    ).all()

    excluded = excluded_label_ids or frozenset()

    def _is_excluded(row) -> bool:
        if not excluded:
            return False
        if row.label_taxonomy_id and row.label_taxonomy_id in excluded:
            return True
        if row.label and row.label in excluded:
            return True
        return False

    rows = [r for r in rows if not _is_excluded(r)]

    # A file only votes when its own subject is a species, i.e. its
    # strongest passing detection carries a label. A file whose best box
    # is a person contributes nothing, even if a weaker animal box has a
    # species on it: that is the same rule `_folder_for_file` applies per
    # file, so a burst can never be named after something none of its
    # files would be named after on their own.
    #
    # This used to be `File.observation_type == "animal"` in SQL, which
    # read the stored column derived at the *project* threshold while
    # every other gate in this module re-derives at the *media*
    # threshold. Moving the Save-step slider therefore changed which
    # files were placed but not which ones got a vote. Deriving it here
    # from the same rows removes the second threshold entirely.
    subject_is_species: dict[str, bool] = {}
    for r in rows:
        # Rows are strongest-first, so the first row seen for a file is
        # that file's subject.
        subject_is_species.setdefault(r.file_id, r.label is not None)

    rows = [
        r for r in rows
        if r.label is not None and subject_is_species.get(r.file_id)
    ]

    # A file in multiple events belongs to the event of its own strongest
    # detection. Rows are ordered verified-then-confidence, so the first row
    # seen for a file wins.
    file_event: dict[str, str] = {}
    for r in rows:
        file_event.setdefault(r.file_id, r.event_id)

    # Primary species per event, following the rule in the docstring.
    per_event: dict[str, list] = defaultdict(list)
    for r in rows:
        per_event[r.event_id].append(r)

    event_primary: dict[str, str] = {}
    for event_id, ev_rows in per_event.items():
        verified = [r for r in ev_rows if r.verified]
        if verified:
            # Most common verified species; tie -> highest confidence, then
            # label for a deterministic result.
            agg: dict[str, dict] = {}
            for r in verified:
                key = r.label_taxonomy_id or r.label
                a = agg.setdefault(
                    key, {"count": 0, "conf": 0.0, "label": r.label}
                )
                a["count"] += 1
                a["conf"] = max(a["conf"], r.confidence)
            best = max(
                agg.values(),
                key=lambda a: (a["count"], a["conf"], a["label"]),
            )
            event_primary[event_id] = best["label"]
        else:
            # AI's best guess: the single highest-confidence detection.
            top = max(ev_rows, key=lambda r: (r.confidence, r.label))
            event_primary[event_id] = top.label

    return {
        file_id: event_primary[event_id]
        for file_id, event_id in file_event.items()
        if event_id in event_primary
    }


def _slug(name: str) -> str:
    """Filesystem-safe folder segment for a taxon / species name.

    Lowercases, ASCII-folds accents, and collapses every run of
    non-alphanumeric characters to a single underscore (so spaces, dots,
    and punctuation all go): "grey wolf" -> "grey_wolf", "C. lupus" ->
    "c_lupus", "Cervidae" -> "cervidae", "reeves' muntjac" ->
    "reeves_muntjac". Applied to every path segment so the output tree is
    consistent and survives case-insensitive filesystems (macOS, Windows)
    where "Cervidae" and "cervidae" would otherwise be the same directory.
    """
    ascii_name = (
        unicodedata.normalize("NFKD", name)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")
    return slug or "unknown"


def _leaf_name(
    label: str, taxon: LabelTaxonomy | None, name_mode: NameMode
) -> str:
    """The leaf folder name for a label, following the species-name toggle.

    Common mode (default) uses ``label`` (the common name for SpeciesNet,
    e.g. "grey wolf"). Scientific mode uses the precomputed
    ``scientific_name`` (the abbreviated binomial, e.g. "C. lupus"),
    falling back to ``label`` when there's no taxonomy row or no scientific
    name. Slugged to a filesystem-safe segment either way.
    """
    if name_mode == "scientific" and taxon is not None and taxon.scientific_name:
        return _slug(taxon.scientific_name)
    return _slug(label)


def _taxonomic_path_for_label(
    label: str,
    taxonomy_map: dict[str, LabelTaxonomy],
    name_mode: NameMode = "common",
) -> str:
    """Build the nested taxonomic path for one label.

    Returns a slash-separated path of the form ``<ancestors>/<leaf>``.
    Ancestors are the ``LabelTaxonomy`` columns above the label's own
    rank; missing intermediate ranks are padded with ``UNRANKED_FOLDER``
    so the on-disk depth is consistent across siblings. The leaf segment
    is the label rendered per the species-name toggle (see ``_leaf_name``),
    not ``taxon_species``. Labels with no taxonomy row fall back to
    ``Other/<leaf>``.
    """
    taxon = taxonomy_map.get(label)
    leaf = _leaf_name(label, taxon, name_mode)
    unranked = _slug(UNRANKED_FOLDER)
    if taxon is None:
        return f"{unranked}/{leaf}"

    label_level = (taxon.level or "").lower()
    if label_level not in _TAXONOMIC_RANK_ORDER:
        return leaf or unranked

    label_level_idx = _TAXONOMIC_RANK_ORDER.index(label_level)

    parts: list[str] = []
    for rank in _TAXONOMIC_RANK_ORDER[:label_level_idx]:
        value = getattr(taxon, f"taxon_{rank}", None)
        parts.append(_slug(value) if value else unranked)
    parts.append(leaf)
    return "/".join(parts)


def _folder_for_file(
    db: Session,
    file: File,
    obs_type: str,
    threshold: float,
    taxonomy_map: dict[str, LabelTaxonomy],
    group_by: SeparateGroupBy = "taxonomic",
    excluded_label_ids: frozenset[str] | None = None,
    name_mode: NameMode = "common",
    grouped_label: str | None = None,
) -> str:
    """The single destination folder for one file.

    A file goes to the species of its strongest passing detection, as a
    nested taxonomic path (``taxonomic``) or a single segment
    (``flat``). When that detection carries no species, the file is
    named by its detector category instead: ``person/``, ``vehicle/``,
    ``blank/``, or ``animal/`` for an unclassified animal, or whatever
    else the detector emits. ``grouped_label``, when set, overrides the
    per-file choice with the event's main species (``group_events``).
    The empty string means the output root (``group_by="none"``).

    ``obs_type`` is the file's effective observation type at the media
    threshold (see ``derive_observation_type``), passed in by the caller
    so the blank-skip and the routing here agree on one derivation. It
    is already the raw category of that same strongest detection, which
    is why it can be used verbatim as the fallback folder name.
    """
    if group_by == "none":
        return ""

    def _folder_for(label: str) -> str:
        return (
            _taxonomic_path_for_label(label, taxonomy_map, name_mode)
            if group_by == "taxonomic"
            else _leaf_name(label, taxonomy_map.get(label), name_mode)
        )

    # Event grouping: the whole burst shares the event's main species.
    if grouped_label is not None:
        return _folder_for(grouped_label)

    label = strongest_label_for_file(
        db, file, threshold, excluded_label_ids
    )
    if label is None:
        # No species on the deciding detection, so the category names
        # the folder. `obs_type` and the label come from the same
        # detection, so these two branches can never disagree.
        return _slug(obs_type)

    return _folder_for(label)


def _unique_destination(
    target_dir: Path,
    source_name: str,
    reserved: set[Path] | None = None,
) -> tuple[Path, bool]:
    """Return a path inside ``target_dir`` that doesn't already exist.

    Appends ``_2``, ``_3``, … before the suffix until the name is free.
    Returns the path plus a flag indicating whether a rename happened.

    ``reserved`` holds paths already handed out this run but not yet on
    disk. It's what keeps names unique in ``place_files=False`` (deferred)
    mode, where the physical write happens later in ``annotated_copies``
    and ``candidate.exists()`` alone would keep returning the same name.
    """
    stem = Path(source_name).stem
    suffix = Path(source_name).suffix

    def free(p: Path) -> bool:
        return not p.exists() and (reserved is None or p not in reserved)

    candidate = target_dir / source_name
    if free(candidate):
        return candidate, False
    counter = 2
    while True:
        candidate = target_dir / f"{stem}_{counter}{suffix}"
        if free(candidate):
            return candidate, True
        counter += 1


def video_still_name(video_path: str | Path) -> str:
    """Output filename for a video: its best frame as ``<stem>_still.jpg``.

    The ``_still`` suffix marks the file as a frame pulled from a video
    rather than an original photo, and keeps it from colliding with a
    same-named photo (some cameras shoot ``DSCF0100.jpg`` next to
    ``DSCF0100.mp4``).
    """
    return f"{Path(video_path).stem}_still.jpg"


def _media_source(
    file: File, videos_as_stills: bool
) -> tuple[Path | None, str]:
    """Resolve the on-disk source and the output filename for one file.

    Images and, normally, videos are their own file under their own
    name. With ``videos_as_stills`` a video is its best-frame JPEG,
    named ``<video_stem>_still.jpg``; ``(None, "")`` when that frame is
    not on file, since there is then no picture to write.
    """
    if file.file_type == "video" and videos_as_stills:
        if not file.best_frame_path:
            return None, ""
        return Path(file.best_frame_path), video_still_name(file.file_path)
    p = Path(file.file_path)
    return p, p.name


def build_deployment_folders(db: Session, project_id: str) -> dict[str, str]:
    """Map each deployment id to its source ``folder_path``, so a file's
    original subfolder structure can be derived relative to where it was
    scanned from. Deployments with no folder_path are omitted."""
    rows = db.execute(
        select(Deployment.id, Deployment.folder_path).where(
            Deployment.project_id == project_id
        )
    ).all()
    return {r.id: r.folder_path for r in rows if r.folder_path}


def source_subdir(file_path: str, source_root: str | None) -> str:
    """The file's directory relative to its source folder, as a posix
    path, so the user's original structure is preserved *under* the
    species folder (suffix placement). Empty when the file sits at the
    source root or its path isn't under it.
    """
    if not source_root:
        return ""
    try:
        rel = Path(file_path).parent.relative_to(source_root)
    except ValueError:
        return ""
    return "" if rel == Path(".") else rel.as_posix()


def separate_into_folders(
    db: Session,
    project_id: str,
    ctx: OutputContext,
    *,
    media_threshold: float,
    group_by: SeparateGroupBy = "taxonomic",
    include_empty: bool = True,
    excluded_label_ids: frozenset[str] | None = None,
    name_mode: NameMode = "common",
    group_events: bool = True,
    species_last: bool = False,
    place_files: bool = True,
    videos_as_stills: bool = False,
    progress_cb: Callable[[int, int], None] | None = None,
) -> SeparateFoldersResult:
    """Reorganise every file in the project into subdirectories under
    ``ctx.output_root``.

    Each file lands in exactly one folder, its main species. Under
    ``group_by="taxonomic"`` (default) animal files land in a nested
    chain ``<Class>/<Order>/<Family>/<Genus>/<species>/``; under
    ``group_by="flat"`` a single-segment folder named after the species;
    under ``group_by="none"`` flat at the output root. Non-animal files
    land in their fixed observation-type folder. ``group_events`` keeps
    every file of a burst in one folder, the event's main species.

    Every placement is a copy. ``videos_as_stills`` writes each video as
    its best-frame JPEG instead of the container (the blur case, see
    the module docstring).

    ``species_last`` flips the layering: instead of
    ``<species>/<source subfolder>/`` the file lands at
    ``<source subfolder>/<species>/``, so the user's original folders
    stay on top and species sits inside them (matches the layout
    camtrapR's ``recordTable`` expects: station folder, then species).
    No effect when there is no species folder (``group_by="none"``) or no
    source subfolder.

    ``excluded_label_ids`` filters detections: a file where every passing
    identified detection is excluded is skipped entirely (counted in
    ``skipped_excluded``). A mixed file still goes through, filed under
    its most confident non-excluded label.

    ``media_threshold`` is the media-output confidence threshold picked
    on the Save step: detections below it (unless verified) do not count
    towards placement, and a file's effective observation type is
    re-derived at this threshold rather than trusting the stored
    ``observation_type`` (which is derived at the project threshold).

    Each placement is recorded on ``ctx`` so downstream modules can find
    the file on disk without re-reading ``File.file_path``.

    ``place_files`` controls whether the bytes are actually written here.
    It's ``True`` for a separation-only run. When ``annotated_copies``
    also runs, the worker passes ``False``: that module re-encodes the
    annotated image (or plain-copies unchanged files) straight to the
    destination this function planned, so copying the bytes here first
    would just be overwritten. In deferred mode the placement is still
    computed, the folder is created, the name is reserved, and ``ctx`` is
    recorded, but the copy and the EXIF write are skipped. The counts are
    unchanged, so the completion summary reads the same. A video
    container is the exception and is always copied here: that module
    only writes JPEGs, so it can put a still beside the video but never
    the video itself.
    """
    project = db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id!r} not found")

    target_dir = ctx.output_root
    target_dir.mkdir(parents=True, exist_ok=True)
    threshold = media_threshold
    taxonomy_map = _build_taxonomy_map(db, project)

    # Empty map = per-file placement; populated = whole events grouped.
    event_primary: dict[str, str] = {}
    if group_events:
        event_primary = build_event_primary_labels(
            db, project_id, threshold, excluded_label_ids
        )

    # Source folders per deployment, to preserve each file's original
    # subfolder structure under the species folder.
    dep_folders = build_deployment_folders(db, project_id)

    # Stable order (matches output_preview) so collision suffixes land on
    # the same files every run: two same-named files from different
    # subfolders must not swap IMG_006.jpg / IMG_006_2.jpg between runs.
    files = db.execute(
        select(File)
        .join(Deployment, File.deployment_id == Deployment.id)
        .where(Deployment.project_id == project_id)
        .order_by(File.file_path)
    ).scalars().all()

    result = SeparateFoldersResult()

    # Destinations handed out this run. In deferred mode the files aren't
    # written until annotated_copies runs, so on-disk existence can't be
    # the uniqueness check; this set is.
    reserved: set[Path] = set()

    # One ExifToolHelper for the whole batch so the underlying
    # exiftool process stays alive across writes — start-up is the
    # expensive part.
    total = len(files)
    with ExifBatch() as exif_batch:
        for i, file in enumerate(files):
            # Report files processed so far (advances through skips too). The
            # worker throttles these before they reach the WebSocket.
            if progress_cb is not None:
                progress_cb(i, total)
            source, out_name = _media_source(file, videos_as_stills)
            if source is None:
                result.skipped_missing_source += 1
                result.errors.append(
                    f"Video has no best frame on disk: {file.file_path}"
                )
                continue
            if not source.exists():
                result.skipped_missing_source += 1
                result.errors.append(
                    f"Source file no longer on disk: {source}"
                )
                continue

            if file_is_dropped_by_filter(
                db, file, threshold, excluded_label_ids
            ):
                result.skipped_excluded += 1
                continue

            # Effective observation type at the media threshold. The
            # stored column is derived at the project threshold, which
            # no longer matches the media outputs' own confidence.
            #
            # Gated to the file's visible surface, which for a video is
            # the best frame: the artefact written here IS that frame's
            # JPEG, so deciding whether it is empty, and which folder it
            # goes in, from a box that is not in the picture is the bug
            # this module's sibling helpers already prevent. Without the
            # gate a clip whose best frame shows a person, with an animal
            # box on some other frame, was copied into `animal/`.
            obs_type = derive_observation_type(
                visible_detections(file, file.detections), threshold
            )

            # Empties are skipped from the copies unless opted in.
            if not include_empty and obs_type == "blank":
                continue

            folder = _folder_for_file(
                db,
                file,
                obs_type,
                threshold,
                taxonomy_map,
                group_by,
                excluded_label_ids,
                name_mode,
                grouped_label=event_primary.get(file.id),
            )

            # Preserve the original subfolder structure under the species
            # folder (suffix). Empty parts collapse, so "none" mode with
            # no subfolder lands flat at the output root.
            subdir = source_subdir(
                file.file_path, dep_folders.get(file.deployment_id)
            )
            ordered = (subdir, folder) if species_last else (folder, subdir)
            parts = [p for p in ordered if p]
            dest_dir = target_dir.joinpath(*parts) if parts else target_dir
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest, renamed = _unique_destination(dest_dir, out_name, reserved)
            reserved.add(dest)

            # A container gets its annotated still beside it later, and
            # that name has to be unique too: two clips with one stem and
            # different extensions, or a photo the camera named like a
            # still, would otherwise overwrite it. The name is allocated
            # here, where every name this run hands out is known, and
            # recorded for annotated_copies to read.
            is_container = not is_image_path(dest)
            still: Path | None = None
            if is_container:
                still, _ = _unique_destination(
                    dest_dir, video_still_name(dest.name), reserved
                )
                reserved.add(still)

            # Deferred mode hands the write to annotated_copies, which
            # only writes JPEGs: an image it re-encodes or plain-copies to
            # ``dest``, a video container it cannot write at all. So the
            # container is copied here whatever ``place_files`` says.
            if place_files or is_container:
                try:
                    # copy2 preserves mtime. Important for downstream tools
                    # that read modification time as a proxy for capture
                    # time.
                    shutil.copy2(source, dest)
                except OSError as e:
                    result.errors.append(f"Failed to copy {source}: {e}")
                    logger.exception(
                        f"separate_folders: copy failed for {source}"
                    )
                    continue

            result.copied_count += 1
            result.by_label[folder] += 1
            if renamed:
                result.renamed_count += 1
            ctx.record(file.id, dest)
            if still is not None:
                ctx.record_still(file.id, still)

            # Silent EXIF when the placement is an image format that
            # carries EXIF (a video still is one; a video container is
            # not, and gets no tags). Skipped in deferred mode:
            # ``annotated_copies`` writes the file and stamps the same tag
            # set. When it also draws / blurs it overwrites this anyway.
            if place_files and is_image_path(dest):
                tag_set = build_tag_set(
                    db,
                    file,
                    project,
                    APP_VERSION,
                    media_threshold=threshold,
                    excluded_label_ids=excluded_label_ids,
                )
                if tag_set is not None:
                    try:
                        exif_batch.write(dest, tag_set)
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            f"separate_folders: EXIF write failed for "
                            f"{dest}: {e}"
                        )

    if progress_cb is not None:
        progress_cb(total, total)

    logger.info(
        f"separate_folders: project={project_id} "
        f"group_by={group_by} group_events={group_events} "
        f"species_last={species_last} videos_as_stills={videos_as_stills} "
        f"written={result.written_count} renamed={result.renamed_count} "
        f"missing={result.skipped_missing_source} "
        f"excluded={result.skipped_excluded}"
    )
    return result
