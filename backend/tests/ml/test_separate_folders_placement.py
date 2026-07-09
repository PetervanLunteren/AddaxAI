"""Tests for the Save-step media controls on ``separate_folders``:

- Single-destination placement: every file lands in its main-species folder.
- ``group_events``: keeps a burst together under the event's main species.
- The uniform label filter dropping excluded person / vehicle files.
- Videos written as their best-frame JPEG.

The ``output_preview`` mirror is checked for the same scenarios so the live
preview never disagrees with the real run.
"""

import uuid
from datetime import datetime
from pathlib import Path

from app.ml.postprocessing_outputs._output_context import OutputContext
from app.ml.postprocessing_outputs.output_preview import build_output_preview
from app.ml.postprocessing_outputs.separate_folders import (
    separate_into_folders,
)
from app.ml.taxonomy_db import ensure_builtin_labels
from app.models.event import Event, event_files
from tests.conftest import (
    make_deployment,
    make_detection,
    make_file,
    make_project,
)


def _make_source(tmp_path: Path, name: str, content: bytes = b"x") -> str:
    src = tmp_path / "source" / name
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(content)
    return str(src)


def _ctx(output_root: Path) -> OutputContext:
    return OutputContext(output_root=output_root)


def _link_event(db, deployment_id: str, file_ids: list[str]) -> Event:
    """Create an event and attach existing files to it."""
    ev = Event(
        id=str(uuid.uuid4()),
        deployment_id=deployment_id,
        event_start_local=datetime(2024, 1, 1, 8, 0, 0),
        event_end_local=datetime(2024, 1, 1, 8, 1, 0),
        file_count=len(file_ids),
    )
    db.add(ev)
    db.flush()
    for seq, fid in enumerate(file_ids):
        db.execute(
            event_files.insert().values(
                event_id=ev.id, file_id=fid, sequence_number=seq
            )
        )
    db.flush()
    return ev


# ---------------------------------------------------------------------
# Primary-only placement (the new default)
# ---------------------------------------------------------------------


def test_places_multi_species_file_once(db, tmp_path):
    """A dog + wolf file lands in its most confident species' folder
    only: one copy, no duplication."""
    project = make_project(db, name="prim", counting_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    src = _make_source(tmp_path, "IMG_1.jpg")
    f = make_file(
        db, deployment_id=dep.id, file_path=src, observation_type="animal"
    )
    make_detection(db, file_id=f.id, confidence=0.95, label="dog")
    make_detection(db, file_id=f.id, confidence=0.80, label="wolf")

    target = tmp_path / "out"
    result = separate_into_folders(db, project.id, _ctx(target), media_threshold=0.5)

    assert result.copied_count == 1
    assert (target / "other" / "dog" / "IMG_1.jpg").is_file()
    assert not (target / "other" / "wolf").exists()


# ---------------------------------------------------------------------
# Event grouping
# ---------------------------------------------------------------------


def test_group_events_keeps_burst_in_one_folder(db, tmp_path):
    """Two files in one event: file A's top label is dog (0.95), file B's
    is fox (0.80). With grouping the whole event lands under dog/."""
    project = make_project(db, name="grp", counting_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    src_a = _make_source(tmp_path, "A.jpg")
    src_b = _make_source(tmp_path, "B.jpg")
    fa = make_file(
        db, deployment_id=dep.id, file_path=src_a, observation_type="animal"
    )
    fb = make_file(
        db, deployment_id=dep.id, file_path=src_b, observation_type="animal"
    )
    make_detection(db, file_id=fa.id, confidence=0.95, label="dog")
    make_detection(db, file_id=fb.id, confidence=0.80, label="fox")
    _link_event(db, dep.id, [fa.id, fb.id])

    target = tmp_path / "out"
    result = separate_into_folders(
        db, project.id, _ctx(target), group_events=True
    , media_threshold=0.5)

    # Both files land under the event's primary species (dog).
    assert (target / "other" / "dog" / "A.jpg").is_file()
    assert (target / "other" / "dog" / "B.jpg").is_file()
    assert not (target / "other" / "fox").exists()
    assert result.copied_count == 2


def test_group_events_off_splits_burst_per_file(db, tmp_path):
    """Same burst, grouping off: each file follows its own top label."""
    project = make_project(db, name="nogrp", counting_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    src_a = _make_source(tmp_path, "A.jpg")
    src_b = _make_source(tmp_path, "B.jpg")
    fa = make_file(
        db, deployment_id=dep.id, file_path=src_a, observation_type="animal"
    )
    fb = make_file(
        db, deployment_id=dep.id, file_path=src_b, observation_type="animal"
    )
    make_detection(db, file_id=fa.id, confidence=0.95, label="dog")
    make_detection(db, file_id=fb.id, confidence=0.80, label="fox")
    _link_event(db, dep.id, [fa.id, fb.id])

    target = tmp_path / "out"
    separate_into_folders(db, project.id, _ctx(target), group_events=False, media_threshold=0.5)

    assert (target / "other" / "dog" / "A.jpg").is_file()
    assert (target / "other" / "fox" / "B.jpg").is_file()


# ---------------------------------------------------------------------
# Uniform label filter: person / vehicle now droppable
# ---------------------------------------------------------------------


def test_excluding_person_drops_person_file(db, tmp_path):
    """Excluding the builtin person label drops a person file from the
    copies, while an animal file in the same run survives."""
    builtin = ensure_builtin_labels(db)
    person_tid = builtin["person"]

    project = make_project(db, name="excl-person", counting_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)

    person_src = _make_source(tmp_path, "PERSON.jpg")
    pf = make_file(
        db,
        deployment_id=dep.id,
        file_path=person_src,
        observation_type="human",
    )
    make_detection(
        db,
        file_id=pf.id,
        category="person",
        confidence=0.9,
        label=None,
        label_taxonomy_id=person_tid,
    )

    animal_src = _make_source(tmp_path, "DEER.jpg")
    af = make_file(
        db,
        deployment_id=dep.id,
        file_path=animal_src,
        observation_type="animal",
    )
    make_detection(db, file_id=af.id, confidence=0.9, label="deer")

    target = tmp_path / "out"
    result = separate_into_folders(
        db,
        project.id,
        _ctx(target),
        excluded_label_ids=frozenset({person_tid}),
        media_threshold=0.5,
    )

    assert result.skipped_excluded == 1
    assert not (target / "person").exists()
    assert (target / "other" / "deer" / "DEER.jpg").is_file()


# ---------------------------------------------------------------------
# Preview parity
# ---------------------------------------------------------------------


def test_preview_counts_one_placement(db):
    """Preview: a multi-species file is one placement, in its main
    species' folder."""
    project = make_project(db, name="prev-prim", counting_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    f = make_file(
        db, deployment_id=dep.id, observation_type="animal"
    )
    make_detection(db, file_id=f.id, confidence=0.95, label="dog")
    make_detection(db, file_id=f.id, confidence=0.80, label="wolf")

    preview = build_output_preview(db, project.id, group_by="taxonomic", media_threshold=0.5)

    assert preview.by_media_tree == {"other/dog": 1}


def test_video_copied_as_best_frame_jpeg(db, tmp_path):
    """A video is written as its best-frame JPEG (``<stem>_still.jpg``),
    not the original container — never the full .MP4."""
    project = make_project(db, name="vid", counting_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    frame = tmp_path / "cache" / "frame.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"best-frame-bytes")
    f = make_file(
        db,
        deployment_id=dep.id,
        file_path="/no/such/CLIP01.MP4",  # need not exist on disk
        file_type="video",
        file_format="mp4",
        best_frame_path=str(frame),
        observation_type="animal",
    )
    make_detection(db, file_id=f.id, confidence=0.9, label="deer")

    target = tmp_path / "out"
    result = separate_into_folders(db, project.id, _ctx(target), media_threshold=0.5)

    assert result.copied_count == 1
    jpeg = target / "other" / "deer" / "CLIP01_still.jpg"
    assert jpeg.is_file()
    assert jpeg.read_bytes() == b"best-frame-bytes"
    assert not (target / "other" / "deer" / "CLIP01.MP4").exists()


def test_video_without_best_frame_is_skipped(db, tmp_path):
    """A video with no best frame on file is skipped, not copied as the
    raw container."""
    project = make_project(db, name="vid-nobf", counting_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    f = make_file(
        db,
        deployment_id=dep.id,
        file_path="/no/such/CLIP02.MP4",
        file_type="video",
        file_format="mp4",
        best_frame_path=None,
        observation_type="animal",
    )
    make_detection(db, file_id=f.id, confidence=0.9, label="deer")

    target = tmp_path / "out"
    result = separate_into_folders(db, project.id, _ctx(target), media_threshold=0.5)

    assert result.copied_count == 0
    assert result.skipped_missing_source == 1


# ---------------------------------------------------------------------
# Original subfolder preservation (suffix placement)
# ---------------------------------------------------------------------


def test_preserves_source_subfolder_under_species(db, tmp_path):
    """A file under ``source/cam01/`` lands at ``species/cam01/`` —
    species on top, the user's original structure preserved beneath."""
    project = make_project(db, name="subdir", counting_threshold=0.5)
    source = tmp_path / "source"
    dep = make_deployment(
        db, project_id=project.id, folder_path=str(source)
    )
    src = source / "cam01" / "sub" / "IMG_1.jpg"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"x")
    f = make_file(
        db, deployment_id=dep.id, file_path=str(src),
        observation_type="animal",
    )
    make_detection(db, file_id=f.id, confidence=0.9, label="deer")

    target = tmp_path / "out"
    separate_into_folders(db, project.id, _ctx(target), group_by="flat", media_threshold=0.5)

    assert (target / "deer" / "cam01" / "sub" / "IMG_1.jpg").is_file()
    # Not flattened to the species root.
    assert not (target / "deer" / "IMG_1.jpg").exists()


def test_species_last_puts_source_folder_on_top(db, tmp_path):
    """``species_last`` flips the layering: ``cam01/sub/species/`` instead
    of ``species/cam01/sub/`` — the user's folders on top, species inside
    (the camtrapR station/species layout)."""
    project = make_project(db, name="species-last", counting_threshold=0.5)
    source = tmp_path / "source"
    dep = make_deployment(
        db, project_id=project.id, folder_path=str(source)
    )
    src = source / "cam01" / "sub" / "IMG_1.jpg"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"x")
    f = make_file(
        db, deployment_id=dep.id, file_path=str(src),
        observation_type="animal",
    )
    make_detection(db, file_id=f.id, confidence=0.9, label="deer")

    target = tmp_path / "out"
    separate_into_folders(
        db, project.id, _ctx(target), group_by="flat", species_last=True
    , media_threshold=0.5)

    assert (target / "cam01" / "sub" / "deer" / "IMG_1.jpg").is_file()
    # Not the default species-on-top layout.
    assert not (target / "deer" / "cam01" / "sub" / "IMG_1.jpg").exists()


def test_none_mode_mirrors_source_tree(db, tmp_path):
    """``group_by="none"`` mirrors the source tree: no species folder,
    original structure preserved."""
    project = make_project(db, name="mirror", counting_threshold=0.5)
    source = tmp_path / "source"
    dep = make_deployment(
        db, project_id=project.id, folder_path=str(source)
    )
    src = source / "cam01" / "IMG_2.jpg"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"x")
    f = make_file(
        db, deployment_id=dep.id, file_path=str(src),
        observation_type="animal",
    )
    make_detection(db, file_id=f.id, confidence=0.9, label="deer")

    target = tmp_path / "out"
    separate_into_folders(db, project.id, _ctx(target), group_by="none", media_threshold=0.5)

    assert (target / "cam01" / "IMG_2.jpg").is_file()
    assert not (target / "deer").exists()


def test_preview_subfolder_reported_in_media_tree(db, tmp_path):
    """Under "No subfolders" a file in a source subfolder is counted in
    by_media_tree at that subfolder, not the root-file list."""
    project = make_project(db, name="subdir-prev", counting_threshold=0.5)
    source = tmp_path / "source"
    dep = make_deployment(
        db, project_id=project.id, folder_path=str(source)
    )
    f = make_file(
        db,
        deployment_id=dep.id,
        file_path=str(source / "cam01" / "IMG_3.jpg"),
        observation_type="animal",
    )
    make_detection(db, file_id=f.id, confidence=0.9, label="deer")

    preview = build_output_preview(db, project.id, group_by="none", media_threshold=0.5)

    assert preview.by_media_tree == {"cam01": 1}
    assert preview.root_files == []


def test_preview_root_files_reported_as_filename_list(db, tmp_path):
    """Under "No subfolders" a file at the source root feeds the
    root-file sample, not the media tree."""
    project = make_project(db, name="root-prev", counting_threshold=0.5)
    source = tmp_path / "source"
    dep = make_deployment(
        db, project_id=project.id, folder_path=str(source)
    )
    f = make_file(
        db,
        deployment_id=dep.id,
        file_path=str(source / "IMG_4.jpg"),
        observation_type="animal",
    )
    make_detection(db, file_id=f.id, confidence=0.9, label="deer")

    preview = build_output_preview(db, project.id, group_by="none", media_threshold=0.5)

    assert preview.by_media_tree == {}
    assert preview.root_files == ["IMG_4.jpg"]


def test_preview_drops_excluded_person(db):
    """Preview mirrors the run: excluding person counts the person file
    in dropped_by_filter, not in scope."""
    builtin = ensure_builtin_labels(db)
    person_tid = builtin["person"]

    project = make_project(db, name="prev-person", counting_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    pf = make_file(db, deployment_id=dep.id, observation_type="human")
    make_detection(
        db,
        file_id=pf.id,
        category="person",
        confidence=0.9,
        label=None,
        label_taxonomy_id=person_tid,
    )

    preview = build_output_preview(
        db, project.id, excluded_label_ids=frozenset({person_tid})
    , media_threshold=0.5)

    assert preview.dropped_by_filter == 1
    assert preview.in_scope_files == 0
