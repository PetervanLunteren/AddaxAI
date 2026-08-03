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


def test_group_events_prefers_verified_over_higher_confidence(db, tmp_path):
    """A human-verified species owns the event folder even when an unverified
    AI box scored higher."""
    project = make_project(db, name="grp-verified", counting_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    fa = make_file(
        db, deployment_id=dep.id,
        file_path=_make_source(tmp_path, "A.jpg"), observation_type="animal",
    )
    fb = make_file(
        db, deployment_id=dep.id,
        file_path=_make_source(tmp_path, "B.jpg"), observation_type="animal",
    )
    make_detection(db, file_id=fa.id, confidence=0.95, label="dog")
    make_detection(
        db, file_id=fb.id, confidence=0.40, label="fox", verified=True
    )
    _link_event(db, dep.id, [fa.id, fb.id])

    target = tmp_path / "out"
    separate_into_folders(
        db, project.id, _ctx(target), group_events=True, media_threshold=0.5
    )

    assert (target / "other" / "fox" / "A.jpg").is_file()
    assert (target / "other" / "fox" / "B.jpg").is_file()
    assert not (target / "other" / "dog").exists()


def test_group_events_verified_multispecies_picks_most_common(db, tmp_path):
    """Several verified species in one event: the most common wins, even when
    another verified species scored higher."""
    project = make_project(db, name="grp-common", counting_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    file_ids: list[str] = []
    # One verified deer at high confidence...
    fd = make_file(
        db, deployment_id=dep.id,
        file_path=_make_source(tmp_path, "deer.jpg"), observation_type="animal",
    )
    make_detection(db, file_id=fd.id, confidence=0.9, label="deer", verified=True)
    file_ids.append(fd.id)
    # ...vs three verified chickens at lower confidence -> chicken is most common.
    for i in range(3):
        fc = make_file(
            db, deployment_id=dep.id,
            file_path=_make_source(tmp_path, f"chick{i}.jpg"),
            observation_type="animal",
        )
        make_detection(
            db, file_id=fc.id, confidence=0.5, label="chicken", verified=True
        )
        file_ids.append(fc.id)
    _link_event(db, dep.id, file_ids)

    target = tmp_path / "out"
    separate_into_folders(
        db, project.id, _ctx(target), group_events=True, media_threshold=0.5
    )

    # The whole event, deer file included, lands under chicken.
    assert (target / "other" / "chicken" / "deer.jpg").is_file()
    assert (target / "other" / "chicken" / "chick0.jpg").is_file()
    assert not (target / "other" / "deer").exists()


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
        best_frame_number=0,
        best_frame_path=str(frame),
        observation_type="animal",
    )
    # On the best frame: json_pipeline derives best_frame_path *from*
    # best_frame_number, so a video can never have one without the other,
    # and only that frame's detections describe the JPEG being written.
    make_detection(
        db, file_id=f.id, confidence=0.9, label="deer", frame_number=0
    )

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


def test_video_is_filed_by_its_best_frame_not_another_frame(db, tmp_path):
    """The bug this gate fixes. The JPEG written here IS the best frame, so
    deciding its folder from a box on another frame files a picture under a
    label that picture does not show. Best frame holds a person; frame 50
    holds a more confident animal called red fox. The copy must land in
    `person/`, and the preview must say the same thing."""
    project = make_project(db, name="vid-frame", counting_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    ensure_builtin_labels(db)
    frame = tmp_path / "cache" / "bf.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"best-frame-bytes")
    f = make_file(
        db,
        deployment_id=dep.id,
        file_path="/no/such/CLIP02.MP4",
        file_type="video",
        file_format="mp4",
        best_frame_number=3,
        best_frame_path=str(frame),
        observation_type="animal",
    )
    make_detection(
        db, file_id=f.id, category="person", confidence=0.80, frame_number=3
    )
    make_detection(
        db,
        file_id=f.id,
        category="animal",
        confidence=0.95,
        label="red fox",
        frame_number=50,
    )
    db.commit()

    target = tmp_path / "out"
    result = separate_into_folders(
        db, project.id, _ctx(target), media_threshold=0.5, group_by="flat"
    )

    assert result.copied_count == 1
    assert (target / "person" / "CLIP02_still.jpg").is_file()
    assert not (target / "red-fox").exists()

    preview = build_output_preview(
        db, project.id, group_by="flat", media_threshold=0.5
    )
    assert preview.by_media_tree == {"person": 1}


def test_video_whose_best_frame_is_empty_reads_blank(db, tmp_path):
    """Nothing passes on the frame being written, so the copy is a blank,
    not a red fox. With "copy empties" off it is skipped entirely. Either
    way the off-frame box is still in the data exports."""
    project = make_project(db, name="vid-empty", counting_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    frame = tmp_path / "cache" / "empty.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"bytes")
    f = make_file(
        db,
        deployment_id=dep.id,
        file_path="/no/such/CLIP03.MP4",
        file_type="video",
        file_format="mp4",
        best_frame_number=3,
        best_frame_path=str(frame),
        observation_type="animal",
    )
    make_detection(
        db,
        file_id=f.id,
        category="animal",
        confidence=0.95,
        label="red fox",
        frame_number=50,
    )
    db.commit()

    target = tmp_path / "out"
    result = separate_into_folders(
        db, project.id, _ctx(target), media_threshold=0.5, group_by="flat"
    )

    assert result.copied_count == 1
    assert (target / "blank" / "CLIP03_still.jpg").is_file()
    assert not (target / "red-fox").exists()

    skipped = separate_into_folders(
        db,
        project.id,
        _ctx(tmp_path / "out2"),
        media_threshold=0.5,
        group_by="flat",
        include_empty=False,
    )
    assert skipped.copied_count == 0


def test_event_grouping_uses_each_videos_best_frame(db, tmp_path):
    """`group_events` keeps a burst in one folder, and that folder is
    decided by build_event_primary_labels. It must count only boxes that
    exist as pictures: a video is copied to disk as its best frame, so an
    off-frame box naming the burst's folder files every still in it under
    a label none of them show. Caught by an end-to-end run, not by the
    audit, because grouping is on by default."""
    project = make_project(db, name="ev-frame", counting_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)

    frames = []
    files = []
    for i in range(2):
        frame = tmp_path / "cache" / f"bf{i}.jpg"
        frame.parent.mkdir(parents=True, exist_ok=True)
        frame.write_bytes(b"bytes")
        frames.append(frame)
        f = make_file(
            db,
            deployment_id=dep.id,
            file_path=f"/no/such/EV{i}.MP4",
            file_type="video",
            file_format="mp4",
            best_frame_number=3,
            best_frame_path=str(frame),
            observation_type="animal",
        )
        files.append(f)
        # On the saved frame: bushbuck. Off it, a more confident cattle box
        # that used to name the whole burst.
        make_detection(
            db, file_id=f.id, confidence=0.80, label="bushbuck", frame_number=3
        )
        make_detection(
            db, file_id=f.id, confidence=0.95, label="cattle", frame_number=50
        )
    _link_event(db, dep.id, [f.id for f in files])
    db.commit()

    target = tmp_path / "out"
    separate_into_folders(
        db,
        project.id,
        _ctx(target),
        media_threshold=0.5,
        group_by="flat",
        group_events=True,
    )

    assert (target / "bushbuck").is_dir()
    assert not (target / "cattle").exists()
