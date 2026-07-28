"""Tests for the combined annotated_copies module.

Covers the four (draw_bboxes, anonymise) toggle combinations on a
single image, multi-placement under separation (one source ending up
in two label folders), the video best-frame routing, and the
file-level filter / missing-source / no-effect short-circuits.

We use small in-memory PIL images written to ``tmp_path`` rather than
shipping fixture JPEGs; the visual style is covered separately in
``test_visualisation_style.py`` and ``_visualisation_style.py``.
"""

from pathlib import Path

import pytest
from PIL import Image

from app.ml.postprocessing_outputs._output_context import OutputContext
from app.ml.postprocessing_outputs.annotated_copies import (
    write_annotated_copies,
)
from app.ml.postprocessing_outputs.separate_folders import (
    separate_into_folders,
)
from tests.conftest import (
    make_deployment,
    make_detection,
    make_file,
    make_project,
)


def _write_jpeg(path: Path, size: tuple[int, int] = (200, 200)) -> str:
    """Create a tiny JPEG so PIL has something to open."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (180, 200, 220)).save(path, format="JPEG")
    return str(path)


def _ctx(output_root: Path, resolved: dict[str, list[Path]] | None = None) -> OutputContext:
    ctx = OutputContext(output_root=output_root)
    if resolved:
        ctx.resolved_paths.update(resolved)
    return ctx


def test_requires_at_least_one_effect(db, tmp_path):
    """Calling with both flags off is a programming error; the worker
    short-circuits before reaching the module."""
    project = make_project(db, name="empty-effects")
    with pytest.raises(ValueError, match="at least one"):
        write_annotated_copies(
            db,
            project.id,
            _ctx(tmp_path / "out"),
            draw_bboxes=False,
            anonymise=False,
            media_threshold=0.5,
        )


def test_visualise_only_writes_bboxes_and_skips_blur(db, tmp_path):
    project = make_project(db, name="vis-only", counting_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    src = _write_jpeg(tmp_path / "src" / "animal.jpg")
    file = make_file(
        db, deployment_id=dep.id, file_path=src, observation_type="animal"
    )
    make_detection(db, file_id=file.id, category="animal", confidence=0.9, label="dog")

    target = tmp_path / "out"
    result = write_annotated_copies(
        db,
        project.id,
        _ctx(target),
        draw_bboxes=True,
        anonymise=False,
        media_threshold=0.5,
    )

    assert result.written_count == 1
    assert result.bbox_count == 1
    assert result.blurred_box_count == 0
    assert (target / "animal.jpg").is_file()


def test_anonymise_only_writes_blur_and_skips_bboxes(db, tmp_path):
    project = make_project(db, name="blur-only", counting_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    src = _write_jpeg(tmp_path / "src" / "person.jpg")
    file = make_file(
        db, deployment_id=dep.id, file_path=src, observation_type="human"
    )
    make_detection(db, file_id=file.id, category="person", confidence=0.95)

    target = tmp_path / "out"
    result = write_annotated_copies(
        db,
        project.id,
        _ctx(target),
        draw_bboxes=False,
        anonymise=True,
        media_threshold=0.5,
    )

    assert result.written_count == 1
    assert result.blurred_box_count == 1
    assert result.bbox_count == 0


def test_combined_writes_one_image_with_blur_and_bboxes(db, tmp_path):
    """Both flags on: one image per source, blurred first, then boxes
    drawn over the obscured pixels. Counts both effects."""
    project = make_project(db, name="combined", counting_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    src = _write_jpeg(tmp_path / "src" / "mixed.jpg")
    file = make_file(
        db, deployment_id=dep.id, file_path=src, observation_type="animal"
    )
    make_detection(
        db, file_id=file.id, category="animal", confidence=0.95, label="dog"
    )
    make_detection(
        db,
        file_id=file.id,
        category="person",
        confidence=0.9,
        bbox_x=0.6,
        bbox_y=0.2,
        bbox_width=0.2,
        bbox_height=0.4,
    )

    target = tmp_path / "out"
    result = write_annotated_copies(
        db,
        project.id,
        _ctx(target),
        draw_bboxes=True,
        anonymise=True,
        media_threshold=0.5,
    )

    assert result.written_count == 1
    assert result.bbox_count == 2  # animal + person both drawn
    assert result.blurred_box_count == 1  # person bbox blurred
    assert (target / "mixed.jpg").is_file()


def test_no_separation_fallback_uses_output_root(db, tmp_path):
    """With no resolved paths on the context, the destination is
    ``output_root/<original_name>``."""
    project = make_project(db, name="no-sep", counting_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    src = _write_jpeg(tmp_path / "src" / "IMG_001.jpg")
    file = make_file(
        db, deployment_id=dep.id, file_path=src, observation_type="animal"
    )
    make_detection(db, file_id=file.id, category="animal", confidence=0.9, label="dog")

    target = tmp_path / "out"
    write_annotated_copies(
        db,
        project.id,
        _ctx(target),
        draw_bboxes=True,
        anonymise=False,
        media_threshold=0.5,
    )
    assert (target / "IMG_001.jpg").is_file()


def test_multi_placement_writes_to_every_destination(db, tmp_path):
    """When separation placed the file in two label folders, the
    annotated copy is written to both."""
    project = make_project(db, name="multi-placement", counting_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    src = _write_jpeg(tmp_path / "src" / "multi.jpg")
    file = make_file(
        db, deployment_id=dep.id, file_path=src, observation_type="animal"
    )
    make_detection(db, file_id=file.id, category="animal", confidence=0.95, label="dog")
    make_detection(db, file_id=file.id, category="animal", confidence=0.85, label="wolf")

    target = tmp_path / "out"
    # Pretend separation already placed copies; the resolved paths must
    # exist on disk because PIL would otherwise refuse to overwrite a
    # nonexistent file's parent... actually PIL.save creates files, but
    # we need the parent folder to exist.
    dog_dst = target / "dog" / "multi.jpg"
    wolf_dst = target / "wolf" / "multi.jpg"
    dog_dst.parent.mkdir(parents=True, exist_ok=True)
    wolf_dst.parent.mkdir(parents=True, exist_ok=True)

    ctx = _ctx(target, {file.id: [dog_dst, wolf_dst]})
    result = write_annotated_copies(
        db,
        project.id,
        ctx,
        draw_bboxes=True,
        anonymise=False,
        media_threshold=0.5,
    )

    assert result.written_count == 2
    assert dog_dst.is_file()
    assert wolf_dst.is_file()


def test_video_annotated_best_frame_lands_at_recorded_still(db, tmp_path):
    """For a video, separation records the best-frame JPEG
    (``clip_still.jpg``); annotation overwrites that same file with the
    boxed version. The video container is never written."""
    project = make_project(db, name="video-anno", counting_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    # The video file itself doesn't need to exist on disk — we read the
    # best-frame JPEG.
    best_frame = _write_jpeg(tmp_path / "cache" / "frame000001.jpg")
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=str(tmp_path / "src" / "clip.mp4"),
        file_type="video",
        file_format="mp4",
        observation_type="animal",
        best_frame_number=1,
        best_frame_path=best_frame,
    )
    make_detection(
        db,
        file_id=file.id,
        category="animal",
        confidence=0.95,
        label="dog",
        frame_number=1,
    )

    target = tmp_path / "out"
    placed = target / "dog" / "clip_still.jpg"
    placed.parent.mkdir(parents=True, exist_ok=True)
    ctx = _ctx(target, {file.id: [placed]})
    result = write_annotated_copies(
        db,
        project.id,
        ctx,
        draw_bboxes=True,
        anonymise=False,
        media_threshold=0.5,
    )

    assert result.written_count == 1
    assert placed.is_file()
    assert not (target / "dog" / "clip.mp4").exists()  # container untouched


def test_no_change_skip_when_only_anonymise_and_no_targets(db, tmp_path):
    """Anonymise only + an animal-only file (no person / vehicle) → no
    visible change, file skipped to avoid identical copies."""
    project = make_project(db, name="no-targets", counting_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    src = _write_jpeg(tmp_path / "src" / "fox.jpg")
    file = make_file(
        db, deployment_id=dep.id, file_path=src, observation_type="animal"
    )
    make_detection(db, file_id=file.id, category="animal", confidence=0.9, label="fox")

    target = tmp_path / "out"
    result = write_annotated_copies(
        db,
        project.id,
        _ctx(target),
        draw_bboxes=False,
        anonymise=True,
        media_threshold=0.5,
    )

    assert result.written_count == 0
    assert result.skipped_no_change == 1
    assert not (target / "fox.jpg").exists()


def test_missing_source_is_skipped(db, tmp_path):
    project = make_project(db, name="missing-src", counting_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    # File row points at a path that doesn't exist on disk.
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=str(tmp_path / "nope" / "ghost.jpg"),
        observation_type="animal",
    )
    make_detection(db, file_id=file.id, category="animal", confidence=0.9, label="dog")

    target = tmp_path / "out"
    result = write_annotated_copies(
        db,
        project.id,
        _ctx(target),
        draw_bboxes=True,
        anonymise=False,
        media_threshold=0.5,
    )

    assert result.written_count == 0
    assert result.skipped_missing_source == 1


def test_excluded_label_skips_file(db, tmp_path):
    """File-level filter drops animal files whose every passing label
    is in the exclusion set — same rule as separate_folders."""
    project = make_project(db, name="excl", counting_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    src = _write_jpeg(tmp_path / "src" / "dog.jpg")
    file = make_file(
        db, deployment_id=dep.id, file_path=src, observation_type="animal"
    )
    make_detection(db, file_id=file.id, category="animal", confidence=0.9, label="dog")

    target = tmp_path / "out"
    result = write_annotated_copies(
        db,
        project.id,
        _ctx(target),
        draw_bboxes=True,
        anonymise=False,
        excluded_label_ids=frozenset({"dog"}),
        media_threshold=0.5,
    )

    assert result.written_count == 0
    assert result.skipped_excluded == 1
    assert not (target / "dog.jpg").exists()


def test_deferred_separation_writes_each_file_once(db, tmp_path):
    """Both separate + annotate: separation defers the bytes
    (place_files=False), annotation owns every write. An effect file is
    written by annotation; a placed no-effect file (blank, kept via
    include_empty) is plain-copied by annotation, not left missing. No
    file is written twice."""
    project = make_project(db, name="deferred", counting_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)

    animal_src = _write_jpeg(tmp_path / "src" / "animal.jpg")
    animal = make_file(
        db, deployment_id=dep.id, file_path=animal_src,
        observation_type="animal",
    )
    make_detection(
        db, file_id=animal.id, category="animal", confidence=0.9, label="dog"
    )

    blank_src = _write_jpeg(tmp_path / "src" / "blank.jpg")
    blank = make_file(
        db, deployment_id=dep.id, file_path=blank_src,
        observation_type="blank",
    )  # no detections → nothing to draw

    media_root = tmp_path / "out" / "addaxai-media"
    ctx = OutputContext(output_root=media_root)

    sep = separate_into_folders(
        db, project.id, ctx,
        media_threshold=0.5, mode="copy", group_by="none",
        include_empty=True, group_events=False,
        place_files=False,
    )
    # Placements were planned for both files, but nothing is on disk yet.
    assert ctx.resolved_for(animal.id)
    assert ctx.resolved_for(blank.id)
    for dests in (ctx.resolved_for(animal.id), ctx.resolved_for(blank.id)):
        assert not dests[0].exists()

    ann = write_annotated_copies(
        db, project.id, ctx,
        media_threshold=0.5, draw_bboxes=True, anonymise=False,
        copy_unchanged=True,
    )

    # Effect file drawn; blank plain-copied (still counted no-change).
    assert ann.written_count == 1
    assert ann.bbox_count == 1
    assert ann.skipped_no_change == 1
    assert ctx.resolved_for(animal.id)[0].is_file()
    assert ctx.resolved_for(blank.id)[0].is_file()
    # Separation counted both placements, so the summary is unchanged.
    assert sep.copied_count == 2


def test_unknown_project_raises(db, tmp_path):
    with pytest.raises(ValueError, match="not found"):
        write_annotated_copies(
            db,
            "no-such-id",
            _ctx(tmp_path / "out"),
            draw_bboxes=True,
            anonymise=False,
            media_threshold=0.5,
        )
