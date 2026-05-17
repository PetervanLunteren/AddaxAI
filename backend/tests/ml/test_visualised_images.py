"""Tests for the visualised_images postprocess output module.

The PIL drawing itself we trust; what we pin are the routing rules:
which files get a visualised copy, which get skipped, that videos
draw on the best frame, that thresholds are respected, and that
collisions get renamed.
"""

from pathlib import Path

import pytest
from PIL import Image

from app.ml.postprocessing_outputs.visualised_images import (
    visualise_images,
)
from tests.conftest import (
    make_deployment,
    make_detection,
    make_file,
    make_project,
)


def _write_image(path: Path, size: tuple[int, int] = (200, 150)) -> str:
    """Write a minimal RGB image and return its absolute path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (180, 180, 180)).save(path)
    return str(path)


def test_image_with_detection_writes_visualised_copy(db, tmp_path):
    project = make_project(db, name="vis-image", detection_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    src = _write_image(tmp_path / "src" / "IMG_001.jpg")
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=src,
        observation_type="animal",
    )
    make_detection(
        db,
        file_id=file.id,
        category="animal",
        confidence=0.9,
        label="dog",
        bbox_x=0.1,
        bbox_y=0.1,
        bbox_width=0.3,
        bbox_height=0.4,
    )

    target = tmp_path / "out"
    result = visualise_images(db, project.id, target)

    assert result.written_count == 1
    assert (target / "IMG_001.jpg").is_file()
    # Output is the same size as the source.
    with Image.open(target / "IMG_001.jpg") as img:
        assert img.size == (200, 150)


def test_image_without_detection_skipped(db, tmp_path):
    project = make_project(db, name="vis-no-det")
    dep = make_deployment(db, project_id=project.id)
    src = _write_image(tmp_path / "src" / "IMG_002.jpg")
    make_file(
        db,
        deployment_id=dep.id,
        file_path=src,
        observation_type="blank",
    )

    target = tmp_path / "out"
    result = visualise_images(db, project.id, target)

    assert result.written_count == 0
    assert result.skipped_no_bbox == 1
    assert not (target / "IMG_002.jpg").exists()


def test_detection_below_threshold_does_not_produce_output(db, tmp_path):
    project = make_project(db, name="vis-thresh", detection_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    src = _write_image(tmp_path / "src" / "IMG_003.jpg")
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=src,
        observation_type="animal",
    )
    make_detection(
        db,
        file_id=file.id,
        category="animal",
        confidence=0.2,
        label="dog",
        bbox_x=0.1,
        bbox_y=0.1,
        bbox_width=0.3,
        bbox_height=0.4,
        verified=False,
    )

    target = tmp_path / "out"
    result = visualise_images(db, project.id, target)

    # Detection is below threshold and not verified, file has no
    # other detections, so it ends up in the "no bbox" bucket.
    assert result.written_count == 0
    assert result.skipped_no_bbox == 1


def test_verified_detection_below_threshold_still_drawn(db, tmp_path):
    """The threshold-with-verified-override rule applies here too:
    a human-verified low-confidence box must still appear on the
    visualised copy."""
    project = make_project(db, name="vis-verified", detection_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    src = _write_image(tmp_path / "src" / "IMG_004.jpg")
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=src,
        observation_type="animal",
    )
    make_detection(
        db,
        file_id=file.id,
        category="animal",
        confidence=0.2,
        label="cat",
        bbox_x=0.1,
        bbox_y=0.1,
        bbox_width=0.3,
        bbox_height=0.4,
        verified=True,
    )

    target = tmp_path / "out"
    result = visualise_images(db, project.id, target)

    assert result.written_count == 1


def test_video_uses_best_frame(db, tmp_path):
    project = make_project(db, name="vis-video", detection_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    video_path = str(tmp_path / "src" / "VID_001.mp4")
    best_frame = _write_image(tmp_path / "frames" / "frame000042.jpg")
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=video_path,
        file_type="video",
        file_format="mp4",
        observation_type="animal",
        best_frame_number=42,
        best_frame_path=best_frame,
    )
    # Detection on the best frame.
    make_detection(
        db,
        file_id=file.id,
        category="animal",
        confidence=0.9,
        label="elephant",
        bbox_x=0.2,
        bbox_y=0.2,
        bbox_width=0.4,
        bbox_height=0.4,
        frame_number=42,
    )
    # Detection on a different frame — must be ignored when drawing
    # on the best frame.
    make_detection(
        db,
        file_id=file.id,
        category="animal",
        confidence=0.9,
        label="lion",
        bbox_x=0.5,
        bbox_y=0.5,
        bbox_width=0.2,
        bbox_height=0.2,
        frame_number=10,
    )

    target = tmp_path / "out"
    result = visualise_images(db, project.id, target)

    assert result.written_count == 1
    # Destination is the video stem with .jpg.
    assert (target / "VID_001.jpg").is_file()


def test_video_without_best_frame_skipped(db, tmp_path):
    project = make_project(db, name="vis-video-noframe")
    dep = make_deployment(db, project_id=project.id)
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=str(tmp_path / "VID.mp4"),
        file_type="video",
        file_format="mp4",
        observation_type="animal",
        best_frame_number=None,
        best_frame_path=None,
    )
    make_detection(
        db,
        file_id=file.id,
        category="animal",
        confidence=0.9,
        bbox_x=0.1,
        bbox_y=0.1,
        bbox_width=0.3,
        bbox_height=0.3,
        frame_number=10,
    )

    target = tmp_path / "out"
    result = visualise_images(db, project.id, target)

    assert result.written_count == 0
    assert result.skipped_missing_source == 1


def test_collision_rename(db, tmp_path):
    project = make_project(db, name="vis-collide")
    dep = make_deployment(db, project_id=project.id)
    # Two source images with the same basename in different folders.
    src1 = _write_image(tmp_path / "a" / "IMG_005.jpg")
    src2 = _write_image(tmp_path / "b" / "IMG_005.jpg")
    f1 = make_file(
        db,
        deployment_id=dep.id,
        file_path=src1,
        observation_type="animal",
    )
    f2 = make_file(
        db,
        deployment_id=dep.id,
        file_path=src2,
        observation_type="animal",
    )
    for f in (f1, f2):
        make_detection(
            db,
            file_id=f.id,
            category="animal",
            confidence=0.9,
            bbox_x=0.1,
            bbox_y=0.1,
            bbox_width=0.3,
            bbox_height=0.3,
        )

    target = tmp_path / "out"
    result = visualise_images(db, project.id, target)

    assert result.written_count == 2
    assert result.renamed_count == 1
    names = sorted(p.name for p in target.iterdir())
    assert names == ["IMG_005.jpg", "IMG_005_2.jpg"]


def test_event_level_detection_without_bbox_skipped(db, tmp_path):
    project = make_project(db, name="vis-event")
    dep = make_deployment(db, project_id=project.id)
    src = _write_image(tmp_path / "src" / "IMG_006.jpg")
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=src,
        observation_type="animal",
    )
    # Event-level observation: confidence high but no bbox.
    make_detection(
        db,
        file_id=file.id,
        category="animal",
        confidence=0.9,
        label="bird",
        bbox_x=None,
        bbox_y=None,
        bbox_width=None,
        bbox_height=None,
    )

    target = tmp_path / "out"
    result = visualise_images(db, project.id, target)

    assert result.written_count == 0
    assert result.skipped_no_bbox == 1


def test_unknown_project_raises(db, tmp_path):
    with pytest.raises(ValueError, match="not found"):
        visualise_images(db, "missing-id", tmp_path / "out")
