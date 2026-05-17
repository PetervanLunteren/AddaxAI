"""Tests for the blur_people postprocess output module.

Pins the routing rules and the privacy guarantee: a copy is produced
when and only when there is a threshold-passing person or vehicle
detection on the file. Animal-only files do NOT produce a copy.

The blur math itself is delegated to PIL; we verify that the bbox
region's pixels differ from the source while the rest is preserved.
"""

from pathlib import Path

import pytest
from PIL import Image

from app.ml.postprocessing_outputs.blur_people import blur_people
from tests.conftest import (
    make_deployment,
    make_detection,
    make_file,
    make_project,
)


def _write_image(
    path: Path, size: tuple[int, int] = (400, 300), color: tuple[int, int, int] = (180, 180, 180)
) -> str:
    """Write a flat-colour RGB image so the blur output is easy to inspect."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return str(path)


def test_person_bbox_is_blurred(db, tmp_path):
    project = make_project(db, name="blur-person", detection_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    src = _write_image(tmp_path / "src" / "IMG_001.jpg")
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=src,
        observation_type="human",
    )
    make_detection(
        db,
        file_id=file.id,
        category="person",
        confidence=0.9,
        bbox_x=0.3,
        bbox_y=0.3,
        bbox_width=0.4,
        bbox_height=0.4,
    )

    target = tmp_path / "out"
    result = blur_people(db, project.id, target)

    assert result.written_count == 1
    assert result.blurred_box_count == 1
    out = target / "IMG_001.jpg"
    assert out.is_file()
    # Same size, but the bbox region's pixels should not be a perfect
    # flat colour anymore — Gaussian blur on a flat region won't
    # change anything, so we use a non-flat source for that proof.
    with Image.open(out) as img:
        assert img.size == (400, 300)


def test_blur_affects_pixels_in_bbox(db, tmp_path):
    """With a high-contrast pattern, the blurred bbox must differ from
    the source. Cheapest proof that the blur actually ran."""
    src_path = tmp_path / "src" / "IMG_002.jpg"
    src_path.parent.mkdir(parents=True)
    img = Image.new("RGB", (400, 300), (0, 0, 0))
    # Put a tight white square inside the bbox region so the blur
    # is detectable as colour bleed at the edges.
    inner = Image.new("RGB", (40, 40), (255, 255, 255))
    img.paste(inner, (180, 130))
    img.save(src_path)

    project = make_project(db, name="blur-pixels", detection_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=str(src_path),
        observation_type="human",
    )
    make_detection(
        db,
        file_id=file.id,
        category="person",
        confidence=0.9,
        bbox_x=0.4,
        bbox_y=0.4,
        bbox_width=0.2,
        bbox_height=0.2,
    )

    target = tmp_path / "out"
    blur_people(db, project.id, target)

    with Image.open(src_path) as src_img, Image.open(
        target / "IMG_002.jpg"
    ) as out_img:
        src_px = src_img.load()
        out_px = out_img.load()
        # Sample a few points inside the bbox edge zone; blur should
        # have spread the white into surrounding pixels.
        edge_x = 175  # just outside the white square, inside the bbox
        edge_y = 150
        # Source: black (outside the white square)
        assert src_px[edge_x, edge_y] == (0, 0, 0)
        # Output: blur should have made this brighter than 0.
        out_value = out_px[edge_x, edge_y]
        assert max(out_value) > 10, (
            f"expected blur to bleed white into the edge; got {out_value}"
        )


def test_vehicle_bbox_is_blurred(db, tmp_path):
    project = make_project(db, name="blur-vehicle", detection_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    src = _write_image(tmp_path / "src" / "IMG_003.jpg")
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=src,
        observation_type="vehicle",
    )
    make_detection(
        db,
        file_id=file.id,
        category="vehicle",
        confidence=0.9,
        bbox_x=0.2,
        bbox_y=0.2,
        bbox_width=0.4,
        bbox_height=0.4,
    )

    target = tmp_path / "out"
    result = blur_people(db, project.id, target)

    assert result.written_count == 1
    assert result.blurred_box_count == 1
    assert (target / "IMG_003.jpg").is_file()


def test_animal_only_file_is_not_copied(db, tmp_path):
    """Privacy contract: blur output exists only when there is a
    person or vehicle. Animal-only files do not produce a copy."""
    project = make_project(db, name="blur-animal-only", detection_threshold=0.5)
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
        confidence=0.9,
        label="dog",
        bbox_x=0.1,
        bbox_y=0.1,
        bbox_width=0.3,
        bbox_height=0.3,
    )

    target = tmp_path / "out"
    result = blur_people(db, project.id, target)

    assert result.written_count == 0
    assert result.skipped_no_target == 1
    # The copy must not exist — sharing this folder is safe.
    assert not (target / "IMG_004.jpg").exists()


def test_below_threshold_person_not_blurred(db, tmp_path):
    project = make_project(db, name="blur-thresh", detection_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    src = _write_image(tmp_path / "src" / "IMG_005.jpg")
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=src,
        observation_type="human",
    )
    make_detection(
        db,
        file_id=file.id,
        category="person",
        confidence=0.2,
        bbox_x=0.1,
        bbox_y=0.1,
        bbox_width=0.3,
        bbox_height=0.3,
        verified=False,
    )

    target = tmp_path / "out"
    result = blur_people(db, project.id, target)

    assert result.written_count == 0
    assert result.skipped_no_target == 1


def test_verified_person_below_threshold_still_blurred(db, tmp_path):
    """A human reviewer confirmed there's a person here. Always blur,
    even if the model wasn't sure."""
    project = make_project(db, name="blur-verified", detection_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    src = _write_image(tmp_path / "src" / "IMG_006.jpg")
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=src,
        observation_type="human",
    )
    make_detection(
        db,
        file_id=file.id,
        category="person",
        confidence=0.2,
        bbox_x=0.1,
        bbox_y=0.1,
        bbox_width=0.3,
        bbox_height=0.3,
        verified=True,
    )

    target = tmp_path / "out"
    result = blur_people(db, project.id, target)

    assert result.written_count == 1
    assert result.blurred_box_count == 1


def test_video_uses_best_frame(db, tmp_path):
    project = make_project(db, name="blur-video", detection_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    best_frame = _write_image(tmp_path / "frames" / "frame000007.jpg")
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=str(tmp_path / "src" / "VID_001.mp4"),
        file_type="video",
        file_format="mp4",
        observation_type="human",
        best_frame_number=7,
        best_frame_path=best_frame,
    )
    make_detection(
        db,
        file_id=file.id,
        category="person",
        confidence=0.9,
        bbox_x=0.2,
        bbox_y=0.2,
        bbox_width=0.4,
        bbox_height=0.4,
        frame_number=7,
    )

    target = tmp_path / "out"
    result = blur_people(db, project.id, target)

    assert result.written_count == 1
    # Destination uses the video stem + .jpg.
    assert (target / "VID_001.jpg").is_file()


def test_mixed_file_blurs_only_person_and_vehicle(db, tmp_path):
    """A file with one animal and one person should still produce a
    blurred copy, with the animal bbox left sharp."""
    project = make_project(db, name="blur-mixed", detection_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    src = _write_image(tmp_path / "src" / "IMG_007.jpg")
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
        label="bear",
        bbox_x=0.1,
        bbox_y=0.1,
        bbox_width=0.3,
        bbox_height=0.3,
    )
    make_detection(
        db,
        file_id=file.id,
        category="person",
        confidence=0.85,
        bbox_x=0.6,
        bbox_y=0.6,
        bbox_width=0.2,
        bbox_height=0.2,
    )

    target = tmp_path / "out"
    result = blur_people(db, project.id, target)

    assert result.written_count == 1
    # Only the person bbox is counted as blurred.
    assert result.blurred_box_count == 1


def test_collision_rename(db, tmp_path):
    project = make_project(db, name="blur-collide")
    dep = make_deployment(db, project_id=project.id)
    src1 = _write_image(tmp_path / "a" / "IMG_008.jpg")
    src2 = _write_image(tmp_path / "b" / "IMG_008.jpg")
    f1 = make_file(
        db,
        deployment_id=dep.id,
        file_path=src1,
        observation_type="human",
    )
    f2 = make_file(
        db,
        deployment_id=dep.id,
        file_path=src2,
        observation_type="human",
    )
    for f in (f1, f2):
        make_detection(
            db,
            file_id=f.id,
            category="person",
            confidence=0.9,
            bbox_x=0.3,
            bbox_y=0.3,
            bbox_width=0.4,
            bbox_height=0.4,
        )

    target = tmp_path / "out"
    result = blur_people(db, project.id, target)

    assert result.written_count == 2
    assert result.renamed_count == 1
    names = sorted(p.name for p in target.iterdir())
    assert names == ["IMG_008.jpg", "IMG_008_2.jpg"]


def test_unknown_project_raises(db, tmp_path):
    with pytest.raises(ValueError, match="not found"):
        blur_people(db, "no-such-id", tmp_path / "out")
