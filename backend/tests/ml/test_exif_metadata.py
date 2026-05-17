"""Tests for the EXIF writing — shared helper + explicit module.

The helper (`_exif_writer.build_tag_set`) is mostly string formatting,
worth pinning so it stays stable for downstream scripts.

The explicit module (`exif_metadata.write_exif_predictions`) has two
modes (overwrite / copy) that need to be exercised end-to-end against
actual JPEG bytes, since exiftool reads/writes the file on disk.

We skip the exiftool round-trip tests when the binary is not on PATH
(CI environments without it) — the helper test runs everywhere.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from app.ml.postprocessing_outputs._exif_writer import (
    ExifBatch,
    build_tag_set,
    is_image_path,
)
from app.ml.postprocessing_outputs.exif_metadata import (
    EXIF_COPIES_SUBDIR,
    write_exif_predictions,
)
from tests.conftest import (
    make_deployment,
    make_detection,
    make_file,
    make_project,
)


def _exiftool_available() -> bool:
    return shutil.which("exiftool") is not None


pytestmark_exiftool = pytest.mark.skipif(
    not _exiftool_available(),
    reason="exiftool binary not available on PATH",
)


def _write_jpeg(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (200, 150), (100, 100, 100)).save(path, "JPEG")
    return str(path)


def _read_tag(path: Path, tag: str) -> str | None:
    """Read a single EXIF/XMP tag back from disk using the exiftool
    binary directly so the assertion doesn't depend on the read side
    of the same library code under test."""
    out = subprocess.run(
        ["exiftool", "-s", "-s", "-s", f"-{tag}", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    line = out.stdout.strip()
    return line or None


# ---------------------------------------------------------------------
# Helper: build_tag_set
# ---------------------------------------------------------------------


def test_build_tag_set_skips_files_without_detections(db):
    project = make_project(db, name="exif-empty", detection_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    file = make_file(
        db, deployment_id=dep.id, observation_type="blank"
    )
    assert build_tag_set(db, file, project, "0.0.0") is None


def test_build_tag_set_summary_and_species(db):
    project = make_project(db, name="exif-summary", detection_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    file = make_file(
        db, deployment_id=dep.id, observation_type="animal"
    )
    make_detection(
        db,
        file_id=file.id,
        category="animal",
        confidence=0.9,
        label="dog",
        label_confidence=0.85,
    )
    make_detection(
        db,
        file_id=file.id,
        category="animal",
        confidence=0.85,
        label="wolf",
        label_confidence=0.80,
    )

    tag_set = build_tag_set(db, file, project, "0.2.0-beta.2")

    assert tag_set is not None
    # Summary is "<Label> <pct>%" joined, in confidence order.
    assert tag_set.image_description.startswith("Dog 90%")
    assert "Wolf 85%" in tag_set.image_description
    # Species tags carry distinct, original-case labels.
    assert tag_set.species_tags == ("dog", "wolf")
    # Software is recognisable as AddaxAI plus the model id.
    assert "AddaxAI" in tag_set.software
    assert "0.2.0-beta.2" in tag_set.software
    # UserComment is parseable JSON carrying the full detection list.
    payload = json.loads(tag_set.user_comment_json)
    assert payload["app"] == "AddaxAI 0.2.0-beta.2"
    assert len(payload["detections"]) == 2


def test_build_tag_set_caps_summary_at_five(db):
    """Six detections collapse to a top-5 summary plus a `+ N more`
    suffix so ImageDescription stays readable in thumbnail viewers."""
    project = make_project(db, name="exif-cap", detection_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    file = make_file(
        db, deployment_id=dep.id, observation_type="animal"
    )
    for n, conf in enumerate([0.99, 0.95, 0.9, 0.85, 0.8, 0.75]):
        make_detection(
            db,
            file_id=file.id,
            category="animal",
            confidence=conf,
            label=f"sp{n}",
        )

    tag_set = build_tag_set(db, file, project, "0.0.0")
    assert tag_set is not None
    assert tag_set.image_description.count(",") == 5  # 5 entries + 1 "+ N more"
    assert "+ 1 more" in tag_set.image_description


def test_is_image_path():
    assert is_image_path(Path("a.jpg"))
    assert is_image_path(Path("a.JPEG"))
    assert is_image_path(Path("a.tiff"))
    assert is_image_path(Path("a.PNG"))
    assert not is_image_path(Path("a.mp4"))
    assert not is_image_path(Path("a.txt"))


# ---------------------------------------------------------------------
# Helper: ExifBatch round-trip (requires exiftool on PATH)
# ---------------------------------------------------------------------


@pytestmark_exiftool
def test_exif_batch_writes_image_description(db, tmp_path):
    project = make_project(db, name="exif-rt", detection_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    src = _write_jpeg(tmp_path / "IMG.jpg")
    file = make_file(
        db, deployment_id=dep.id, file_path=src, observation_type="animal"
    )
    make_detection(
        db,
        file_id=file.id,
        category="animal",
        confidence=0.9,
        label="dog",
    )

    tag_set = build_tag_set(db, file, project, "0.0.1")
    assert tag_set is not None

    with ExifBatch() as batch:
        batch.write(Path(src), tag_set)

    assert _read_tag(Path(src), "ImageDescription") == "Dog 90%"


# ---------------------------------------------------------------------
# write_exif_predictions: copy vs overwrite mode
# ---------------------------------------------------------------------


@pytestmark_exiftool
def test_write_exif_copy_mode_leaves_source_untouched(db, tmp_path):
    project = make_project(db, name="exif-copy", detection_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    src = _write_jpeg(tmp_path / "src" / "IMG_A.jpg")
    file = make_file(
        db, deployment_id=dep.id, file_path=src, observation_type="animal"
    )
    make_detection(
        db, file_id=file.id, category="animal", confidence=0.9, label="dog"
    )

    target = tmp_path / "out"
    result = write_exif_predictions(db, project.id, target, mode="copy")

    # Tagged copy exists.
    tagged = target / EXIF_COPIES_SUBDIR / "IMG_A.jpg"
    assert tagged.is_file()
    assert _read_tag(tagged, "ImageDescription") == "Dog 90%"

    # Source untouched (no ImageDescription on the original).
    assert _read_tag(Path(src), "ImageDescription") is None

    assert result.written_count == 1
    assert result.mode == "copy"


@pytestmark_exiftool
def test_write_exif_overwrite_mode_modifies_source_in_place(db, tmp_path):
    project = make_project(db, name="exif-over", detection_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    src = _write_jpeg(tmp_path / "src" / "IMG_B.jpg")
    file = make_file(
        db, deployment_id=dep.id, file_path=src, observation_type="animal"
    )
    make_detection(
        db, file_id=file.id, category="animal", confidence=0.9, label="wolf"
    )

    target = tmp_path / "out"
    result = write_exif_predictions(db, project.id, target, mode="overwrite")

    # Source now carries the tag in place.
    assert _read_tag(Path(src), "ImageDescription") == "Wolf 90%"
    # No tagged copy folder was created in overwrite mode.
    assert not (target / EXIF_COPIES_SUBDIR).exists()

    assert result.written_count == 1
    assert result.mode == "overwrite"


def test_write_exif_skips_videos(db, tmp_path):
    project = make_project(db, name="exif-video")
    dep = make_deployment(db, project_id=project.id)
    # Make the path exist on disk so the skipped-video branch runs
    # rather than the missing-source one.
    video_path = tmp_path / "VID.mp4"
    video_path.write_bytes(b"x")
    make_file(
        db,
        deployment_id=dep.id,
        file_path=str(video_path),
        file_type="video",
        file_format="mp4",
        observation_type="animal",
    )

    target = tmp_path / "out"
    result = write_exif_predictions(db, project.id, target, mode="copy")

    assert result.skipped_video == 1
    assert result.written_count == 0


def test_write_exif_skips_files_without_detections(db, tmp_path):
    project = make_project(db, name="exif-blank")
    dep = make_deployment(db, project_id=project.id)
    make_file(
        db,
        deployment_id=dep.id,
        file_path=_write_jpeg(tmp_path / "src" / "IMG.jpg"),
        observation_type="blank",
    )

    target = tmp_path / "out"
    result = write_exif_predictions(db, project.id, target, mode="copy")

    assert result.skipped_no_detections == 1
    assert result.written_count == 0


def test_write_exif_unknown_project_raises(db, tmp_path):
    with pytest.raises(ValueError, match="not found"):
        write_exif_predictions(db, "no-such", tmp_path / "out")
