"""A previous run's output folder must never be re-ingested as input.

The folder-run save step writes separated / annotated copies into an
output folder (default: a subfolder of the source) and drops an
OUTPUT_DIR_MARKER there. Both the preview scan and the analysis worker's
input enumeration must skip any folder carrying that marker — otherwise a
re-run reprocesses the prior output (the copies have no EXIF, so they
surface as "no capture timestamp", and missing ones error out).

Regression test: the preview scanner and the worker scanners had drifted
(only the preview skipped output folders), which let output media get
reprocessed.
"""

from PIL import Image

from app.services.folder_scanner import OUTPUT_DIR_MARKER, scan_folder
from app.workers.detection_worker import (
    scan_folder_for_images,
    scan_folder_for_videos,
)


def _write_jpeg(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (1, 2, 3)).save(path, format="JPEG")


def _make_tree(tmp_path):
    """Source folder with one real image + an AddaxAI output folder holding
    a separated copy. The output folder carries the marker."""
    _write_jpeg(tmp_path / "IMG_0001.jpg")
    output = tmp_path / "AddaxAI-output"
    (output).mkdir()
    (output / OUTPUT_DIR_MARKER).touch()
    _write_jpeg(output / "mammalia" / "REC0028.jpg")
    (tmp_path / "VID_0001.mp4").write_bytes(b"\x00")  # bare file, not real video
    return tmp_path


def test_worker_image_scan_skips_output_folder(tmp_path):
    _make_tree(tmp_path)
    found = scan_folder_for_images(tmp_path)
    names = {p.name for p in found}
    assert "IMG_0001.jpg" in names
    assert "REC0028.jpg" not in names  # the separated output copy is excluded


def test_worker_video_scan_skips_output_folder(tmp_path):
    _make_tree(tmp_path)
    # Put a (bare) video inside the output folder too.
    (tmp_path / "AddaxAI-output" / "VID_OUT.mp4").write_bytes(b"\x00")
    found = scan_folder_for_videos(tmp_path)
    names = {p.name for p in found}
    assert "VID_OUT.mp4" not in names


def test_preview_scan_skips_output_folder(tmp_path):
    _make_tree(tmp_path)
    preview = scan_folder(str(tmp_path))
    # Only the one real source image is counted, not the output copy.
    assert preview["image_count"] == 1
