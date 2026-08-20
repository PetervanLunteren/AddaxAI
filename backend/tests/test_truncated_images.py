"""Truncated images decode instead of raising.

Camera traps write partly-finished JPEGs routinely: a battery dies
mid-write, a card is pulled, a copy is interrupted. Pillow refuses those
by default, which took out every surface that decodes pixels
(thumbnails, crops, filmstrips, annotated copies, the EXIF date read)
for a file the user can open in any photo viewer.

MegaDetector already sets this flag itself, in
`megadetector/visualization/visualization_utils.py`, and loads every
image through it. So before this change the detector could read files
the app around it could not, which is the wrong way round. Measured on
a real 2,281-file deployment: 24 files (1.1%) returned a 500 from the
thumbnail endpoint while the detector had processed them without a
single failure.

The setting is three module-level assignments in three files, one per
process that decodes pixels. Nothing links them, so these tests are what
stops one being dropped in a tidy-up.
"""

import ast
import io
from pathlib import Path

import pytest
from PIL import Image, ImageFile

INFERENCE_DIR = (
    Path(__file__).resolve().parents[1] / "app" / "ml" / "inference"
)


@pytest.fixture
def truncated_jpeg(tmp_path: Path) -> Path:
    """A JPEG with its last third cut off, like a half-written file."""
    src = tmp_path / "whole.jpg"
    Image.new("RGB", (640, 480), (90, 120, 60)).save(src, "JPEG", quality=95)
    data = src.read_bytes()
    cut = tmp_path / "truncated.jpg"
    cut.write_bytes(data[: int(len(data) * 0.66)])
    return cut


def test_the_app_decodes_a_truncated_jpeg(truncated_jpeg: Path):
    """Importing `app` is what turns this on, process-wide."""
    import app  # noqa: F401

    with Image.open(truncated_jpeg) as im:
        im.convert("RGB").load()
        assert im.size == (640, 480)


def test_pillow_would_refuse_it_without_the_setting(truncated_jpeg: Path):
    """Proves the file really is broken, so the test above is not passing
    because the fixture happens to produce a readable JPEG."""
    import app  # noqa: F401

    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        with pytest.raises(OSError):
            with Image.open(truncated_jpeg) as im:
                im.convert("RGB").load()
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = True


@pytest.mark.parametrize(
    "script", ["classification_worker.py", "embedding_script.py"]
)
def test_the_inference_subprocesses_set_it_too(script: str):
    """These run in their own conda environments and cannot import
    `app`, so each has to carry its own assignment.

    Checked by parsing rather than by running: the scripts need torch
    and cv2, which live in those environments and not in the backend
    venv, so an execution-based test would skip here and on CI and pin
    nothing at all. This asserts the statement exists at module level
    (not tucked inside a function, where it would run too late or not at
    all) and that `ImageFile` is actually imported.
    """
    tree = ast.parse((INFERENCE_DIR / script).read_text())

    imports_image_file = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "PIL"
        and any(alias.name == "ImageFile" for alias in node.names)
        for node in tree.body
    )
    assert imports_image_file, f"{script} does not import PIL.ImageFile"

    enables_it = any(
        isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and node.value.value is True
        and any(
            isinstance(t, ast.Attribute)
            and t.attr == "LOAD_TRUNCATED_IMAGES"
            and isinstance(t.value, ast.Name)
            and t.value.id == "ImageFile"
            for t in node.targets
        )
        for node in tree.body
    )
    assert enables_it, (
        f"{script} does not set ImageFile.LOAD_TRUNCATED_IMAGES = True at "
        f"module level, so truncated camera-trap files will fail there"
    )


def test_a_truncated_jpeg_still_serves_a_thumbnail(truncated_jpeg: Path):
    """The endpoint path that broke: an in-memory resize of a partly
    written file. Exercised through the helper rather than the route so
    it needs no deployment fixture."""
    from app.api.routers.files import _render_thumbnail_bytes

    data = _render_thumbnail_bytes(truncated_jpeg)
    with Image.open(io.BytesIO(data)) as thumb:
        assert thumb.format == "JPEG"
        assert thumb.width > 0
