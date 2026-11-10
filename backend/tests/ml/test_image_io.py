"""The shared photo opener follows MegaDetector's EXIF rule exactly:
rotate for orientation 3, 6 and 8, leave everything else alone, so a
crop taken with a MegaDetector box lands on the same pixels."""

from pathlib import Path

import pytest
from PIL import Image

from app.ml.inference.image_io import (
    EXIF_ORIENTATION_TAG,
    exif_rotation,
    open_image_rgb,
)

# Four solid quadrants: top-left red, top-right green, bottom-left blue,
# bottom-right white. Big enough that JPEG blur does not reach the
# sample points.
QUADRANTS = {
    (0, 0): (255, 0, 0),
    (1, 0): (0, 255, 0),
    (0, 1): (0, 0, 255),
    (1, 1): (255, 255, 255),
}
W, H = 160, 96


def _upright() -> Image.Image:
    img = Image.new("RGB", (W, H))
    for (qx, qy), color in QUADRANTS.items():
        img.paste(color, (qx * W // 2, qy * H // 2, (qx + 1) * W // 2, (qy + 1) * H // 2))
    return img


def _save(path: Path, image: Image.Image, orientation: int | None) -> Path:
    exif = Image.Exif()
    if orientation is not None:
        exif[EXIF_ORIENTATION_TAG] = orientation
    image.save(path, format="JPEG", quality=95, exif=exif)
    return path


def _quadrant_colors(image: Image.Image) -> dict[tuple[int, int], tuple[int, ...]]:
    w, h = image.size
    return {
        (qx, qy): image.getpixel((qx * w // 2 + w // 4, qy * h // 2 + h // 4))
        for (qx, qy) in QUADRANTS
    }


def _close(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    return all(abs(x - y) <= 12 for x, y in zip(a, b, strict=True))


@pytest.mark.parametrize(
    ("orientation", "degrees"),
    [(None, 0), (1, 0), (2, 0), (3, 180), (4, 0), (5, 0), (6, 270), (7, 0), (8, 90)],
)
def test_rotation_matches_megadetector_table(tmp_path, orientation, degrees):
    """3, 6 and 8 rotate; mirrored orientations and no tag do not."""
    path = _save(tmp_path / "f.jpg", _upright(), orientation)
    with Image.open(path) as img:
        assert exif_rotation(img) == degrees


@pytest.mark.parametrize("orientation", [3, 6, 8])
def test_rotated_file_opens_as_the_upright_picture(tmp_path, orientation):
    """A photo stored on its side with the matching orientation tag comes
    back as the picture the camera saw, so a MegaDetector box crops the
    same pixels as on the upright file."""
    upright = _upright()
    # Store the pixels turned the opposite way, so applying the EXIF
    # rotation brings them back.
    stored = upright.rotate(-{3: 180, 6: 270, 8: 90}[orientation], expand=True)
    path = _save(tmp_path / "rot.jpg", stored, orientation)

    opened = open_image_rgb(path)

    assert opened.size == upright.size
    assert opened.mode == "RGB"
    for quadrant, color in _quadrant_colors(upright).items():
        assert _close(_quadrant_colors(opened)[quadrant], color), quadrant


def test_mirrored_orientation_is_left_alone(tmp_path):
    """MegaDetector does not mirror, so neither does the opener: the
    boxes refer to the pixels as stored."""
    path = _save(tmp_path / "mirror.jpg", _upright(), 2)
    opened = open_image_rgb(path)
    assert _close(_quadrant_colors(opened)[(0, 0)], QUADRANTS[(0, 0)])


@pytest.mark.parametrize("mode", ["L", "RGBA"])
def test_grayscale_and_rgba_are_not_rotated(tmp_path, mode):
    """MegaDetector converts these modes to RGB before it reads the EXIF,
    which loses the tag, so it never rotates them and its boxes refer to
    the stored pixels. The opener follows, or a grayscale night photo on
    its side would be cropped from the wrong place (seen as "blank 0.998"
    against the official "domestic dog 0.868")."""
    path = tmp_path / f"{mode}.{'png' if mode == 'RGBA' else 'jpg'}"
    exif = Image.Exif()
    exif[EXIF_ORIENTATION_TAG] = 6
    stored = _upright().convert(mode).rotate(-270, expand=True)
    stored.save(path, exif=exif.tobytes())
    with Image.open(path) as img:
        assert img.mode == mode
        assert exif_rotation(img) == 0
    opened = open_image_rgb(path)
    assert opened.size == stored.size
    assert opened.mode == "RGB"


def test_unreadable_exif_means_no_rotation(tmp_path):
    """A PNG has no EXIF; a palette image still comes out as RGB."""
    path = tmp_path / "plain.png"
    _upright().convert("P").save(path)
    opened = open_image_rgb(path)
    assert opened.mode == "RGB"
    assert opened.size == (W, H)
