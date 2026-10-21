"""
Open a photo the way MegaDetector does, so every crop and overlay
lands on the pixels the box coordinates describe.

MegaDetector rotates the decoded pixels for EXIF orientation 3, 6 and 8
before detecting and reports boxes on the rotated image. It leaves the
mirrored orientations (2, 4, 5, 7) and unreadable EXIF alone. Anything
that uses those boxes on the source file (classification crops,
embedding crops, drawn boxes) has to apply the same rule, or on a
rotated photo it reads the wrong part of the picture. Pillow's
``ImageOps.exif_transpose`` is the usual answer, but it also mirrors,
which the detector does not, so the table below mirrors
``megadetector.visualization.visualization_utils.open_image`` instead.

This module runs inside the model subprocesses (no ``app.*`` on the
path, Python 3.8 in the oldest env) as well as in the backend, so it
imports nothing but Pillow.
"""

from __future__ import annotations

from PIL import Image

EXIF_ORIENTATION_TAG = 274

# EXIF orientation -> degrees to rotate counter-clockwise, as MegaDetector
# applies them. Every other value, including the mirrored ones, is 0.
_ROTATION_BY_ORIENTATION = {3: 180, 6: 270, 8: 90}


# Modes MegaDetector converts to RGB before it reads the EXIF. The
# converted copy has no ``_getexif``, the read fails silently, and the
# image is not rotated. Its boxes then sit on the unrotated pixels, so
# a crop has to stay unrotated too.
_MODES_MEGADETECTOR_NEVER_ROTATES = ("L", "RGBA")


def exif_rotation(image: Image.Image) -> int:
    """Degrees MegaDetector rotates this image by before detecting (0, 90,
    180 or 270). Broken or missing EXIF counts as no rotation, as it does
    there, and so does a grayscale or RGBA source."""
    if image.mode in _MODES_MEGADETECTOR_NEVER_ROTATES:
        return 0
    try:
        orientation = image.getexif().get(EXIF_ORIENTATION_TAG)
    except Exception:  # noqa: BLE001 - corrupt EXIF is a no-rotation, not an error
        return 0
    return _ROTATION_BY_ORIENTATION.get(orientation, 0)


def upright_rgb(image: Image.Image) -> Image.Image:
    """The image as MegaDetector saw it: rotated per ``exif_rotation`` and
    converted to RGB, so every model's crop gets a 3-channel image even
    from a grayscale, palette or RGBA source."""
    rotation = exif_rotation(image)
    if rotation:
        image = image.rotate(rotation, expand=True)
    return image.convert("RGB")


def open_image_rgb(path) -> Image.Image:
    """Open ``path`` and return it upright and RGB (see ``upright_rgb``)."""
    return upright_rgb(Image.open(path))
