"""
The geometry of a detection crop: a square around the box with context
padding, and a blurred fill where the square runs off the picture.

Shared by the crop service (photo cards, cut on request), the tracking
script (track cards, cut while the video is decoded) and the two places
that tell the frontend where the box sits inside a finished crop, so
every card looks the same and every overlay lands in the same place.

The tracking script runs in the detector's own environment with no
``app.*`` on its path and loads this module by file path, like
``video_iter``; the labels script imports it by name as a sibling. Keep
it PIL only, with no app imports.
"""

from __future__ import annotations

from PIL import Image, ImageFilter

BLUR_RADIUS = 30


def compute_expanded_crop_region(
    bbox_x: float,
    bbox_y: float,
    bbox_w: float,
    bbox_h: float,
    img_w: int,
    img_h: int,
    padding: float = 0.10,
) -> tuple[int, int, int, int]:
    """Compute square crop region centered on bbox with padding.

    Returns (left, top, right, bottom) in pixel coords. Values may be
    negative or exceed image dimensions; the caller handles overflow
    with `crop_with_blur_fill`.
    """
    bx, by = bbox_x * img_w, bbox_y * img_h
    bw, bh = bbox_w * img_w, bbox_h * img_h

    max_side = max(bw, bh)
    pad = max_side * padding
    crop_side = max_side + 2 * pad

    cx, cy = bx + bw / 2, by + bh / 2
    left = cx - crop_side / 2
    top = cy - crop_side / 2

    return int(left), int(top), int(left + crop_side), int(top + crop_side)


def crop_with_blur_fill(
    img: Image.Image, left: int, top: int, right: int, bottom: int
) -> Image.Image:
    """Crop a region from the image, filling out-of-bounds areas with a
    blurred stretch of the valid part."""
    img_w, img_h = img.size
    crop_w = right - left
    crop_h = bottom - top

    # Fast path: entirely within bounds
    if left >= 0 and top >= 0 and right <= img_w and bottom <= img_h:
        return img.crop((left, top, right, bottom))

    # Clamp to valid region
    valid_left = max(0, left)
    valid_top = max(0, top)
    valid_right = min(img_w, right)
    valid_bottom = min(img_h, bottom)

    if valid_right <= valid_left or valid_bottom <= valid_top:
        return img.crop((0, 0, min(crop_w, img_w), min(crop_h, img_h)))

    # Stretch the valid region to fill the full canvas, then blur heavily.
    # This gives the overflow areas natural image colors instead of
    # replicating edge pixels (which copies black info bars on camera traps).
    valid_crop = img.crop((valid_left, valid_top, valid_right, valid_bottom))
    canvas = valid_crop.resize((crop_w, crop_h), Image.BILINEAR)
    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))

    # Paste sharp original on top
    paste_x = valid_left - left
    paste_y = valid_top - top
    canvas.paste(valid_crop, (paste_x, paste_y))

    return canvas


def bbox_within_crop(
    bbox_width: float,
    bbox_height: float,
    img_w: int | None,
    img_h: int | None,
) -> dict[str, float] | None:
    """Where the box sits inside the finished crop, as 0-1 fractions.

    The crop is the square ``compute_expanded_crop_region`` cuts, always
    centred on the box, so the box's own position in the picture does
    not come into it: only how wide and tall it is against the square.
    The frontend draws its overlay from this.

    ``None`` when the file's pixel size is unknown, which is the honest
    answer: without it the square's side cannot be worked out.
    """
    if not img_w or not img_h:
        return None
    bw = bbox_width * img_w
    bh = bbox_height * img_h
    crop_side = max(bw, bh) * 1.2  # the box plus 10% padding on each side
    if crop_side <= 0:
        return None
    return {
        "x": (crop_side - bw) / 2 / crop_side,
        "y": (crop_side - bh) / 2 / crop_side,
        "w": bw / crop_side,
        "h": bh / crop_side,
    }
