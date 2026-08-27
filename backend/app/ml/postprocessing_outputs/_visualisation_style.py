"""Shared bbox / pill-label rendering spec.

This module is the Python-side mirror of the canvas rendering used by
the verify page in the frontend. The visualised-images postprocess
output reads its colours and layout from here so a labelled JPEG
written to disk looks like what the user sees in the verify grid:
rounded bounding box outlines and rounded label pills with white
text on a dark background.

Colours: the category colours mirror ``getCategoryColor`` in
``frontend/src/lib/detection-utils.ts``. Species colours are not
computed here at all: ``app.api.crud.label_colors`` assigns them once
per project and the frontend fetches that same map, so there is one
implementation and the JPEG on disk matches the grid on screen by
construction. Callers pass the map into ``detection_color``.

Layout is NOT a fixed-pixel mirror of the frontend. The frontend
rescales its fixed 10/12px constants to screen pixels at render time
(`s = imgW / displayWidth`), so a label is always ~the same fraction
of the displayed image. A saved JPEG has no such render-time scaling,
so fixed pixels there are illegible on a multi-megapixel photo. Instead
`render_metrics(width, height)` derives every size as a fraction of the
image, tuned to roughly match the on-screen overlay's relative size.
The export uses a more opaque box stroke and pill than the live grid
because the JPEG is viewed full-size without the grid's spotlight-dim
backdrop.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from app.api.crud.label_colors import fallback_color

# ─────────────────────────────────────────────────────────────────
# Category colours — canonical map mirrors getCategoryColor() in
# frontend/src/lib/detection-utils.ts.
# ─────────────────────────────────────────────────────────────────
_CATEGORY_RGB: dict[str, tuple[int, int, int]] = {
    "animal": (15, 96, 100),  # #0f6064
    "person": (255, 137, 69),  # #ff8945
    "vehicle": (113, 183, 186),  # #71b7ba
}
_DEFAULT_CATEGORY_RGB: tuple[int, int, int] = (136, 32, 0)  # #882000


def category_color(category: str) -> tuple[int, int, int]:
    """Resolve an MD category ("animal" / "person" / "vehicle") to RGB.

    Unknown categories fall back to the brand "bad" red so the
    surfacing of unexpected category strings is visible in output.
    """
    return _CATEGORY_RGB.get(category, _DEFAULT_CATEGORY_RGB)


def detection_color(
    label: str | None,
    category: str,
    colors: Mapping[str, str],
) -> tuple[int, int, int]:
    """Pick the colour a detection's box and pill dot use.

    Species colour wins when a label exists, exactly like
    ``getDetectionColor`` in the frontend. ``colors`` is the project's
    map from ``assign_label_colors``; a label it does not know (one
    that passes the media threshold but not the project's counting
    threshold) takes the deterministic fallback. Unlabelled
    detections use the category colour.
    """
    if label:
        return _hex_to_rgb(
            colors.get(label.strip().lower()) or fallback_color(label)
        )
    return category_color(category)


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


# ─────────────────────────────────────────────────────────────────
# Contrast / fill constants (resolution-independent).
# ─────────────────────────────────────────────────────────────────
STROKE_ALPHA = 235  # box outline alpha — near-solid so it reads on busy scenes
PILL_BG_RGBA = (0, 0, 0, 175)  # pill background, more solid than the live grid

WHITE = (255, 255, 255)


# ─────────────────────────────────────────────────────────────────
# Image-proportional layout. Every size is a fraction of the image so
# the result looks the same on a 1 MP or a 24 MP photo. Floors keep
# small images legible. Ratios are tuned to sit close to the frontend
# overlay's on-screen relative size. Both pill lines share one font.
# ─────────────────────────────────────────────────────────────────
_FONT_FRACTION = 0.014  # text line height as a fraction of image height
_FONT_MIN = 12
_STROKE_FRACTION = 0.0040  # box stroke as a fraction of the long side
_STROKE_MIN = 3
_RADIUS_RATIO = 2.2  # corner radius relative to stroke width
_PAD_X_RATIO = 0.55  # pill horizontal padding relative to the font
_PAD_Y_RATIO = 0.45  # pill vertical padding relative to the font
_LINE_GAP_RATIO = 0.18  # gap between the two text lines


@dataclass(frozen=True)
class RenderMetrics:
    """Pixel sizes for one image, derived from its dimensions."""

    font: int
    stroke: int
    radius: int
    pad_x: int
    pad_y: int
    line_gap: int
    text_start_x: int


def render_metrics(width: int, height: int) -> RenderMetrics:
    """Resolution-aware layout sizes for an image of ``width`` x ``height``."""
    long_side = max(width, height)
    font = max(_FONT_MIN, round(height * _FONT_FRACTION))
    stroke = max(_STROKE_MIN, round(long_side * _STROKE_FRACTION))
    pad_x = round(font * _PAD_X_RATIO)
    return RenderMetrics(
        font=font,
        stroke=stroke,
        radius=round(stroke * _RADIUS_RATIO),
        pad_x=pad_x,
        pad_y=round(font * _PAD_Y_RATIO),
        line_gap=round(font * _LINE_GAP_RATIO),
        text_start_x=pad_x,
    )
