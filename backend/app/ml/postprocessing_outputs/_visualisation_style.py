"""Shared bbox / pill-label rendering spec.

This module is the Python-side mirror of the canvas rendering used by
the verify page in the frontend. The visualised-images postprocess
output reads its colours and layout from here so a labelled JPEG
written to disk looks like what the user sees in the verify grid:
rounded bounding box outlines and rounded label pills with white
text on a dark background.

Canonical references (keep the COLOUR algorithm in sync — single spec,
two implementations):

- frontend/src/lib/detection-utils.ts    (category colours)
- frontend/src/utils/species-colors.ts   (species colour algorithm)

The species colour is a deterministic FNV-1a hash of the lowercased
label string mapped onto an RGB-interpolated gradient between
`#0f6064` and `#f9f871`. Same algorithm as `chroma.scale([...])` with
default RGB interpolation in the frontend, so a label that looks
"teal-green" in the verify grid also looks "teal-green" in the
exported JPEG.

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

from dataclasses import dataclass

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


# ─────────────────────────────────────────────────────────────────
# Species gradient endpoints. Mirror `chroma.scale(['#0f6064',
# '#f9f871'])` in frontend/src/utils/species-colors.ts. chroma.scale
# defaults to RGB interpolation, so the lerp here is plain componentwise.
# ─────────────────────────────────────────────────────────────────
_SPECIES_GRADIENT_FROM: tuple[int, int, int] = (15, 96, 100)  # #0f6064
_SPECIES_GRADIENT_TO: tuple[int, int, int] = (249, 248, 113)  # #f9f871


def _fnv1a_position(text: str) -> float:
    """FNV-1a hash of `text` mapped to a 0..1 gradient position.

    Mirrors `hashToPosition` in species-colors.ts byte-for-byte: same
    FNV offset basis and prime, same mod 1000 / 1000 step, same
    32-bit unsigned right-shift before mod. Different input
    characters here would produce a different colour in the verify
    grid, so this MUST stay matched.
    """
    hash_value = 2166136261
    for ch in text:
        hash_value ^= ord(ch)
        # Imul-equivalent multiplication, truncated to 32 bits.
        hash_value = (hash_value * 16777619) & 0xFFFFFFFF
    return (hash_value % 1000) / 1000


def _lerp_rgb(
    a: tuple[int, int, int],
    b: tuple[int, int, int],
    t: float,
) -> tuple[int, int, int]:
    """Componentwise RGB lerp clamped to 0..255 integers."""
    return (
        int(round(a[0] + (b[0] - a[0]) * t)),
        int(round(a[1] + (b[1] - a[1]) * t)),
        int(round(a[2] + (b[2] - a[2]) * t)),
    )


def species_color(label: str) -> tuple[int, int, int]:
    """Resolve a species label to its RGB colour.

    Same answer the verify UI would give: hash the lowercased label,
    interpolate between the two gradient endpoints. Deterministic and
    stateless — no project context required.
    """
    position = _fnv1a_position(label.strip().lower())
    return _lerp_rgb(_SPECIES_GRADIENT_FROM, _SPECIES_GRADIENT_TO, position)


def category_color(category: str) -> tuple[int, int, int]:
    """Resolve an MD category ("animal" / "person" / "vehicle") to RGB.

    Unknown categories fall back to the brand "bad" red so the
    surfacing of unexpected category strings is visible in output.
    """
    return _CATEGORY_RGB.get(category, _DEFAULT_CATEGORY_RGB)


def detection_color(
    label: str | None, category: str
) -> tuple[int, int, int]:
    """Pick the colour a detection's box and pill dot use.

    Species colour wins when a label exists, exactly like
    `getDetectionColor` in the frontend. Falls back to the category
    colour for unlabelled detections (typical when a project runs
    detection-only).
    """
    if label:
        return species_color(label)
    return category_color(category)


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
