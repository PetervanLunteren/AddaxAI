"""Shared bbox / pill-label rendering spec.

This module is the Python-side mirror of the canvas rendering used by
the verify page in the frontend. The visualised-images postprocess
output reads its constants and helpers from here so a labelled JPEG
written to disk looks the same as what the user sees in the verify
grid: rounded bounding box outlines at 50% opacity, rounded label
pills with a coloured species dot, white text on a dim background.

Canonical references (keep in sync — single spec, two implementations):

- frontend/src/lib/detection-overlay.ts  (layout + style constants)
- frontend/src/lib/detection-utils.ts    (category colours)
- frontend/src/utils/species-colors.ts   (species colour algorithm)

The species colour is a deterministic FNV-1a hash of the lowercased
label string mapped onto an RGB-interpolated gradient between
`#0f6064` and `#f9f871`. Same algorithm as `chroma.scale([...])` with
default RGB interpolation in the frontend, so a label that looks
"teal-green" in the verify grid also looks "teal-green" in the
exported JPEG.
"""

from __future__ import annotations

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
# Layout constants. Mirror frontend/src/lib/detection-overlay.ts.
# Keep tuples / numbers literally equal so the rendered pills look
# the same width regardless of where they are drawn.
# ─────────────────────────────────────────────────────────────────
PILL_PAD_X = 6
PILL_PAD_Y = 4
DOT_R = 4
DOT_GAP = 5
LINE_GAP = 2
FONT_SM = 10
FONT_LG = 12
TEXT_START_X = PILL_PAD_X + DOT_R * 2 + DOT_GAP  # 19

BBOX_STROKE_WIDTH = 2
BBOX_CORNER_RADIUS = 4
BBOX_OPACITY = 0.5  # 0..1 alpha, blended over the source image
PILL_BG_RGBA = (0, 0, 0, 128)  # rgba(0,0,0,0.5) — 128/255 ≈ 0.5

WHITE = (255, 255, 255)
WHITE_DIM = (255, 255, 255, 179)  # rgba(255,255,255,0.7) for small text
