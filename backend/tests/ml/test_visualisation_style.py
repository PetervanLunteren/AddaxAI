"""Pin the visualisation style spec.

Species colours are no longer computed here: `detection_color` reads
the project's map from `crud/label_colors.py` (tested in
`tests/api/test_label_colors.py`) and only falls back to a hash for a
label the map does not know. The category colours still mirror
`getCategoryColor` in `frontend/src/lib/detection-utils.ts`.
"""

from app.api.crud.label_colors import SPECIES_PALETTE, fallback_color
from app.ml.postprocessing_outputs._visualisation_style import (
    category_color,
    detection_color,
    render_metrics,
)


def _rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def test_category_color_known_values():
    """Canonical category palette mirrors detection-utils.ts."""
    assert category_color("animal") == (15, 96, 100)
    assert category_color("person") == (255, 137, 69)
    assert category_color("vehicle") == (113, 183, 186)
    # Unknown categories fall back to the brand "bad" red.
    assert category_color("alien") == (136, 32, 0)


def test_detection_color_reads_the_project_map():
    """A labelled detection takes the colour the map assigned, looked up
    case-insensitively, exactly like the grid does."""
    colors = {"leopard": "#17559b"}
    assert detection_color("Leopard", "animal", colors) == _rgb("#17559b")


def test_detection_color_falls_back_for_an_unknown_label():
    """A label outside the project's counting threshold is not in the
    map; it still draws, deterministically, from the same palette."""
    a = detection_color("aardvark", "animal", {})
    assert a == detection_color("AARDVARK", "animal", {})
    assert a == _rgb(fallback_color("aardvark"))
    assert fallback_color("aardvark") in SPECIES_PALETTE


def test_detection_color_prefers_label_over_category():
    """An unlabelled detection falls back to the category colour."""
    assert detection_color(None, "animal", {}) == category_color("animal")
    assert detection_color("", "person", {}) == category_color("person")


def test_render_metrics_single_font_and_no_dot():
    """The simplified pill uses one font for both lines and has no dot,
    so text starts flush with the horizontal padding (no dot offset)."""
    m = render_metrics(4000, 3000)
    # One shared font size (both pill lines use it).
    assert isinstance(m.font, int) and m.font > 0
    # No dot: text starts at the padding, not past a dot + gap.
    assert m.text_start_x == m.pad_x
