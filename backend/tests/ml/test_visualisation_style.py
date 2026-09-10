"""Pin the visualisation style spec.

Species colours are no longer computed here: `detection_color` reads
the project's map from `crud/label_colors.py` (tested in
`tests/api/test_label_colors.py`) and only falls back to a hash for a
class the map does not know. There is no category table here any more:
a detector's category is a class like any other and is looked up in the
same map.
"""

from app.api.crud.label_colors import SPECIES_PALETTE, fallback_color
from app.ml.postprocessing_outputs._visualisation_style import (
    detection_color,
    render_metrics,
)


def _rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def test_an_unlabelled_box_takes_its_category_from_the_same_map():
    """A detector's category is a class: it is looked up in the project
    map exactly like a species, whichever detector named it. This is the
    case that used to draw the brand red in the JPEG and teal on
    screen."""
    colors = {"elasmobranch": "#73c076", "animal": "#882000"}
    assert detection_color(None, "elasmobranch", colors) == _rgb("#73c076")
    assert detection_color(None, "animal", colors) == _rgb("#882000")
    # Case-insensitive, like every other lookup.
    assert detection_color(None, "Animal", colors) == _rgb("#882000")


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
    """The label names the class when there is one, the category when
    there is not; an empty label counts as none."""
    colors = {"leopard": "#17559b", "animal": "#882000"}
    assert detection_color("leopard", "animal", colors) == _rgb("#17559b")
    assert detection_color(None, "animal", colors) == _rgb("#882000")
    assert detection_color("", "animal", colors) == _rgb("#882000")


def test_render_metrics_single_font_and_no_dot():
    """The simplified pill uses one font for both lines and has no dot,
    so text starts flush with the horizontal padding (no dot offset)."""
    m = render_metrics(4000, 3000)
    # One shared font size (both pill lines use it).
    assert isinstance(m.font, int) and m.font > 0
    # No dot: text starts at the padding, not past a dot + gap.
    assert m.text_start_x == m.pad_x
