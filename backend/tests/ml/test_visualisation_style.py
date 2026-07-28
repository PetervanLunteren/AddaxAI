"""Pin the visualisation style spec.

Two-way contract test: the Python species-colour algorithm must match
the TypeScript reference in `frontend/src/utils/species-colors.ts`.
If anyone tweaks one and forgets the other, the verify grid and the
exported JPEGs drift apart and start looking different.

Spec recap:
- FNV-1a hash of the lowercased label string
- 32-bit unsigned result mod 1000, divided by 1000 → 0..1 position
- RGB-space lerp between #0f6064 and #f9f871

The reference values below are computed by running the same algorithm
on the same inputs in the TypeScript file. If the algorithm has
genuinely changed, recompute and update both sides together.
"""

import pytest

from app.ml.postprocessing_outputs._visualisation_style import (
    _fnv1a_position,
    category_color,
    detection_color,
    render_metrics,
    species_color,
)


# Reference positions computed by running the canonical FNV-1a
# algorithm (`hashToPosition` in frontend/src/utils/species-colors.ts)
# on the same input strings. Any drift between the two
# implementations of the same spec must show up here.
@pytest.mark.parametrize(
    "label,expected_position",
    [
        ("dog", 0.817),
        ("cat", 0.031),
        ("leopard", 0.370),
        ("aardvark", 0.179),
        ("elephant", 0.860),
    ],
)
def test_fnv1a_position_matches_reference(label, expected_position):
    assert _fnv1a_position(label) == pytest.approx(expected_position)


def test_species_color_is_deterministic():
    """Same input always gives the same colour. No global state."""
    assert species_color("leopard") == species_color("leopard")
    assert species_color("leopard") == species_color("LEOPARD")  # case-insensitive


def test_species_color_differs_per_label():
    a = species_color("dog")
    b = species_color("cat")
    assert a != b


def test_species_color_endpoints():
    """A label that hashes to the extremes should land near the
    gradient endpoints. Use synthetic labels chosen so the FNV-1a
    output mod 1000 lands at 0 (close to #0f6064) and ~999 (close
    to #f9f871)."""
    # We can't pick exact hash-zero labels by inspection, but we can
    # at least assert the gradient covers both extremes by checking
    # that some labels in a sample produce colors closer to either
    # endpoint.
    samples = ["aardvark", "bear", "cat", "dog", "elephant", "fox", "gorilla"]
    rs = [species_color(s)[0] for s in samples]
    # Gradient goes from r=15 (dark teal) to r=249 (light yellow).
    # A reasonable random spread should cover both halves.
    assert min(rs) < 100, "expected some labels near the teal end"
    assert max(rs) > 150, "expected some labels near the yellow end"


def test_category_color_known_values():
    """Canonical category palette mirrors detection-utils.ts."""
    assert category_color("animal") == (15, 96, 100)
    assert category_color("person") == (255, 137, 69)
    assert category_color("vehicle") == (113, 183, 186)
    # Unknown categories fall back to the brand "bad" red.
    assert category_color("alien") == (136, 32, 0)


def test_detection_color_prefers_label_over_category():
    """A labelled detection uses its species colour; an unlabelled
    one falls back to the category colour."""
    labelled = detection_color("dog", "animal")
    unlabelled = detection_color(None, "animal")
    assert labelled == species_color("dog")
    assert unlabelled == category_color("animal")


def test_render_metrics_single_font_and_no_dot():
    """The simplified pill uses one font for both lines and has no dot,
    so text starts flush with the horizontal padding (no dot offset)."""
    m = render_metrics(4000, 3000)
    # One shared font size (both pill lines use it).
    assert isinstance(m.font, int) and m.font > 0
    # No dot: text starts at the padding, not past a dot + gap.
    assert m.text_start_x == m.pad_x
