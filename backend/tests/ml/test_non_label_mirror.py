"""The frontend's copy of the "nothing here" labels must match ours.

`NON_LABEL_CLASSES` decides three things that have to agree: which
detections the ingest refuses to load, which ones every count and export
treats as not-a-detection, and which ones the UI refuses to draw a box
around. The first two live in `app/ml/label_exclusion.py`; the third
lives in `frontend/src/lib/detection-utils.ts`, because the drawing
decision is made in the browser off an already-fetched payload and a
round trip to ask the server would be absurd.

Two copies of a list with a "keep in sync" comment is exactly the drift
this repo has been bitten by before, so it gets a test rather than a
comment. Same idea as `test_visualisation_style.py`, which pins the
species-colour algorithm across the same boundary.

Reading the TypeScript with a regex is deliberate. The alternative is
generating one file from the other at build time, which buys nothing
here: the set changes about once a year, and a generator is a build step
to maintain plus a generated file nobody may hand-edit.
"""

import re
from pathlib import Path

import pytest

from app.ml.label_exclusion import NON_LABEL_CLASSES

_TS_FILE = (
    Path(__file__).resolve().parents[3]
    / "frontend"
    / "src"
    / "lib"
    / "detection-utils.ts"
)

_DECLARATION = re.compile(
    r"export const NON_LABEL_CLASSES = new Set\(\[(.*?)\]\)",
    re.DOTALL,
)


def _frontend_non_label_classes() -> set[str]:
    """The set as the browser sees it."""
    if not _TS_FILE.is_file():
        pytest.skip(f"frontend source not present at {_TS_FILE}")
    match = _DECLARATION.search(_TS_FILE.read_text())
    assert match, (
        f"Could not find `export const NON_LABEL_CLASSES = new Set([...])` "
        f"in {_TS_FILE}. If it was renamed or reshaped, update this test "
        f"rather than deleting it."
    )
    return set(re.findall(r'"([^"]+)"', match.group(1)))


def test_the_frontend_knows_the_same_non_label_classes():
    assert _frontend_non_label_classes() == set(NON_LABEL_CLASSES), (
        "The two copies of NON_LABEL_CLASSES have drifted. A label only "
        "the backend knows is stripped from counts while the UI still "
        "draws a box around it; a label only the frontend knows is drawn "
        "nowhere while it still counts."
    )


def test_the_classes_are_lowercase_on_both_sides():
    """Both sides match with `.lower()` / `.toLowerCase()`.

    An upper-case entry in either set therefore matches nothing at all,
    silently, which is worse than a missing entry because it looks
    present.
    """
    for name in set(NON_LABEL_CLASSES) | _frontend_non_label_classes():
        assert name == name.lower(), f"{name!r} must be lower-case"
