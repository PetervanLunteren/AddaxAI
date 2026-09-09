"""A column selected in SQL reaches the wire.

`similarity_script` builds the Labels grid's rows in three steps: the
SELECT list, `_row_to_meta`, then `_build_summary`. Miss the middle one
and the field is null on the wire with nothing to show for it, no error
and no failing test. That is not hypothetical: `frame_number` was
selected and dropped there for months, and the detail view carries a
comment working around it.

The grid runs in a subprocess with no `app.*` on its path, so the module
is loaded by file path here, the way the app runs it.
"""

import importlib.util
import sqlite3
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "app" / "ml" / "inference" / "similarity_script.py"
)


@pytest.fixture(scope="module")
def script():
    """The script as the subprocess sees it: its own directory first on
    the path, so `from crop_box import ...` resolves to the sibling."""
    pytest.importorskip("numpy")
    import sys

    sys.path.insert(0, str(_SCRIPT.parent))
    try:
        spec = importlib.util.spec_from_file_location("similarity_script", _SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(_SCRIPT.parent))


def _row(**over):
    """One sqlite3.Row with the shape `_DETECTION_COLUMNS` produces."""
    values = {
        "detection_id": "det-1", "label": "grey reef shark",
        "label_taxonomy_id": None, "label_confidence": 0.8,
        "scientific_name": None, "common_name": None,
        "confidence": 0.9, "category": "elasmobranch",
        "verified": 0, "suggestion_dismissed": 0,
        "classification_method": "machine", "file_id": "file-1",
        "frame_number": 4836, "bbox_x": 0.1, "bbox_y": 0.2,
        "bbox_width": 0.3, "bbox_height": 0.4,
        "track_id": "track-1", "track_frames": 31,
        "deployment_id": "dep-1", "captured_at_local": "2026-03-14 09:12:00",
        "width_px": 1920, "height_px": 1080,
        "file_flagged": 0, "file_favorited": 0, "site_name": "Reef A",
        "event_id": "event-1", "event_sequence": 1,
        "event_start_local": "2026-03-14 09:00:00",
    }
    values.update(over)
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cols = ", ".join(f"? AS {name}" for name in values)
    return conn.execute(f"SELECT {cols}", list(values.values())).fetchone()


def test_the_track_fields_reach_the_summary(script):
    """The three fields the opened track needs, all the way through."""
    meta = script._row_to_meta(_row())
    summary = script._build_summary("det-1", meta)

    assert summary["frame_number"] == 4836
    assert summary["track_id"] == "track-1"
    assert summary["track_frames"] == 31


def test_an_image_row_carries_no_track(script):
    """A photo has no track, and says so rather than guessing."""
    meta = script._row_to_meta(
        _row(frame_number=None, track_id=None, track_frames=None)
    )
    summary = script._build_summary("det-1", meta)

    assert summary["frame_number"] is None
    assert summary["track_id"] is None
    assert summary["track_frames"] is None


def test_every_selected_column_is_copied_into_meta(script):
    """The guard that would have caught the frame_number bug.

    Every column `_DETECTION_COLUMNS` names is either in the meta dict
    or in this list of ones the summary genuinely does not need. Adding
    a column without deciding which it is fails here.
    """
    not_needed = {
        "detection_id",  # passed separately
        "bbox_x", "bbox_y", "bbox_width", "bbox_height",  # kept, see below
        "event_sequence",  # ordering only, never rendered
    }
    row = _row()
    meta = script._row_to_meta(row)
    # bbox and size are in meta under their own names, for crop_bbox.
    for name in ("bbox_x", "bbox_y", "bbox_width", "bbox_height",
                 "width_px", "height_px"):
        assert name in meta

    selected = {
        line.split(" AS ")[-1].strip().rstrip(",")
        if " AS " in line
        else line.strip().rstrip(",").split(".")[-1]
        for line in script._DETECTION_COLUMNS.splitlines()
        if line.strip() and not line.strip().startswith("--")
        and "(SELECT" not in line and "FROM" not in line
        and "WHERE" not in line and "JOIN" not in line
    }
    selected = {name for name in selected if name.isidentifier()}
    missing = selected - set(meta) - not_needed
    assert not missing, f"selected but never copied into meta: {sorted(missing)}"
