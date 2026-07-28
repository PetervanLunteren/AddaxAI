"""Tests for the OutputContext record-keeping object.

The context is the dumb glue between ``separate_folders`` (which
records placements) and the downstream modules (``annotated_copies``,
CSV / XLSX) that look the same placements up. The behaviour to pin:

- ``record`` appends in order, supporting multi-placement files.
- ``resolved_for`` returns a fresh list when entries exist, ``None``
  otherwise so callers can distinguish "separation didn't place this"
  from "separation placed this nowhere".
"""

from pathlib import Path

from app.ml.postprocessing_outputs._output_context import OutputContext


def test_record_appends_in_order():
    ctx = OutputContext(output_root=Path("/out"))
    ctx.record("file-1", Path("/out/dog/img.jpg"))
    ctx.record("file-1", Path("/out/wolf/img.jpg"))
    assert ctx.resolved_paths["file-1"] == [
        Path("/out/dog/img.jpg"),
        Path("/out/wolf/img.jpg"),
    ]


def test_resolved_for_returns_none_when_unrecorded():
    ctx = OutputContext(output_root=Path("/out"))
    assert ctx.resolved_for("never-seen") is None


def test_resolved_for_returns_copy_not_reference():
    """The list returned to callers must not let them mutate the
    context's internal record by accident."""
    ctx = OutputContext(output_root=Path("/out"))
    ctx.record("file-1", Path("/out/dog/img.jpg"))
    snapshot = ctx.resolved_for("file-1")
    assert snapshot == [Path("/out/dog/img.jpg")]
    snapshot.append(Path("/out/wolf/img.jpg"))  # mutate the copy
    # Internal state untouched.
    assert ctx.resolved_paths["file-1"] == [Path("/out/dog/img.jpg")]


def test_resolved_for_empty_list_treated_as_none():
    """If a file_id were ever recorded with an empty list (defensive),
    ``resolved_for`` would still report None. Callers branch on
    truthiness, so the contract is "None means no resolved
    destinations, period"."""
    ctx = OutputContext(output_root=Path("/out"))
    ctx.resolved_paths["file-1"] = []
    assert ctx.resolved_for("file-1") is None
