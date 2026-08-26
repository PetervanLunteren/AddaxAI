"""Tests for the folder-run column policy.

``folder_run_table`` is the only thing standing between the shared
export builders and the folder-run CSV / XLSX writers, so it gets
covered on its own: it is a pure function and the writers only need to
prove they call it.
"""

from app.ml.postprocessing_outputs._table_columns import (
    OMITTED_COLUMNS,
    folder_run_table,
)


def test_drops_deployment_id_and_notes():
    headers = ["file_id", "deployment_id", "event_id", "file_type", "notes"]
    rows = [["f1", "dep1", "e1", "image", ""]]

    out_headers, out_rows = folder_run_table(headers, rows)

    assert out_headers == ["file_id", "event_id", "file_type"]
    assert out_rows == [["f1", "e1", "image"]]


def test_omitted_columns_is_the_documented_set():
    assert OMITTED_COLUMNS == {
        "deployment_id", "notes", "n_events", "n_individuals",
    }


def test_keeps_event_id():
    """event_id survives the trim: it is the only column saying which
    files share a burst."""
    headers = ["file_id", "event_id", "deployment_id"]
    rows = [["f1", "e1", "d1"]]

    out_headers, out_rows = folder_run_table(headers, rows)

    assert out_headers == ["file_id", "event_id"]
    assert out_rows == [["f1", "e1"]]


def test_does_not_mutate_input_rows():
    headers = ["file_id", "notes"]
    rows = [["f1", ""]]

    folder_run_table(headers, rows)

    assert rows == [["f1", ""]]


def test_table_without_omitted_columns_passes_through():
    headers = ["detection_id", "bbox_x"]
    rows = [["d1", 0.5]]

    out_headers, out_rows = folder_run_table(headers, rows)

    assert out_headers == headers
    assert out_rows == rows


def test_preserves_column_order():
    headers = ["a", "deployment_id", "b", "notes", "c"]
    rows = [[1, 2, 3, 4, 5]]

    out_headers, out_rows = folder_run_table(headers, rows)

    assert out_headers == ["a", "b", "c"]
    assert out_rows == [[1, 3, 5]]
