"""Tests for the observations_csv postprocess output module.

The row schema lives in ``export_crud.build_observation_rows`` and has
its own coverage there. Here we pin that this wrapper writes the file
at the right path, includes the ``relative_path`` column populated
from ``OutputContext.resolved_paths``, and falls back to a blank
column when separation did not run.
"""

from pathlib import Path

import pytest

from app.ml.postprocessing_outputs._output_context import OutputContext
from app.ml.postprocessing_outputs.observations_csv import (
    CSV_FILENAME,
    write_observations_csv,
)
from tests.conftest import (
    make_deployment,
    make_detection,
    make_file,
    make_project,
)


def _write_placeholder(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return str(path)


def _ctx(output_root: Path) -> OutputContext:
    return OutputContext(output_root=output_root)


def test_writes_file_at_canonical_path(db, tmp_path):
    project = make_project(db, name="csv-basic")
    dep = make_deployment(db, project_id=project.id)
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=_write_placeholder(tmp_path / "src" / "IMG.jpg"),
        observation_type="animal",
    )
    make_detection(
        db,
        file_id=file.id,
        category="animal",
        confidence=0.9,
        label="dog",
        bbox_x=0.1,
        bbox_y=0.1,
        bbox_width=0.2,
        bbox_height=0.2,
    )

    target = tmp_path / "out"
    result = write_observations_csv(db, project.id, _ctx(target))

    assert result.output_path.endswith(CSV_FILENAME)
    assert (target / CSV_FILENAME).is_file()
    assert (target / CSV_FILENAME).stat().st_size > 0


def test_row_count_matches_observation_rows(db, tmp_path):
    """Two detections on two files should give exactly two rows."""
    project = make_project(db, name="csv-rows")
    dep = make_deployment(db, project_id=project.id)
    for n, label in enumerate(["dog", "cat"]):
        file = make_file(
            db,
            deployment_id=dep.id,
            file_path=_write_placeholder(tmp_path / "src" / f"IMG_{n}.jpg"),
            observation_type="animal",
        )
        make_detection(
            db,
            file_id=file.id,
            category="animal",
            confidence=0.9,
            label=label,
            bbox_x=0.1,
            bbox_y=0.1,
            bbox_width=0.2,
            bbox_height=0.2,
        )

    target = tmp_path / "out"
    result = write_observations_csv(db, project.id, _ctx(target))

    assert result.row_count == 2


def test_relative_path_column_blank_without_separation(db, tmp_path):
    """When the context has no resolved paths, the new relative_path
    column sits empty for every row."""
    project = make_project(db, name="csv-relpath-empty")
    dep = make_deployment(db, project_id=project.id)
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=_write_placeholder(tmp_path / "src" / "IMG.jpg"),
        observation_type="animal",
    )
    make_detection(
        db,
        file_id=file.id,
        category="animal",
        confidence=0.9,
        label="dog",
        bbox_x=0.1,
        bbox_y=0.1,
        bbox_width=0.2,
        bbox_height=0.2,
    )

    target = tmp_path / "out"
    write_observations_csv(db, project.id, _ctx(target))

    csv = (target / CSV_FILENAME).read_text()
    header_line, *data_lines = csv.splitlines()
    headers = header_line.split(",")
    assert "relative_path" in headers
    rel_idx = headers.index("relative_path")
    for row in data_lines:
        # CSV cells are comma-separated; the relative_path column is
        # blank when separation did not run, so a strict empty cell.
        assert row.split(",")[rel_idx] == ""


def test_relative_path_column_reflects_resolved_paths(db, tmp_path):
    """When ``ctx.resolved_paths`` is populated (separation ran), the
    new column holds a forward-slash path relative to output_root."""
    project = make_project(db, name="csv-relpath-set")
    dep = make_deployment(db, project_id=project.id)
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=_write_placeholder(tmp_path / "src" / "IMG.jpg"),
        observation_type="animal",
    )
    make_detection(
        db,
        file_id=file.id,
        category="animal",
        confidence=0.9,
        label="dog",
        bbox_x=0.1,
        bbox_y=0.1,
        bbox_width=0.2,
        bbox_height=0.2,
    )

    target = tmp_path / "out"
    ctx = _ctx(target)
    # Simulate the separation having placed the file at out/dog/IMG.jpg.
    ctx.record(file.id, target / "dog" / "IMG.jpg")
    write_observations_csv(db, project.id, ctx)

    csv = (target / CSV_FILENAME).read_text()
    header_line, *data_lines = csv.splitlines()
    headers = header_line.split(",")
    rel_idx = headers.index("relative_path")
    rel_values = {row.split(",")[rel_idx] for row in data_lines}
    assert rel_values == {"dog/IMG.jpg"}


def test_unknown_project_raises(db, tmp_path):
    with pytest.raises(ValueError, match="not found"):
        write_observations_csv(db, "no-such-id", _ctx(tmp_path / "out"))
