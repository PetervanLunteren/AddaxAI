"""Tests for the observations_csv postprocess output module.

The row schema lives in `export_crud.build_observation_rows` and has
its own coverage there. Here we only pin that this wrapper produces
the file at the right path with non-empty content matching the
existing CSV serialiser.
"""

from pathlib import Path

import pytest

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
    result = write_observations_csv(db, project.id, target)

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
    result = write_observations_csv(db, project.id, target)

    assert result.row_count == 2


def test_unknown_project_raises(db, tmp_path):
    with pytest.raises(ValueError, match="not found"):
        write_observations_csv(db, "no-such-id", tmp_path / "out")
