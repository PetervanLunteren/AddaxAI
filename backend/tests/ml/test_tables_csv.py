"""Tests for the tables_csv postprocess output module.

The row schemas live in the ``export_crud`` builders and have their own
coverage there. Here we pin that this wrapper writes all three files
(``files.csv`` + ``detections.csv`` + ``counts.csv``) at the right paths,
and that ``relative_path`` on the files table is the file's path under its
deployment's source folder.
"""

from pathlib import Path

import pytest

from app.ml.postprocessing_outputs.tables_csv import (
    DETECTIONS_FILENAME,
    FILES_FILENAME,
    write_tables_csv,
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



def test_writes_both_files_at_canonical_paths(db, tmp_path):
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
    result = write_tables_csv(db, project.id, target)

    # Files + detections only: a folder run has no deployments or
    # counts tables (ecological interpretation lives in projects mode).
    assert (target / FILES_FILENAME).is_file()
    assert (target / DETECTIONS_FILENAME).is_file()
    assert (target / DETECTIONS_FILENAME).stat().st_size > 0
    assert len(result.output_paths) == 2
    assert not (target / "addaxai-deployments.csv").exists()
    assert not (target / "addaxai-counts.csv").exists()


def test_row_count_totals_all_tables(db, tmp_path):
    """Two files, each with one detection. Total row_count =
    2 files + 2 detections."""
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
    result = write_tables_csv(db, project.id, target)

    assert result.row_count == 4


def test_relative_path_is_relative_to_deployment_folder(db, tmp_path):
    """relative_path on the files table is the file's path under its
    deployment's source folder."""
    project = make_project(db, name="csv-relpath")
    dep = make_deployment(
        db, project_id=project.id, folder_path=str(tmp_path / "CameraA"),
    )
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=str(tmp_path / "CameraA" / "sub" / "IMG.jpg"),
        observation_type="animal",
    )
    make_detection(
        db, file_id=file.id, category="animal", confidence=0.9, label="dog",
        bbox_x=0.1, bbox_y=0.1, bbox_width=0.2, bbox_height=0.2,
    )

    target = tmp_path / "out"
    write_tables_csv(db, project.id, target)

    csv = (target / FILES_FILENAME).read_text()
    header_line, *data_lines = csv.splitlines()
    headers = header_line.split(",")
    rel_idx = headers.index("relative_path")
    rel_values = {row.split(",")[rel_idx] for row in data_lines}
    assert rel_values == {"sub/IMG.jpg"}


def test_relative_path_falls_back_to_filename(db, tmp_path):
    """When the deployment has no source folder, relative_path is the
    bare filename."""
    project = make_project(db, name="csv-relpath-fallback")
    dep = make_deployment(db, project_id=project.id, folder_path=None)
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=str(tmp_path / "anywhere" / "IMG.jpg"),
        observation_type="animal",
    )
    make_detection(
        db, file_id=file.id, category="animal", confidence=0.9, label="dog",
        bbox_x=0.1, bbox_y=0.1, bbox_width=0.2, bbox_height=0.2,
    )

    target = tmp_path / "out"
    write_tables_csv(db, project.id, target)

    csv = (target / FILES_FILENAME).read_text()
    header_line, *data_lines = csv.splitlines()
    headers = header_line.split(",")
    rel_idx = headers.index("relative_path")
    rel_values = {row.split(",")[rel_idx] for row in data_lines}
    assert rel_values == {"IMG.jpg"}


def test_unknown_project_raises(db, tmp_path):
    with pytest.raises(ValueError, match="not found"):
        write_tables_csv(db, "no-such-id", tmp_path / "out")


def test_detections_export_ignores_project_threshold(db, tmp_path):
    """The folder-run data exports are the complete record: a detection
    below the project's detection_threshold still appears in
    addaxai-detections.csv (Dan's must-fix). Thresholding is an in-app /
    media-output concern only."""
    project = make_project(db, name="csv-complete", detection_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=_write_placeholder(tmp_path / "src" / "IMG_LOW.jpg"),
        observation_type="animal",
    )
    make_detection(
        db, file_id=file.id, category="animal", confidence=0.12,
        label="cat", verified=False,
        bbox_x=0.1, bbox_y=0.1, bbox_width=0.2, bbox_height=0.2,
    )

    target = tmp_path / "out"
    write_tables_csv(db, project.id, target)

    csv = (target / DETECTIONS_FILENAME).read_text()
    assert "cat" in csv
    assert "0.12" in csv
