"""Tests for the observations_xlsx postprocess output module.

Same row schema as the CSV exporter and the research-projects Export
page (covered by their own tests); this wrapper just makes sure the
file lands at the right path with non-zero bytes that begin with the
XLSX magic header.
"""

from pathlib import Path

import pytest

from app.ml.postprocessing_outputs._output_context import OutputContext
from app.ml.postprocessing_outputs.observations_xlsx import (
    XLSX_FILENAME,
    write_observations_xlsx,
)
from tests.conftest import (
    make_deployment,
    make_detection,
    make_file,
    make_project,
)

# XLSX files are ZIP archives. ZIP magic is "PK\x03\x04".
_XLSX_MAGIC = b"PK\x03\x04"


def _write_placeholder(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return str(path)


def _ctx(output_root: Path) -> OutputContext:
    return OutputContext(output_root=output_root)


def test_writes_xlsx_at_canonical_path(db, tmp_path):
    project = make_project(db, name="xlsx-basic")
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
    result = write_observations_xlsx(db, project.id, _ctx(target))

    assert result.output_path.endswith(XLSX_FILENAME)
    output_path = target / XLSX_FILENAME
    assert output_path.is_file()

    # Sanity check the bytes look like an XLSX (ZIP archive magic
    # number). Anything that opens as a valid spreadsheet starts
    # with these four bytes.
    with open(output_path, "rb") as f:
        magic = f.read(4)
    assert magic == _XLSX_MAGIC


def test_row_count_matches_observation_rows(db, tmp_path):
    project = make_project(db, name="xlsx-rows")
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
    result = write_observations_xlsx(db, project.id, _ctx(target))

    assert result.row_count == 2


def test_unknown_project_raises(db, tmp_path):
    with pytest.raises(ValueError, match="not found"):
        write_observations_xlsx(db, "no-such-id", _ctx(tmp_path / "out"))
