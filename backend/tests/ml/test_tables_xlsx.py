"""Tests for the tables_xlsx postprocess output module.

The row schemas are covered by the export-layer tests; this wrapper just
makes sure one workbook with the Detections and Counts sheets lands at the
right path.
"""

from pathlib import Path

import pytest

from app.ml.postprocessing_outputs.tables_xlsx import (
    XLSX_FILENAME,
    write_tables_xlsx,
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



def test_writes_two_sheet_workbook(db, tmp_path):
    from openpyxl import load_workbook

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
    result = write_tables_xlsx(db, project.id, target)

    assert result.output_path.endswith(XLSX_FILENAME)
    output_path = target / XLSX_FILENAME
    assert output_path.is_file()
    with open(output_path, "rb") as f:
        assert f.read(4) == _XLSX_MAGIC

    wb = load_workbook(output_path)
    assert wb.sheetnames == ["Deployments", "Files", "Detections", "Counts"]


def test_unknown_project_raises(db, tmp_path):
    with pytest.raises(ValueError, match="not found"):
        write_tables_xlsx(db, "no-such-id", tmp_path / "out")
