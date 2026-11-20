"""Tests for the tables_xlsx postprocess output module.

The row schemas are covered by the export-layer tests; this wrapper just
makes sure one workbook with the Summary, Counts, Detections and Files
sheets lands at the right path, trimmed to the folder-run column set.
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
    make_track,
)

# XLSX files are ZIP archives. ZIP magic is "PK\x03\x04".
_XLSX_MAGIC = b"PK\x03\x04"


def _write_placeholder(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return str(path)



def test_writes_four_sheet_workbook(db, tmp_path):
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
    result = write_tables_xlsx(db, project.id, target, include_frames=False)

    assert result.output_path.endswith(XLSX_FILENAME)
    output_path = target / XLSX_FILENAME
    assert output_path.is_file()
    with open(output_path, "rb") as f:
        assert f.read(4) == _XLSX_MAGIC

    wb = load_workbook(output_path)
    # The projects workbook's order minus Deployments: Summary first, the
    # sheet that says what was found, then the analysis-ready Counts.
    assert wb.sheetnames == ["Summary", "Counts", "Detections", "Files"]

    # Same trimmed column set as the folder-run CSVs.
    for sheet in ("Counts", "Files", "Detections"):
        headers = [c.value for c in wb[sheet][1]]
        assert "deployment_id" not in headers
        assert "notes" not in headers
        assert "event_id" in headers
    for sheet in ("Files", "Detections"):
        headers = [c.value for c in wb[sheet][1]]
        assert "file_id" in headers

    summary = list(wb["Summary"].values)
    headers, row = list(summary[0]), list(summary[1])
    assert row[headers.index("classification_label")] == "dog"
    assert row[headers.index("n_images")] == 1
    assert row[headers.index("n_detections")] == 1
    assert "n_events" in headers
    assert "n_individuals" in headers


def test_files_sheet_keeps_the_species_columns(db, tmp_path):
    """The XLSX writer is wired independently of the CSV one, so the folder-run
    Files sheet needs its own pin that the one label per file is present."""
    from openpyxl import load_workbook

    project = make_project(db, name="xlsx-species")
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
        label="red fox",
        scientific_name="Vulpes vulpes",
        common_name="Red fox",
    )

    target = tmp_path / "out"
    write_tables_xlsx(db, project.id, target, include_frames=False)

    wb = load_workbook(target / XLSX_FILENAME)
    rows = list(wb["Files"].values)
    headers, row = list(rows[0]), list(rows[1])
    species = [
        row[headers.index(name)]
        for name in ("classification_label", "scientific_name", "common_name")
    ]
    assert species == ["red fox", "Vulpes vulpes", "Red fox"]
    assert "taxon_family" in headers
    assert "detection_confidence" in headers
    assert "classification_confidence" in headers
    # openpyxl gives these back as real numbers, not text, unlike every
    # other cell in the Files sheet.
    assert row[headers.index("detection_confidence")] == 0.9


def test_unknown_project_raises(db, tmp_path):
    with pytest.raises(ValueError, match="not found"):
        write_tables_xlsx(db, "no-such-id", tmp_path / "out", include_frames=False)


def test_frames_sheet_is_added_only_when_asked(db, tmp_path):
    """A clip with one track of three boxes: with ``include_frames`` the
    workbook gains a Frames sheet of three rows, without it there is none."""
    from openpyxl import load_workbook

    project = make_project(db, name="xlsx-frames")
    dep = make_deployment(db, project_id=project.id)
    clip = make_file(
        db,
        deployment_id=dep.id,
        file_path=_write_placeholder(tmp_path / "src" / "CLIP.mp4"),
        file_type="video",
        file_format="mp4",
        frame_rate=30.0,
        best_frame_number=0,
        observation_type="animal",
    )
    track = make_track(db, file_id=clip.id, track_key=1, start_frame=0, end_frame=60,
                       representative_frame_number=30)
    for frame in (0, 30, 60):
        make_detection(db, file_id=clip.id, frame_number=frame, track_id=track.id,
                       category="animal", confidence=0.9, label="dog")

    def sheets(include_frames: bool, name: str) -> dict[str, int]:
        result = write_tables_xlsx(db, project.id, tmp_path / name,
                                   include_frames=include_frames)
        wb = load_workbook(result.output_path)
        return {ws.title: ws.max_row - 1 for ws in wb.worksheets}

    assert "Frames" not in sheets(False, "off")
    on = sheets(True, "on")
    assert list(on) == ["Summary", "Counts", "Detections", "Files", "Frames"]
    assert on["Frames"] == 3
