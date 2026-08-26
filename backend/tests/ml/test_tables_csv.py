"""Tests for the tables_csv postprocess output module.

The row schemas live in the ``export_crud`` builders and have their own
coverage there. Here we pin that this wrapper writes all three files
(``addaxai-summary.csv`` + ``addaxai-files.csv`` +
``addaxai-detections.csv``) at the right paths, that it trims to the
folder-run column set, and that ``relative_path`` on the files and
detections tables is the file's path under its deployment's source folder.
"""

import csv
from datetime import datetime
from pathlib import Path

import pytest

from app.ml.postprocessing_outputs.tables_csv import (
    DETECTIONS_FILENAME,
    FILES_FILENAME,
    SUMMARY_FILENAME,
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



def test_writes_all_three_files_at_canonical_paths(db, tmp_path):
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

    # Summary + files + detections only: a folder run has no deployments
    # or counts tables (ecological interpretation lives in projects mode).
    assert (target / SUMMARY_FILENAME).is_file()
    assert (target / FILES_FILENAME).is_file()
    assert (target / DETECTIONS_FILENAME).is_file()
    assert (target / DETECTIONS_FILENAME).stat().st_size > 0
    assert len(result.output_paths) == 3
    assert not (target / "addaxai-deployments.csv").exists()
    assert not (target / "addaxai-counts.csv").exists()

    with open(target / SUMMARY_FILENAME, newline="") as f:
        summary = list(csv.DictReader(f))
    assert [r["classification_label"] for r in summary] == ["dog"]
    assert summary[0]["n_images"] == "1"
    assert summary[0]["n_detections"] == "1"
    # No ecological interpretation in a folder run: no events figure and
    # no Counts total. Photos, videos and boxes per species only.
    assert "n_events" not in summary[0]
    assert "n_individuals" not in summary[0]
    assert set(summary[0]) >= {"n_images", "n_videos", "n_detections"}


def test_folder_run_headers_omit_deployment_id_and_notes(db, tmp_path):
    """A folder run has one synthetic deployment and never writes notes,
    so neither column carries information. `event_id` stays: it is the
    only thing saying which files share a burst."""
    project = make_project(db, name="csv-headers")
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
    write_tables_csv(db, project.id, target)

    files_headers = (
        (target / FILES_FILENAME).read_text().splitlines()[0].split(",")
    )
    det_headers = (
        (target / DETECTIONS_FILENAME).read_text().splitlines()[0].split(",")
    )

    for headers in (files_headers, det_headers):
        assert "deployment_id" not in headers
        assert "notes" not in headers
        assert "file_id" in headers
        assert "event_id" in headers

    assert "detection_id" in det_headers
    assert "relative_path" in files_headers
    assert "relative_path" in det_headers

    # The camera EXIF columns survive the folder-run trim: they are file
    # metadata, not projects-only structure.
    for name in (
        "camera_make",
        "camera_model",
        "ambient_temperature",
        "camera_serial",
    ):
        assert name in files_headers


def test_folder_run_files_table_keeps_the_species_columns(db, tmp_path):
    """A folder run's whole point is "what is in my folder", so the one label
    per file has to survive the folder-run column trim. `folder_run_table`
    drops by name, so this only breaks if someone adds them to OMITTED_COLUMNS."""
    project = make_project(db, name="csv-species")
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
    write_tables_csv(db, project.id, target)

    with open(target / FILES_FILENAME, newline="", encoding="utf-8") as f:
        headers, row = list(csv.reader(f))

    species = [
        row[headers.index(name)]
        for name in ("classification_label", "scientific_name", "common_name")
    ]
    assert species == ["red fox", "Vulpes vulpes", "Red fox"]
    assert row[headers.index("observation_type")] == "animal"
    # The five ranks come along too, so a folder run can be grouped by family
    # or order without joining anything.
    for rank in (
        "taxon_class",
        "taxon_order",
        "taxon_family",
        "taxon_genus",
        "taxon_species",
    ):
        assert rank in headers
    # Both confidences survive the folder-run column trim too, each still
    # paired with what it scores.
    obs = headers.index("observation_type")
    assert headers[obs : obs + 4] == [
        "observation_type",
        "detection_confidence",
        "classification_label",
        "classification_confidence",
    ]
    assert row[headers.index("detection_confidence")] == "0.9"


def test_projects_mode_builders_keep_every_column(db, tmp_path):
    """The trimming lives in the folder-run writers, never in the shared
    builders. Pinned so a future edit cannot silently narrow the
    projects Export page."""
    from app.api.crud import export as export_crud

    project = make_project(db, name="csv-projects-untouched")
    dep = make_deployment(db, project_id=project.id)
    make_file(
        db,
        deployment_id=dep.id,
        file_path=_write_placeholder(tmp_path / "src" / "IMG.jpg"),
        observation_type="animal",
    )

    files_headers, _rows = export_crud.build_files_rows(db, project)
    scoped = export_crud.get_scoped_detection_rows(db, project)
    det_headers, _det_rows = export_crud.build_detection_rows(
        db, project, scoped
    )

    assert "deployment_id" in files_headers
    assert "notes" in files_headers
    assert "deployment_id" in det_headers


def test_event_id_is_blank_for_unclustered_files(db, tmp_path):
    """A file with no event carries a blank event_id, never its own file
    id. Pinned here too because the folder-run CSVs are the artifact a
    user actually joins on."""
    project = make_project(db, name="csv-no-event")
    dep = make_deployment(db, project_id=project.id)
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=_write_placeholder(tmp_path / "src" / "IMG.jpg"),
        observation_type="animal",
    )
    make_detection(db, file_id=file.id, category="animal", confidence=0.9)

    target = tmp_path / "out"
    write_tables_csv(db, project.id, target)

    for filename in (FILES_FILENAME, DETECTIONS_FILENAME):
        lines = (target / filename).read_text().splitlines()
        headers = lines[0].split(",")
        row = lines[1].split(",")
        event_idx = headers.index("event_id")
        file_idx = headers.index("file_id")

        assert row[file_idx] != "", filename
        assert row[event_idx] == "", filename


def test_event_id_carries_the_real_event_id_when_clustered(db, tmp_path):
    """A clustered file reports its actual event id in both tables, so
    the two folder-run CSVs group consistently on event_id."""
    from app.models import File
    from tests.conftest import make_event_with_files

    project = make_project(db, name="csv-with-event")
    dep = make_deployment(db, project_id=project.id)
    event = make_event_with_files(
        db,
        deployment_id=dep.id,
        event_start_local=datetime(2024, 6, 1, 12, 0, 0),
    )
    clustered = db.query(File).filter(File.deployment_id == dep.id).one()
    clustered.file_path = _write_placeholder(tmp_path / "src" / "IMG.jpg")
    make_detection(db, file_id=clustered.id, category="animal", confidence=0.9)
    db.commit()

    target = tmp_path / "out"
    write_tables_csv(db, project.id, target)

    for filename in (FILES_FILENAME, DETECTIONS_FILENAME):
        lines = (target / filename).read_text().splitlines()
        headers = lines[0].split(",")
        row = lines[1].split(",")
        assert row[headers.index("event_id")] == event.id, filename


def test_row_count_totals_all_tables(db, tmp_path):
    """Two files, each with one detection of its own species. Total
    row_count = 2 summary rows + 2 files + 2 detections."""
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

    assert result.row_count == 6


def _relative_paths(target: Path, filename: str) -> set[str]:
    with open(target / filename, newline="") as f:
        return {r["relative_path"] for r in csv.DictReader(f)}


def test_relative_path_is_relative_to_deployment_folder(db, tmp_path):
    """relative_path on the files and detections tables is the file's path
    under its deployment's source folder. Both tables, so a reader of
    either can find the photo without a join."""
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

    assert _relative_paths(target, FILES_FILENAME) == {"sub/IMG.jpg"}
    assert _relative_paths(target, DETECTIONS_FILENAME) == {"sub/IMG.jpg"}


def test_relative_path_falls_back_to_filename(db, tmp_path):
    """When the deployment has no source folder, relative_path is the
    bare filename, on both tables."""
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

    assert _relative_paths(target, FILES_FILENAME) == {"IMG.jpg"}
    assert _relative_paths(target, DETECTIONS_FILENAME) == {"IMG.jpg"}


def test_unknown_project_raises(db, tmp_path):
    with pytest.raises(ValueError, match="not found"):
        write_tables_csv(db, "no-such-id", tmp_path / "out")


def test_detections_export_honours_project_threshold(db, tmp_path):
    """The folder-run tables honour the counting threshold, the same rule
    projects mode uses, so the spreadsheet holds what the Labels step
    showed and the user could correct.

    This reverses an earlier decision (the tables used to be the complete
    record and ignored the threshold). It was reversed because the two
    sheets disagreed with each other and with the app: addaxai-files.csv
    was thresholded while addaxai-detections.csv beside it was not, and
    users read the extra rows as species the app had hidden from them.
    addaxai-recognitions.json is the complete record now.

    The verified override still applies: a human decision outranks the
    score, so a box someone checked survives however low it scored."""
    project = make_project(db, name="csv-complete", counting_threshold=0.5)
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
    make_detection(
        db, file_id=file.id, category="animal", confidence=0.11,
        label="badger", verified=True,
        bbox_x=0.5, bbox_y=0.5, bbox_width=0.2, bbox_height=0.2,
    )

    target = tmp_path / "out"
    write_tables_csv(db, project.id, target)

    # Parse the label column rather than searching the raw text: "cat" is
    # a substring of the "classification_label" header.
    with open(target / DETECTIONS_FILENAME, newline="") as f:
        labels = {r["classification_label"] for r in csv.DictReader(f)}
    assert labels == {"badger"}
