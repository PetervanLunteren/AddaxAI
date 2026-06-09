"""Tests for the /api/projects/{project_id}/export endpoints."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import sqlite3
import tempfile
import uuid
import zipfile
from datetime import date, datetime
from unittest.mock import patch

import pytest

from app.api.crud import export_formats
from app.models.label_taxonomy import LabelTaxonomy
from tests.conftest import (
    make_deployment,
    make_detection,
    make_event_with_files,
    make_file,
    make_project,
    make_site,
)


def _run_camtrap_dp_export(client, db, project_id: str):
    """Helper: drive the full prepare → worker → download cycle.

    The prepare endpoint registers the worker with ws_manager and waits
    for a frontend "ready" signal before running it. In tests we have no
    WebSocket, so we kick the worker directly (after patching its
    `get_db` to use a session bound to the test engine) and then GET
    the download endpoint.

    The worker's `finally: db.close()` would otherwise detach the
    fixture's `db` instance and break post-test assertions. We hand it
    a separate session bound to the same StaticPool engine so closing
    is harmless and the in-memory data stays visible.

    Returns the download Response. Raises if prepare returns non-202;
    in that case the caller should call /prepare directly and assert
    the error code instead of using this helper.
    """
    prepare = client.post(f"/api/projects/{project_id}/export/camtrap-dp/prepare")
    assert prepare.status_code == 202, prepare.text
    job_id = prepare.json()["job_id"]

    from sqlalchemy.orm import sessionmaker

    from app.workers import camtrap_export_worker
    from tests.conftest import _engine  # noqa: PLC2701 — shared in-memory engine

    worker_session_factory = sessionmaker(bind=_engine)

    def _fake_get_db():
        s = worker_session_factory()
        try:
            yield s
        finally:
            s.close()

    # Make sure the worker reads any rows the fixture committed before
    # we forked off this helper. Without it the fresh worker session
    # could miss rows that are still buffered on the fixture session.
    db.commit()

    with patch.object(camtrap_export_worker, "get_db", _fake_get_db):
        asyncio.run(camtrap_export_worker.process_camtrap_export_job(job_id))

    return client.get(
        f"/api/projects/{project_id}/export/camtrap-dp/download?job_id={job_id}"
    )

# ---------------------------------------------------------------------------
# Factory sugar
# ---------------------------------------------------------------------------


def _build_simple_project(db, *, timezone: str = "UTC", detection_threshold: float = 0.5):
    project = make_project(db, timezone=timezone, detection_threshold=detection_threshold)
    site = make_site(db, project_id=project.id, name="alpha", latitude=52.1, longitude=5.1)
    deployment = make_deployment(
        db,
        site_id=site.id,
        start_date_local=date(2024, 6, 1),
        end_date_local=date(2024, 6, 10),
        camera_model="Cam-X",
        camera_serial="SN-1",
    )
    db.commit()
    return project, site, deployment


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


def test_export_observations_csv_happy_path(client, db):
    project, _site, deployment = _build_simple_project(db, timezone="Europe/Amsterdam")
    f_june = make_file(
        db,
        deployment_id=deployment.id,
        captured_at_local=datetime(2024, 6, 15, 9, 0, 0),
    )
    make_detection(db, file_id=f_june.id, category="animal", confidence=0.9, label="deer")
    make_detection(db, file_id=f_june.id, category="animal", confidence=0.7, label="deer")
    make_detection(db, file_id=f_june.id, category="person", confidence=0.95)
    f_dec = make_file(
        db,
        deployment_id=deployment.id,
        captured_at_local=datetime(2024, 12, 15, 3, 0, 0),
        observation_type="blank",
    )
    db.commit()

    resp = client.get(f"/api/projects/{project.id}/export/observations?format=csv")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    assert "attachment; filename=" in resp.headers["content-disposition"]

    rows = list(csv.reader(io.StringIO(resp.content.decode("utf-8"))))
    headers = rows[0]
    assert headers == [
        "detection_id", "file_id", "event_id", "relative_path",
        "absolute_path",
        "datetime", "site_name", "latitude", "longitude",
        "detection_category", "detection_confidence", "bbox_x", "bbox_y",
        "bbox_width", "bbox_height", "frame_number", "classification_label",
        "classification_confidence", "taxon_class", "taxon_order",
        "taxon_family", "taxon_genus", "taxon_species",
        "scientific_name", "common_name", "is_verified", "count",
    ]
    cls_i = headers.index("classification_label")
    cat_i = headers.index("detection_category")
    conf_i = headers.index("detection_confidence")
    dt_i = headers.index("datetime")
    fid_i = headers.index("file_id")
    did_i = headers.index("detection_id")

    data = rows[1:]
    # One row per detection: 2 deer + 1 person on June; 1 blank row for December.
    assert len(data) == 4

    # Each deer detection keeps its own confidence (no max-aggregation).
    deer = [r for r in data if r[cls_i] == "deer"]
    assert len(deer) == 2
    assert sorted(float(r[conf_i]) for r in deer) == pytest.approx([0.7, 0.9], abs=1e-4)

    # Person: detected but not species-classified, so classification is empty.
    person = next(r for r in data if r[cat_i] == "person")
    assert float(person[conf_i]) == pytest.approx(0.95, abs=1e-4)
    assert person[cls_i] == ""

    # December file is an empty/blank sentinel row.
    blank = next(r for r in data if r[cat_i] == "blank")
    assert blank[did_i] == ""
    assert blank[fid_i] == f_dec.id

    # DST-correct offsets.
    june = next(r for r in data if "2024-06-15" in r[dt_i])
    assert june[dt_i].endswith("+02:00")
    assert blank[dt_i].endswith("+01:00")


def test_export_observations_tsv_and_xlsx(client, db):
    project, _site, deployment = _build_simple_project(db)
    f = make_file(db, deployment_id=deployment.id)
    make_detection(db, file_id=f.id, category="animal", confidence=0.8, label="fox")
    db.commit()

    resp_tsv = client.get(f"/api/projects/{project.id}/export/observations?format=tsv")
    assert resp_tsv.status_code == 200
    assert resp_tsv.headers["content-type"].startswith("text/tab-separated-values")
    tsv_rows = list(csv.reader(io.StringIO(resp_tsv.content.decode("utf-8")), delimiter="\t"))
    assert tsv_rows[0][0] == "detection_id"
    assert any("fox" in r for r in tsv_rows[1:])

    resp_xlsx = client.get(f"/api/projects/{project.id}/export/observations?format=xlsx")
    assert resp_xlsx.status_code == 200
    assert "spreadsheetml" in resp_xlsx.headers["content-type"]
    from openpyxl import load_workbook

    wb = load_workbook(io.BytesIO(resp_xlsx.content))
    ws = wb.active
    assert ws.title == "Observations"
    sheet_rows = list(ws.iter_rows(values_only=True))
    assert sheet_rows[0][0] == "detection_id"
    assert any(
        isinstance(v, str) and "fox" in v
        for row in sheet_rows[1:]
        for v in row
    )


def test_export_observations_respects_threshold_and_verified_override(client, db):
    project, _site, deployment = _build_simple_project(db, detection_threshold=0.5)
    f = make_file(db, deployment_id=deployment.id)
    # Below threshold, unverified → excluded.
    make_detection(db, file_id=f.id, category="animal", confidence=0.3, label="fox")
    # Below threshold, verified → included via override.
    make_detection(
        db, file_id=f.id, category="animal", confidence=0.2, label="deer", verified=True
    )
    # Above threshold → included.
    make_detection(db, file_id=f.id, category="animal", confidence=0.8, label="bear")
    db.commit()

    resp = client.get(f"/api/projects/{project.id}/export/observations?format=csv")
    text = resp.content.decode("utf-8")
    assert "deer" in text
    assert "bear" in text
    assert "fox" not in text


def test_export_observations_respects_excluded_classes(client, db):
    project = make_project(db, timezone="UTC", excluded_classes=["domestic_cat"])
    site = make_site(db, project_id=project.id)
    deployment = make_deployment(db, site_id=site.id)
    f = make_file(db, deployment_id=deployment.id)
    make_detection(db, file_id=f.id, category="animal", confidence=0.9, label="domestic_cat")
    make_detection(db, file_id=f.id, category="animal", confidence=0.9, label="fox")
    db.commit()

    resp = client.get(f"/api/projects/{project.id}/export/observations?format=csv")
    text = resp.content.decode("utf-8")
    assert "fox" in text
    assert "domestic_cat" not in text


def test_export_observations_project_not_found(client):
    resp = client.get("/api/projects/does-not-exist/export/observations?format=csv")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Spatial
# ---------------------------------------------------------------------------


def test_export_spatial_geojson(client, db):
    project, site, deployment = _build_simple_project(db)
    taxonomy = LabelTaxonomy(
        id=str(uuid.uuid4()),
        classification_model_id="TEST-MODEL",
        name="fox",
        scientific_name="Vulpes vulpes",
        level="species",
        taxon_class="mammalia",
        taxon_order="carnivora",
        taxon_family="canidae",
        taxon_genus="vulpes",
        taxon_species="vulpes",
    )
    db.add(taxonomy)
    db.flush()
    f = make_file(db, deployment_id=deployment.id)
    make_detection(
        db, file_id=f.id, category="animal", confidence=0.9,
        label="fox", label_taxonomy_id=taxonomy.id,
    )
    db.commit()

    resp = client.get(f"/api/projects/{project.id}/export/spatial?format=geojson")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/geo+json")
    payload = json.loads(resp.content)
    assert payload["type"] == "FeatureCollection"

    # Two genuinely spatial layers only; the per-detection points are gone.
    layers = {feat["properties"]["layer"] for feat in payload["features"]}
    assert layers == {"deployments", "species_summary"}

    summary = next(
        feat for feat in payload["features"]
        if feat["properties"]["layer"] == "species_summary"
    )
    props = summary["properties"]
    assert props["classification_label"] == "fox"
    assert props["scientific_name"] == "Vulpes vulpes"
    assert props["taxon_genus"] == "vulpes"
    assert props["taxon_species"] == "vulpes"
    assert props["total_count"] == 1

    for feat in payload["features"]:
        assert feat["geometry"]["type"] == "Point"
        lon, lat = feat["geometry"]["coordinates"]
        assert lon == site.longitude
        assert lat == site.latitude


def test_export_spatial_shapefile_zip(client, db):
    project, _site, deployment = _build_simple_project(db)
    f = make_file(db, deployment_id=deployment.id)
    make_detection(db, file_id=f.id, category="animal", confidence=0.9, label="fox")
    db.commit()

    resp = client.get(f"/api/projects/{project.id}/export/spatial?format=shapefile")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/zip")

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = set(zf.namelist())
    expected = {
        "deployments.shp", "deployments.shx", "deployments.dbf", "deployments.prj",
        "species_summary.shp", "species_summary.shx", "species_summary.dbf",
        "species_summary.prj",
    }
    assert expected.issubset(names)
    assert not any(n.startswith("observations.") for n in names)


def test_export_spatial_gpkg(client, db):
    project, _site, deployment = _build_simple_project(db)
    taxonomy = LabelTaxonomy(
        id=str(uuid.uuid4()),
        classification_model_id="TEST-MODEL",
        name="fox",
        scientific_name="Vulpes vulpes",
        level="species",
        taxon_genus="vulpes",
        taxon_species="vulpes",
    )
    db.add(taxonomy)
    db.flush()
    f = make_file(db, deployment_id=deployment.id)
    make_detection(
        db, file_id=f.id, category="animal", confidence=0.9,
        label="fox", label_taxonomy_id=taxonomy.id,
    )
    db.commit()

    resp = client.get(f"/api/projects/{project.id}/export/spatial?format=gpkg")
    assert resp.status_code == 200

    # Round-trip through sqlite3 to confirm the feature tables and that the
    # species_summary attributes are actually populated (not silently blank).
    with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as tmp:
        tmp.write(resp.content)
        tmp_path = tmp.name
    try:
        conn = sqlite3.connect(tmp_path)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT table_name FROM gpkg_contents ORDER BY table_name"
            )
        }
        row = conn.execute(
            "SELECT classification_label, scientific_name, taxon_species, "
            "total_count FROM species_summary"
        ).fetchone()
        conn.close()
    finally:
        import os

        os.unlink(tmp_path)
    assert tables == {"deployments", "species_summary"}
    assert row == ("fox", "Vulpes vulpes", "vulpes", 1)


# ---------------------------------------------------------------------------
# CamTrap DP
# ---------------------------------------------------------------------------


def test_export_camtrap_dp_happy_path(client, db):
    project, _site, deployment = _build_simple_project(db, timezone="Europe/Amsterdam")
    # Link detection to a taxonomy row for scientificName.
    taxonomy = LabelTaxonomy(
        id=str(uuid.uuid4()),
        classification_model_id="TEST-MODEL",
        name="fox",
        scientific_name="Vulpes vulpes",
        level="species",
    )
    db.add(taxonomy)
    db.flush()

    f = make_file(
        db,
        deployment_id=deployment.id,
        captured_at_local=datetime(2024, 6, 15, 9, 0, 0),
    )
    make_detection(
        db,
        file_id=f.id,
        category="animal",
        confidence=0.9,
        label="fox",
        scientific_name="fox",
        label_confidence=0.88,
        label_taxonomy_id=taxonomy.id,
    )
    db.commit()

    resp = _run_camtrap_dp_export(client, db, project.id)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/zip")

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = set(zf.namelist())
        assert names == {
            "datapackage.json",
            "deployments.csv",
            "media.csv",
            "observations.csv",
        }
        dp = json.loads(zf.read("datapackage.json"))
        deps_rows = list(csv.reader(io.StringIO(zf.read("deployments.csv").decode())))
        media_rows = list(csv.reader(io.StringIO(zf.read("media.csv").decode())))
        obs_rows = list(csv.reader(io.StringIO(zf.read("observations.csv").decode())))

    assert dp["title"] == project.name
    assert dp["name"].startswith("addaxai-")
    assert dp["temporal"]["start"] == "2024-06-15"
    assert any(
        entry["scientificName"] == "Vulpes vulpes" for entry in dp["taxonomic"]
    )

    # CSV header sanity.
    assert deps_rows[0][0] == "deploymentID"
    assert media_rows[0] == [
        "mediaID", "deploymentID", "captureMethod", "timestamp",
        "filePath", "filePublic", "fileName", "fileMediatype",
        "exifData", "favorite", "mediaComments",
    ]
    assert obs_rows[0][0] == "observationID"
    # One detection → one animal observation row.
    assert len(obs_rows) == 2
    assert obs_rows[1][7] == "animal"
    assert obs_rows[1][9] == "Vulpes vulpes"


def test_export_camtrap_dp_422_when_no_deployments(client, db):
    project = make_project(db, timezone="UTC")
    db.commit()
    # 422 fires inside /prepare before the worker is dispatched, so we
    # bypass the helper and inspect the prepare response directly.
    resp = client.post(f"/api/projects/{project.id}/export/camtrap-dp/prepare")
    assert resp.status_code == 422


def test_export_camtrap_dp_blank_row_for_file_without_detections(client, db):
    project, _site, deployment = _build_simple_project(db)
    f = make_file(
        db,
        deployment_id=deployment.id,
        captured_at_local=datetime(2024, 6, 15, 9, 0, 0),
        observation_type="blank",
    )
    db.commit()

    resp = _run_camtrap_dp_export(client, db, project.id)
    assert resp.status_code == 200
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        obs_rows = list(csv.reader(io.StringIO(zf.read("observations.csv").decode())))
    assert len(obs_rows) == 2
    assert obs_rows[1][0].startswith("obs-blank-")
    assert obs_rows[1][7] == "blank"
    assert obs_rows[1][2] == f.id


def test_export_camtrap_dp_emits_media_and_event_rows(client, db):
    """Camtrap-DP dual model: one media-level row per bounding box, plus
    one event-level row per species carrying the effective (human) count
    with no bbox. Replaces the retired box-less observation flow."""
    from app.api.crud.event_observation import (
        calculate_max_n_for_event,
        set_human_count,
    )

    project, _site, deployment = _build_simple_project(db, timezone="UTC")
    ev = make_event_with_files(
        db,
        deployment_id=deployment.id,
        event_start_local=datetime(2024, 6, 15, 9, 0, 0),
    )
    det = make_detection(
        db,
        file_id=ev.files[0].id,
        category="animal",
        confidence=0.9,
        label="deer",
        bbox_x=0.1,
        bbox_y=0.1,
        bbox_width=0.2,
        bbox_height=0.2,
    )
    tax = LabelTaxonomy(
        name="deer", level="species", classification_model_id="",
        project_id=project.id, common_name="Deer",
        scientific_name="Cervidae",
    )
    db.add(tax)
    db.flush()
    det.label_taxonomy_id = tax.id
    obs = calculate_max_n_for_event(db, ev.id, project.detection_threshold)
    db.flush()
    # Human bumps the deer count to 3 (more than any single frame showed).
    set_human_count(db, obs[0].id, 3)
    db.commit()

    resp = _run_camtrap_dp_export(client, db, project.id)
    assert resp.status_code == 200
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        headers, *obs_rows = csv.reader(
            io.StringIO(zf.read("observations.csv").decode())
        )

    level_i = headers.index("observationLevel")
    count_i = headers.index("count")
    bx_i = headers.index("bboxX")
    sci_i = headers.index("scientificName")

    media = [r for r in obs_rows if r[level_i] == "media"]
    event = [r for r in obs_rows if r[level_i] == "event"]
    # One media-level row per box (the deer detection), bbox set, count 1.
    assert len(media) == 1
    assert media[0][bx_i] != ""
    assert media[0][count_i] == "1"
    # One event-level row per species carrying the effective count (3),
    # no bbox, with the resolved scientific name.
    assert len(event) == 1
    assert event[0][count_i] == "3"
    assert event[0][bx_i] == ""
    assert event[0][sci_i] == "Cervidae"


# ---------------------------------------------------------------------------
# Unit tests for the pure serializers
# ---------------------------------------------------------------------------


def test_slugify_edges():
    assert export_formats.slugify("My Project!") == "my-project"
    assert export_formats.slugify("under_score name") == "under-score-name"
    assert export_formats.slugify("   ") == "project"


def test_make_gpkg_point_blob_layout():
    blob = export_formats.make_gpkg_point_blob(5.1, 52.1)
    # Header: 'GP' + version(0) + flags(1) + srid(4326, LE int32) = 8 bytes
    assert blob[:2] == b"GP"
    assert blob[2] == 0
    assert blob[3] == 1
    assert int.from_bytes(blob[4:8], "little", signed=True) == 4326
    # WKB Point: byte-order(1) + type(1) + X + Y = 1+4+8+8 = 21 bytes
    assert len(blob) == 8 + 21
    assert blob[8] == 1
    assert int.from_bytes(blob[9:13], "little") == 1


def test_serialize_csv_roundtrip():
    payload = export_formats.serialize_csv(
        ["a", "b"], [[1, "x"], [2, "y,z"]]
    )
    rows = list(csv.reader(io.StringIO(payload.decode("utf-8"))))
    assert rows == [["a", "b"], ["1", "x"], ["2", "y,z"]]
