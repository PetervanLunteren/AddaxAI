"""Tests for the /api/projects/{project_id}/export endpoints."""

from __future__ import annotations

import csv
import io
import json
import sqlite3
import tempfile
import uuid
import zipfile
from datetime import date, datetime

import pytest

from app.api.crud import export_formats
from app.models.label_taxonomy import LabelTaxonomy
from tests.conftest import (
    make_deployment,
    make_detection,
    make_file,
    make_project,
    make_site,
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
        "image_uuid", "filename", "datetime", "camera_name",
        "latitude", "longitude", "species", "scientific_name",
        "count", "sex", "life_stage", "behavior", "max_confidence",
        "classification_method", "observation_comments", "is_verified",
    ]
    data = rows[1:]
    # Expect deer (count=2) + person (count=1) on June file; blank on December file.
    assert len(data) == 3
    species = sorted(r[6] for r in data)
    assert species == ["blank", "deer", "person"]

    deer = next(r for r in data if r[6] == "deer")
    assert deer[8] == "2"
    assert float(deer[12]) == pytest.approx(0.9, abs=1e-4)

    # DST-correct offsets.
    june = next(r for r in data if "2024-06-15" in r[2])
    assert june[2].endswith("+02:00")
    # For December the file is a blank row.
    dec = next(r for r in data if "2024-12-15" in r[2])
    assert dec[2].endswith("+01:00")
    assert f_dec.id == dec[0]


def test_export_observations_tsv_and_xlsx(client, db):
    project, _site, deployment = _build_simple_project(db)
    f = make_file(db, deployment_id=deployment.id)
    make_detection(db, file_id=f.id, category="animal", confidence=0.8, label="fox")
    db.commit()

    resp_tsv = client.get(f"/api/projects/{project.id}/export/observations?format=tsv")
    assert resp_tsv.status_code == 200
    assert resp_tsv.headers["content-type"].startswith("text/tab-separated-values")
    tsv_rows = list(csv.reader(io.StringIO(resp_tsv.content.decode("utf-8")), delimiter="\t"))
    assert tsv_rows[0][0] == "image_uuid"
    assert any("fox" in r for r in tsv_rows[1:])

    resp_xlsx = client.get(f"/api/projects/{project.id}/export/observations?format=xlsx")
    assert resp_xlsx.status_code == 200
    assert "spreadsheetml" in resp_xlsx.headers["content-type"]
    from openpyxl import load_workbook

    wb = load_workbook(io.BytesIO(resp_xlsx.content))
    ws = wb.active
    assert ws.title == "Observations"
    sheet_rows = list(ws.iter_rows(values_only=True))
    assert sheet_rows[0][0] == "image_uuid"
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
    f = make_file(db, deployment_id=deployment.id)
    make_detection(db, file_id=f.id, category="animal", confidence=0.9, label="fox")
    db.commit()

    resp = client.get(f"/api/projects/{project.id}/export/spatial?format=geojson")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/geo+json")
    payload = json.loads(resp.content)
    assert payload["type"] == "FeatureCollection"

    layers = {feat["properties"]["layer"] for feat in payload["features"]}
    assert layers == {"deployments", "observations", "species_summary"}

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
        "observations.shp", "observations.shx", "observations.dbf", "observations.prj",
        "species_summary.shp", "species_summary.shx", "species_summary.dbf",
        "species_summary.prj",
    }
    assert expected.issubset(names)


def test_export_spatial_gpkg(client, db):
    project, _site, deployment = _build_simple_project(db)
    f = make_file(db, deployment_id=deployment.id)
    make_detection(db, file_id=f.id, category="animal", confidence=0.9, label="fox")
    db.commit()

    resp = client.get(f"/api/projects/{project.id}/export/spatial?format=gpkg")
    assert resp.status_code == 200

    # Round-trip through sqlite3 to confirm the three feature tables exist.
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
        conn.close()
    finally:
        import os

        os.unlink(tmp_path)
    assert tables == {"deployments", "observations", "species_summary"}


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
        display_name="Vulpes vulpes",
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
        display_name="fox",
        label_confidence=0.88,
        label_taxonomy_id=taxonomy.id,
    )
    db.commit()

    resp = client.get(f"/api/projects/{project.id}/export/camtrap-dp")
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
        "filePath", "filePublic", "fileMediatype",
    ]
    assert obs_rows[0][0] == "observationID"
    # One detection → one animal observation row.
    assert len(obs_rows) == 2
    assert obs_rows[1][7] == "animal"
    assert obs_rows[1][8] == "Vulpes vulpes"


def test_export_camtrap_dp_422_when_no_deployments(client, db):
    project = make_project(db, timezone="UTC")
    db.commit()
    resp = client.get(f"/api/projects/{project.id}/export/camtrap-dp")
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

    resp = client.get(f"/api/projects/{project.id}/export/camtrap-dp")
    assert resp.status_code == 200
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        obs_rows = list(csv.reader(io.StringIO(zf.read("observations.csv").decode())))
    assert len(obs_rows) == 2
    assert obs_rows[1][0].startswith("obs-blank-")
    assert obs_rows[1][7] == "blank"
    assert obs_rows[1][2] == f.id


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
