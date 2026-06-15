"""Tests for the no-site banner wiring across GPS-dependent endpoints.

Covers:
- GET /api/projects/{id}/deployments-without-site count endpoint
- ObservationRateMapResponse.deployments_without_site
- ActivityPatternResponse.deployments_without_site
- Exports skip null-site deployments for CamtrapDP / GeoJSON
- Exports include null-site deployments with blank lat/lon in flat CSV
"""

from datetime import datetime

from app.api.crud.statistics import (
    get_activity_pattern,
    get_observation_rate_map,
)
from tests.api.test_export import _run_camtrap_dp_export
from tests.conftest import (
    make_deployment,
    make_event_with_files,
    make_file,
    make_project,
    make_site,
)


def _seed_project_with_mixed_sites(db):
    """Two deployments with a site, one null-site deployment, all with files."""
    project = make_project(db)
    site = make_site(db, project_id=project.id, latitude=10.0, longitude=20.0)
    dep_with = make_deployment(db, site_id=site.id)
    dep_with_2 = make_deployment(db, site_id=site.id)
    dep_null = make_deployment(db, project_id=project.id, site_id=None)

    for dep in (dep_with, dep_with_2, dep_null):
        make_file(
            db,
            deployment_id=dep.id,
            captured_at_local=datetime(2024, 6, 1, 12, 0, 0),
        )
        make_event_with_files(
            db,
            deployment_id=dep.id,
            event_start_local=datetime(2024, 6, 1, 12, 0, 0),
            files_verified=[False],
        )

    db.commit()
    return project, site, dep_with, dep_with_2, dep_null


def test_deployments_without_site_endpoint(client, db):
    project, _site, _d1, _d2, dep_null = _seed_project_with_mixed_sites(db)

    resp = client.get(f"/api/projects/{project.id}/deployments-without-site")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["count"] == 1
    assert payload["deployment_ids"] == [dep_null.id]


def test_observation_rate_map_reports_missing_sites(db):
    project, _site, _d1, _d2, _dep_null = _seed_project_with_mixed_sites(db)

    response = get_observation_rate_map(db, project.id)
    # Both with-site deployments share one site, so the map collapses them
    # into a single feature with deployment_count=2.
    assert len(response.features) == 1
    assert response.features[0].deployment_count == 2
    assert all(f.latitude is not None and f.longitude is not None for f in response.features)
    assert response.deployments_without_site == 1


def test_activity_pattern_reports_missing_sites(db):
    project, _site, _d1, _d2, _dep_null = _seed_project_with_mixed_sites(db)

    response = get_activity_pattern(db, project.id)
    assert response.deployments_without_site == 1


def test_export_deployments_csv_writes_blanks_for_null_site(client, db):
    project, _site, _d1, _d2, _dep_null = _seed_project_with_mixed_sites(db)

    resp = client.get(f"/api/projects/{project.id}/export/deployments?format=csv")
    assert resp.status_code == 200
    body = resp.content.decode("utf-8")
    lines = body.strip().splitlines()
    # Header + one row per deployment (3), including the null-site one.
    assert len(lines) >= 4
    header = lines[0].split(",")
    lat_idx = header.index("latitude")
    lon_idx = header.index("longitude")
    # The null-site deployment has blank lat/lon.
    blank_rows = [
        line for line in lines[1:]
        if (line.split(",")[lat_idx] == "" and line.split(",")[lon_idx] == "")
    ]
    assert blank_rows, "expected at least one row with blank lat/lon"


def test_export_camtrap_dp_skips_null_site_and_reports_header(client, db):
    project, _site, _d1, _d2, dep_null = _seed_project_with_mixed_sites(db)

    resp = _run_camtrap_dp_export(client, db, project.id)
    assert resp.status_code == 200
    skipped_header = resp.headers.get("X-Skipped-Deployment-Ids", "")
    assert dep_null.id in skipped_header


def test_export_camtrap_dp_422_when_only_null_sites(client, db):
    project = make_project(db)
    make_deployment(db, project_id=project.id, site_id=None)
    db.commit()

    # 422 fires inside /prepare; bypass the helper to inspect the
    # prepare response directly without trying to run the worker.
    resp = client.post(f"/api/projects/{project.id}/export/camtrap-dp/prepare")
    assert resp.status_code == 422
