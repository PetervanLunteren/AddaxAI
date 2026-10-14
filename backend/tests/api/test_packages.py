"""Tests for the project package endpoints: the export side under
``/api/projects/{id}/export/package`` and the import side under
``/api/packages``. The workers and the format are tested in
``tests/db/test_project_package.py``; here it is the request handling."""

from __future__ import annotations

from pathlib import Path

from app.services import project_package as pkg
from tests.conftest import make_deployment, make_file, make_job, make_project, make_site
from tests.db.conftest import make_machine


def _project_with_two_deployments(db):
    project = make_project(db, name="Wolves")
    site = make_site(db, project_id=project.id)
    deps = [make_deployment(db, site_id=site.id) for _ in range(2)]
    for dep in deps:
        make_file(db, deployment_id=dep.id, size_bytes=1000)
        make_file(db, deployment_id=dep.id, size_bytes=None)
    db.commit()
    return project.id, [d.id for d in deps]


def test_size_counts_the_selected_deployments(client, db) -> None:
    project_id, deps = _project_with_two_deployments(db)

    r = client.get(f"/api/projects/{project_id}/export/package/size")
    assert r.status_code == 200
    whole = r.json()
    assert whole | {"data_bytes": 0} == {
        "deployments": 2, "files": 4, "media_bytes": 2000, "data_bytes": 0
    }

    r = client.get(
        f"/api/projects/{project_id}/export/package/size?deployment_ids={deps[0]}"
    )
    one = r.json()
    assert one | {"data_bytes": 0} == {
        "deployments": 1, "files": 2, "media_bytes": 1000, "data_bytes": 0
    }
    # The data is an estimate from the database size: never nothing, and
    # smaller for a smaller selection.
    assert 0 < one["data_bytes"] < whole["data_bytes"]


def test_prepare_creates_the_job_with_the_resolved_scope(client, db, tmp_path: Path) -> None:
    project_id, deps = _project_with_two_deployments(db)

    r = client.post(
        f"/api/projects/{project_id}/export/package/prepare",
        json={"output_dir": str(tmp_path), "deployment_ids": [deps[1]], "include_media": True},
    )
    assert r.status_code == 202, r.text
    job = client.get(f"/api/jobs/{r.json()['job_id']}").json()
    assert job["type"] == "package_export"
    assert job["payload"] == {
        "project_id": project_id,
        "deployment_ids": [deps[1]],
        "output_dir": str(tmp_path),
        "include_media": True,
    }

    r = client.post(
        f"/api/projects/{project_id}/export/package/prepare",
        json={"output_dir": str(tmp_path)},
    )
    assert r.status_code == 202
    job = client.get(f"/api/jobs/{r.json()['job_id']}").json()
    assert sorted(job["payload"]["deployment_ids"]) == sorted(deps)
    assert job["payload"]["include_media"] is False


def test_prepare_refusals(client, db, tmp_path: Path) -> None:
    project_id, _deps = _project_with_two_deployments(db)
    url = f"/api/projects/{project_id}/export/package/prepare"

    r = client.post(url, json={"output_dir": str(tmp_path), "deployment_ids": []})
    assert r.status_code == 422

    r = client.post(url, json={"output_dir": str(tmp_path / "missing")})
    assert r.status_code == 400
    assert "Choose a folder" in r.json()["detail"]

    run = make_project(db, name="run", mode="folder_run")
    db.commit()
    r = client.post(
        f"/api/projects/{run.id}/export/package/prepare",
        json={"output_dir": str(tmp_path)},
    )
    assert r.status_code == 400
    assert "Only projects" in r.json()["detail"]

    r = client.post(
        "/api/projects/nope/export/package/prepare", json={"output_dir": str(tmp_path)}
    )
    assert r.status_code == 404


def test_preview_refuses_what_is_not_a_package(client, tmp_path: Path) -> None:
    r = client.post("/api/packages/preview", json={"path": str(tmp_path)})
    assert r.status_code == 400
    assert "not an AddaxAI bundle" in r.json()["detail"]

    r = client.post("/api/packages/preview", json={"path": str(tmp_path / "nope")})
    assert r.status_code == 400
    assert "not a folder" in r.json()["detail"]


def _real_package(tmp_path: Path) -> Path:
    """A package written by a separate machine, so the in-memory test
    database plays the importing side."""
    lead = make_machine(tmp_path / "lead")
    with lead.session() as db:
        project = make_project(db, name="Wolves")
        site = make_site(db, project_id=project.id, name="North")
        dep = make_deployment(db, site_id=site.id, folder_path=str(tmp_path / "A"))
        make_file(db, deployment_id=dep.id)
        db.commit()
        project_id, dep_id = project.id, dep.id
    result = pkg.export_package(
        lead.db_path, project_id, [dep_id], tmp_path, include_media=False,
        report=lambda *_: None,
    )
    lead.engine.dispose()
    return Path(result.package_dir)


def test_preview_and_import_of_a_real_package(client, db, tmp_path: Path) -> None:
    package_dir = _real_package(tmp_path)

    r = client.post("/api/packages/preview", json={"path": str(package_dir)})
    assert r.status_code == 200, r.text
    preview = r.json()
    assert preview["project_name"] == "Wolves"
    assert preview["project_exists"] is False
    assert preview["revision_status"] == "current"
    assert preview["problems"] == []
    assert [d["exists"] for d in preview["deployments"]] == [False]

    r = client.post(
        "/api/packages/import", json={"path": str(package_dir), "replace_existing": True}
    )
    assert r.status_code == 202, r.text
    job = client.get(f"/api/jobs/{r.json()['job_id']}").json()
    assert job["type"] == "package_import"
    assert job["payload"] == {
        "path": str(package_dir),
        "replace_existing": True,
        "project_id": preview["project_id"],
    }


def test_import_is_refused_while_a_job_runs_or_a_name_clashes(
    client, db, tmp_path: Path
) -> None:
    package_dir = _real_package(tmp_path)

    make_job(db, status="running")
    db.commit()
    r = client.post("/api/packages/import", json={"path": str(package_dir)})
    assert r.status_code == 409

    db.query(type(make_job(db))).delete()
    make_project(db, name="Wolves")
    db.commit()
    r = client.post("/api/packages/import", json={"path": str(package_dir)})
    assert r.status_code == 400
    assert "different project named" in r.json()["detail"]


def test_prepare_refuses_a_folder_inside_a_camera_folder(client, db, tmp_path: Path) -> None:
    project = make_project(db, name="Wolves")
    site = make_site(db, project_id=project.id)
    cam = tmp_path / "cams" / "A"
    (cam / "exports").mkdir(parents=True)
    make_deployment(db, site_id=site.id, folder_path=str(cam))
    db.commit()

    r = client.post(
        f"/api/projects/{project.id}/export/package/prepare",
        json={"output_dir": str(cam / "exports")},
    )
    assert r.status_code == 400
    assert "outside the camera folders" in r.json()["detail"]
