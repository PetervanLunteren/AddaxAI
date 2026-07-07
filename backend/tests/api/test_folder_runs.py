"""Tests for /api/folder-runs.

The endpoint orchestrates a project (mode='folder_run') and a queue
entry. These tests pin the contract: create returns both, the project
has the right mode, the queue entry has no site, the step state
round-trips through GET, and lookups for non-folder-run project IDs
404 cleanly.
"""

from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import make_project


@pytest.fixture(autouse=True)
def mock_manifest_manager():
    """The folder-run create + lookup paths reach into ManifestManager:
    create validates models, lookup resolves friendly names. Stub it
    out so the test does not need the real model directory. The mock
    manifest exposes both ``model_id`` and ``friendly_name`` as real
    strings; the lookup endpoint's Pydantic schema requires string
    types and would otherwise reject MagicMock attributes."""
    mock_mgr = MagicMock()
    mock_mgr.get_model.return_value = MagicMock(
        model_id="MD5A-0-0", friendly_name="MegaDetector 5a"
    )
    with patch("app.ml.manifest_manager.ManifestManager", return_value=mock_mgr):
        yield mock_mgr


def test_create_folder_run_auto_name(client):
    resp = client.post(
        "/api/folder-runs",
        json={
            "source_folder": "/Volumes/Photos/Kruger_April",
            "image_count": 412,
            "video_count": 7,
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["project"]["name"] == "Kruger_April"
    assert body["project"]["mode"] == "folder_run"
    assert body["project"]["timezone"] == "UTC"
    assert body["project"]["folder_run_state"] == {
        "step": "setup",
        "source_folder": "/Volumes/Photos/Kruger_April",
    }
    assert body["step"] == "setup"
    assert body["queue_entry"]["folder_path"] == "/Volumes/Photos/Kruger_April"
    assert body["queue_entry"]["site_id"] is None
    assert body["queue_entry"]["video_count"] == 7
    assert body["queue_entry"]["image_count"] == 412


def test_create_folder_run_explicit_name(client):
    resp = client.post(
        "/api/folder-runs",
        json={
            "source_folder": "/tmp/anything",
            "name": "my-test-run",
        },
    )
    assert resp.status_code == 201
    assert resp.json()["project"]["name"] == "my-test-run"


def test_create_folder_run_rejects_duplicate_name(client):
    client.post(
        "/api/folder-runs",
        json={"source_folder": "/tmp/a", "name": "dup-run"},
    )
    resp = client.post(
        "/api/folder-runs",
        json={"source_folder": "/tmp/b", "name": "dup-run"},
    )
    assert resp.status_code == 409


def test_same_source_folder_resumes_existing_run(client):
    """Legacy-AddaxAI behaviour: re-selecting an analysed folder
    returns the existing folder-run project rather than creating a
    new one. There is no "recent work" list in the UI; this is how
    users revisit their work."""
    first = client.post(
        "/api/folder-runs",
        json={"source_folder": "/tmp/resume-me"},
    ).json()
    first_id = first["project"]["id"]

    second = client.post(
        "/api/folder-runs",
        json={"source_folder": "/tmp/resume-me"},
    )
    assert second.status_code == 201
    assert second.json()["project"]["id"] == first_id


def test_resume_preserves_persisted_step(client):
    """If the user got to step `save` and then reopens the folder,
    they should land on `save`, not back at the first step."""
    created = client.post(
        "/api/folder-runs",
        json={"source_folder": "/tmp/resume-step"},
    ).json()
    run_id = created["project"]["id"]

    client.patch(
        f"/api/folder-runs/{run_id}/step", json={"step": "save"}
    )

    resumed = client.post(
        "/api/folder-runs",
        json={"source_folder": "/tmp/resume-step"},
    ).json()
    assert resumed["project"]["id"] == run_id
    assert resumed["step"] == "save"


def test_resume_ignores_explicit_name(client):
    """A second POST with a different explicit name on the same
    folder still resumes the existing run rather than failing on
    a duplicate name or creating a parallel project."""
    first = client.post(
        "/api/folder-runs",
        json={
            "source_folder": "/tmp/named-folder",
            "name": "first-attempt",
        },
    ).json()

    second = client.post(
        "/api/folder-runs",
        json={
            "source_folder": "/tmp/named-folder",
            "name": "second-attempt",
        },
    ).json()

    assert second["project"]["id"] == first["project"]["id"]
    # The original name is preserved; we are resuming, not renaming.
    assert second["project"]["name"] == "first-attempt"


def test_different_source_folder_creates_new_run(client):
    first = client.post(
        "/api/folder-runs",
        json={"source_folder": "/tmp/folder-a"},
    ).json()
    second = client.post(
        "/api/folder-runs",
        json={"source_folder": "/tmp/folder-b"},
    ).json()

    assert first["project"]["id"] != second["project"]["id"]


def test_get_folder_run(client):
    created = client.post(
        "/api/folder-runs",
        json={"source_folder": "/tmp/get-test"},
    ).json()
    run_id = created["project"]["id"]

    resp = client.get(f"/api/folder-runs/{run_id}")
    assert resp.status_code == 200
    assert resp.json()["project"]["id"] == run_id
    assert resp.json()["step"] == "setup"


def test_get_folder_run_404_for_unknown(client):
    resp = client.get("/api/folder-runs/does-not-exist")
    assert resp.status_code == 404


def test_output_preview_endpoint_returns_counts(client, db):
    """End-to-end: create a run, manually attach a file + detection,
    and GET the preview. Confirms the response shape matches the
    Pydantic schema and the underlying compute is wired up."""
    from tests.conftest import (
        make_deployment,
        make_detection,
        make_file,
        make_project,
    )

    project = make_project(
        db, name="preview-endpoint", mode="folder_run"
    )
    dep = make_deployment(db, project_id=project.id)
    f = make_file(
        db,
        deployment_id=dep.id,
        observation_type="animal",
        size_bytes=1234,
    )
    make_detection(db, file_id=f.id, confidence=0.95, label="dog")

    resp = client.post(
        f"/api/folder-runs/{project.id}/output-preview",
        json={"separate_group_by": "taxonomic"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_files"] == 1
    assert data["image_count"] == 1
    assert data["video_count"] == 0
    assert data["total_bytes"] == 1234
    assert data["files_with_known_size"] == 1
    assert data["dropped_by_filter"] == 0
    assert data["in_scope_files"] == 1
    # Unmapped label under taxonomic grouping → other/<label> (slugged),
    # no source subfolder so no extra path segment.
    assert data["by_media_tree"] == {"other/dog": 1}


def test_output_preview_honours_excluded_label_ids(client, db):
    """Excluding the only species drops the file from in-scope."""
    from tests.conftest import (
        make_deployment,
        make_detection,
        make_file,
        make_project,
    )

    project = make_project(
        db, name="preview-filter", mode="folder_run"
    )
    dep = make_deployment(db, project_id=project.id)
    f = make_file(
        db,
        deployment_id=dep.id,
        observation_type="animal",
        size_bytes=1234,
    )
    make_detection(db, file_id=f.id, confidence=0.95, label="dog")

    resp = client.post(
        f"/api/folder-runs/{project.id}/output-preview",
        json={"excluded_label_ids": ["dog"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["dropped_by_filter"] == 1
    assert data["in_scope_files"] == 0
    assert data["by_media_tree"] == {}


def test_output_preview_404_for_research_project(client, db):
    research = make_project(db, name="research-preview", mode="research")

    resp = client.post(
        f"/api/folder-runs/{research.id}/output-preview", json={}
    )
    assert resp.status_code == 404


def test_get_folder_run_404_for_research_project(client, db):
    """A research-project id must not resolve as a folder run, even if
    the caller knows the id. Prevents accidentally landing the stepper
    on a real project."""
    research = make_project(db, name="research-only", mode="research")

    resp = client.get(f"/api/folder-runs/{research.id}")
    assert resp.status_code == 404


def test_patch_step_persists_and_round_trips(client):
    created = client.post(
        "/api/folder-runs",
        json={"source_folder": "/tmp/step-test"},
    ).json()
    run_id = created["project"]["id"]

    resp = client.patch(
        f"/api/folder-runs/{run_id}/step",
        json={"step": "setup"},
    )
    assert resp.status_code == 200
    assert resp.json()["step"] == "setup"

    follow_up = client.get(f"/api/folder-runs/{run_id}").json()
    assert follow_up["step"] == "setup"
    # The other state keys (source_folder) survive the step update.
    assert (
        follow_up["project"]["folder_run_state"]["source_folder"]
        == "/tmp/step-test"
    )


def test_patch_step_rejects_unknown_step(client):
    created = client.post(
        "/api/folder-runs",
        json={"source_folder": "/tmp/bad-step"},
    ).json()
    run_id = created["project"]["id"]

    resp = client.patch(
        f"/api/folder-runs/{run_id}/step",
        json={"step": "garbage"},
    )
    assert resp.status_code == 422


def test_folder_run_invisible_in_research_projects_list(client):
    client.post(
        "/api/folder-runs",
        json={"source_folder": "/tmp/hidden"},
    )
    resp = client.get("/api/projects")
    assert resp.status_code == 200
    assert resp.json() == []


def test_folder_run_visible_when_listing_by_mode(client):
    client.post(
        "/api/folder-runs",
        json={"source_folder": "/tmp/listed", "name": "listed-run"},
    )
    resp = client.get("/api/projects?mode=folder_run")
    assert resp.status_code == 200
    names = [p["name"] for p in resp.json()]
    assert "listed-run" in names


# ---------------------------------------------------------------------
# Lookup endpoint + force_new "Discard and start over" flow
# ---------------------------------------------------------------------


def test_lookup_returns_null_when_no_match(client):
    """The common case: user picks a brand-new folder, lookup says
    nothing's there, the form proceeds without a notice."""
    resp = client.get(
        "/api/folder-runs/lookup", params={"folder": "/tmp/never-seen"}
    )
    assert resp.status_code == 200
    assert resp.json() is None


def test_lookup_returns_summary_for_existing_run(client, db):
    """Lookup should give the Step 1 notice card everything it needs:
    project id, name, dates, models, step, and a few headline counts."""
    from tests.conftest import (
        make_deployment,
        make_detection,
        make_file,
    )

    created = client.post(
        "/api/folder-runs",
        json={
            "source_folder": "/tmp/lookup-target",
            "name": "lookup-run",
        },
    ).json()
    run_id = created["project"]["id"]

    # Attach a deployment + files + detections so the count fields
    # have something meaningful to assert on.
    from app.models import Project
    project = db.get(Project, run_id)
    project.detection_model_id = "MD5A-0-0"
    project.classification_model_id = "SPECIESNET-0-0"
    db.commit()
    dep = make_deployment(
        db, project_id=run_id, folder_path="/tmp/lookup-target"
    )
    f1 = make_file(db, deployment_id=dep.id, verified=True)
    f2 = make_file(db, deployment_id=dep.id, verified=False)
    # f1 was verified by the user, so its detection got cascaded to
    # verified=True too. The make_file fixture writes columns directly
    # (no cascade), so reflect the production behaviour by setting both
    # explicitly here.
    make_detection(
        db, file_id=f1.id, confidence=0.9, label="dog", verified=True
    )
    make_detection(db, file_id=f2.id, confidence=0.85, label="wolf")

    resp = client.get(
        "/api/folder-runs/lookup",
        params={"folder": "/tmp/lookup-target"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == run_id
    assert body["name"] == "lookup-run"
    assert body["detection_model_id"] == "MD5A-0-0"
    assert body["classification_model_id"] == "SPECIESNET-0-0"
    # Friendly names fall back to the id when the manifest doesn't
    # know the model (the stubbed manifest in this suite returns the
    # MD5A id), so we just assert presence rather than exact wording.
    assert body["detection_model_name"]
    assert body["classification_model_name"]
    # Resume from step "setup" because the user finished the folder picker.
    assert body["step"] == "setup"
    assert body["file_count"] == 2
    assert body["detection_count"] == 2
    assert body["species_count"] == 2
    assert body["verified_file_count"] == 1
    # 1 of 2 files verified → its detection got cascaded to verified
    # too (see crud/file.py:update_file). The other file's detection
    # is still unverified.
    assert body["verified_detection_count"] == 1


def test_lookup_returns_zero_counts_for_freshly_picked_folder(client):
    """A run with no deployment yet still has a valid lookup response —
    counts just sit at zero. Important: a user might pick a folder,
    close the wizard, reopen later — the lookup must still work even
    before analysis has produced anything."""
    created = client.post(
        "/api/folder-runs",
        json={"source_folder": "/tmp/empty-lookup"},
    ).json()
    run_id = created["project"]["id"]

    resp = client.get(
        "/api/folder-runs/lookup",
        params={"folder": "/tmp/empty-lookup"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == run_id
    assert body["file_count"] == 0
    assert body["detection_count"] == 0
    assert body["species_count"] == 0
    assert body["verified_file_count"] == 0
    assert body["verified_detection_count"] == 0


def test_create_force_new_replaces_existing_run(client):
    """force_new=True on a folder with an existing run cascade-deletes
    the previous project and returns a brand-new one — different id,
    fresh state."""
    first = client.post(
        "/api/folder-runs",
        json={"source_folder": "/tmp/replace-me"},
    ).json()
    first_id = first["project"]["id"]

    second = client.post(
        "/api/folder-runs",
        json={
            "source_folder": "/tmp/replace-me",
            "force_new": True,
        },
    )
    assert second.status_code == 201
    second_id = second.json()["project"]["id"]
    assert second_id != first_id

    # The previous project is gone; loading it 404s.
    assert client.get(f"/api/folder-runs/{first_id}").status_code == 404


def test_create_force_new_with_no_existing_match_still_creates(client):
    """force_new on a folder that has no previous run is harmless —
    same as a regular create. Lets the frontend always pass the flag
    after the user clicked "Discard and start over" without needing
    to track whether the lookup actually found something."""
    resp = client.post(
        "/api/folder-runs",
        json={
            "source_folder": "/tmp/no-previous",
            "force_new": True,
        },
    )
    assert resp.status_code == 201
    assert resp.json()["project"]["folder_run_state"]["source_folder"] == (
        "/tmp/no-previous"
    )


def test_create_default_resumes_existing_run(client):
    """Sanity: force_new defaults to False, so the legacy
    create-or-resume behaviour is unchanged for callers that don't
    pass the flag."""
    first = client.post(
        "/api/folder-runs",
        json={"source_folder": "/tmp/default-resume"},
    ).json()
    first_id = first["project"]["id"]

    second = client.post(
        "/api/folder-runs",
        json={"source_folder": "/tmp/default-resume"},
    ).json()
    assert second["project"]["id"] == first_id


def test_delete_folder_run_removes_cache_folder(db, tmp_path):
    """The cascade-delete helper cleans up the ``.addaxai/projects/<pid>/``
    folder under the deployment's folder_path. Best-effort: missing
    folders are fine."""
    from app.api.crud import project as crud_project
    from tests.conftest import make_deployment, make_project

    source_folder = tmp_path / "source"
    source_folder.mkdir()
    project = make_project(db, name="cleanup-test", mode="folder_run")
    make_deployment(
        db, project_id=project.id, folder_path=str(source_folder)
    )

    # Drop a fake artifact tree where the worker would have written it.
    cache_dir = source_folder / ".addaxai" / "projects" / project.id
    cache_dir.mkdir(parents=True)
    (cache_dir / "results.json").write_text("{}")

    deleted = crud_project.delete_folder_run(db, project.id)
    assert deleted is True
    assert not cache_dir.exists()
    # Empty parent dirs are cleaned up so the source folder is left clean.
    assert not (source_folder / ".addaxai").exists()


def test_delete_folder_run_returns_false_for_unknown(db):
    from app.api.crud import project as crud_project

    assert crud_project.delete_folder_run(db, "does-not-exist") is False


def test_lookup_counts_detection_verifications_independently_of_files(
    client, db
):
    """A user can verify individual observations in the verify grid
    without marking the whole file as done. Those verifications still
    show up in ``verified_detection_count`` so the Step 1 picker can
    honestly say "X of Y observations verified" — independent of
    whether the file-level done flag has been ticked."""
    from tests.conftest import (
        make_deployment,
        make_detection,
        make_file,
    )

    created = client.post(
        "/api/folder-runs",
        json={"source_folder": "/tmp/det-verified-only"},
    ).json()
    run_id = created["project"]["id"]
    dep = make_deployment(
        db, project_id=run_id, folder_path="/tmp/det-verified-only"
    )
    # Two files, neither File.verified, but every detection on f1 is
    # human-confirmed.
    f1 = make_file(db, deployment_id=dep.id, verified=False)
    f2 = make_file(db, deployment_id=dep.id, verified=False)
    make_detection(
        db, file_id=f1.id, confidence=0.9, label="dog", verified=True
    )
    make_detection(db, file_id=f2.id, confidence=0.85, label="wolf")

    resp = client.get(
        "/api/folder-runs/lookup",
        params={"folder": "/tmp/det-verified-only"},
    )
    body = resp.json()
    assert body["verified_file_count"] == 0
    assert body["verified_detection_count"] == 1
    assert body["detection_count"] == 2


def test_create_after_promote_auto_dedupes_name(client, db):
    """Regression: a user analysed /tmp/foo (folder run named "foo"),
    promoted that to a research project, then re-picked the same
    folder. The new folder run must NOT 409 — auto-named runs should
    silently dedup to "foo (2)" so the flow stays zero-friction.
    Reported by the user during beta testing."""
    from app.api.crud import project as crud_project
    from app.api.schemas.project import ProjectUpdate

    first = client.post(
        "/api/folder-runs",
        json={"source_folder": "/tmp/dedupe-after-promote"},
    ).json()
    first_id = first["project"]["id"]
    # The auto-name derives from the folder basename.
    assert first["project"]["name"] == "dedupe-after-promote"

    # Simulate the promote flow: flip the project's mode + clear the
    # folder_run_state. Going through the crud helper keeps this in
    # lockstep with what PromoteDialog actually does.
    crud_project.update_project(
        db,
        first_id,
        ProjectUpdate(mode="research", folder_run_state=None),
    )

    second = client.post(
        "/api/folder-runs",
        json={"source_folder": "/tmp/dedupe-after-promote"},
    )
    assert second.status_code == 201
    body = second.json()
    # New project id (the previous one is now a research project),
    # name dedup'd with the canonical " (N)" suffix.
    assert body["project"]["id"] != first_id
    assert body["project"]["name"] == "dedupe-after-promote (2)"


def test_create_with_explicit_colliding_name_still_409s(client):
    """When the caller passes an explicit name that collides, the
    409 stands and the message says "project" (not "folder run")
    because the existing row might be a research project."""
    client.post(
        "/api/folder-runs",
        json={"source_folder": "/tmp/explicit-a", "name": "explicit-collide"},
    )
    resp = client.post(
        "/api/folder-runs",
        json={"source_folder": "/tmp/explicit-b", "name": "explicit-collide"},
    )
    assert resp.status_code == 409
    assert "project named" in resp.json()["detail"]


def test_rerun_resets_data(client, db):
    """``POST /api/folder-runs/{id}/rerun`` wipes deployment data and
    moves the queue entry back to pending, but keeps the project and
    queue entry rows so the run id survives. Verified detections are
    destroyed by design (the destructive confirm dialog warns)."""
    from tests.conftest import (
        make_deployment,
        make_detection,
        make_file,
    )

    created = client.post(
        "/api/folder-runs",
        json={"source_folder": "/tmp/rerun-target"},
    ).json()
    run_id = created["project"]["id"]
    queue_entry_id = created["queue_entry"]["id"]

    # Simulate a completed run: queue entry processed, one deployment,
    # files, detections (one verified).
    from app.models import DeploymentQueue
    dep = make_deployment(
        db, project_id=run_id, folder_path="/tmp/rerun-target"
    )
    f1 = make_file(db, deployment_id=dep.id, verified=True)
    f2 = make_file(db, deployment_id=dep.id, verified=False)
    make_detection(
        db, file_id=f1.id, confidence=0.9, label="dog", verified=True
    )
    make_detection(db, file_id=f2.id, confidence=0.85, label="wolf")

    queue_entry = db.get(DeploymentQueue, queue_entry_id)
    queue_entry.status = "completed"
    queue_entry.deployment_id = dep.id
    db.commit()

    resp = client.post(f"/api/folder-runs/{run_id}/rerun")
    assert resp.status_code == 200
    body = resp.json()

    # Project + queue entry survive, run id is unchanged.
    assert body["project"]["id"] == run_id
    assert body["queue_entry"] is not None
    assert body["queue_entry"]["id"] == queue_entry_id

    # Queue entry reset to pending with cleared lifecycle fields.
    assert body["queue_entry"]["status"] == "pending"
    assert body["queue_entry"]["deployment_id"] is None
    assert body["queue_entry"]["processed_at_utc"] is None
    assert body["queue_entry"]["error"] is None
    assert body["queue_entry"]["warnings"] is None

    # Deployments + files + detections all gone (cascade through
    # Deployment.delete).
    from app.models import Deployment, Detection, File
    db.expire_all()
    assert db.query(Deployment).filter_by(project_id=run_id).count() == 0
    assert (
        db.query(File)
        .join(Deployment)
        .filter(Deployment.project_id == run_id)
        .count()
        == 0
    )
    assert (
        db.query(Detection)
        .join(File)
        .join(Deployment)
        .filter(Deployment.project_id == run_id)
        .count()
        == 0
    )


def test_rerun_404_for_unknown_run(client):
    resp = client.post("/api/folder-runs/does-not-exist/rerun")
    assert resp.status_code == 404


def test_rerun_404_for_research_project(client, db):
    """Research-mode projects are not folder runs; rerun must 404 on
    them rather than corrupting the project's deployments."""
    proj = make_project(db, mode="research")
    resp = client.post(f"/api/folder-runs/{proj.id}/rerun")
    assert resp.status_code == 404


def _create_run(client, source: str) -> str:
    resp = client.post(
        "/api/folder-runs",
        json={"source_folder": source, "image_count": 1, "video_count": 0},
    )
    assert resp.status_code == 201
    return resp.json()["project"]["id"]


def test_save_outputs_marks_media_subdir_not_output_root(client, tmp_path):
    """The scan-skip marker goes on the addaxai-media subfolder only.

    The output dir defaults to the source folder itself; a marker at
    its root would make every future re-scan skip the user's entire
    source. Pinned here because the regression is silent and severe.
    """
    source = tmp_path / "src"
    source.mkdir()
    run_id = _create_run(client, str(source))

    resp = client.post(
        f"/api/folder-runs/{run_id}/save-outputs",
        json={"output_dir": str(source), "separate_folders": True},
    )
    assert resp.status_code == 200
    assert resp.json()["job_id"]

    assert (source / "addaxai-media" / ".addaxai-output").is_file()
    assert not (source / ".addaxai-output").exists()


def test_save_outputs_data_only_creates_no_media_dir(client, tmp_path):
    """A data-exports-only save (no media modules) must not create the
    addaxai-media folder or any marker."""
    source = tmp_path / "src"
    source.mkdir()
    run_id = _create_run(client, str(source))

    resp = client.post(
        f"/api/folder-runs/{run_id}/save-outputs",
        json={"output_dir": str(source), "csv": True},
    )
    assert resp.status_code == 200

    assert not (source / "addaxai-media").exists()
    assert not (source / ".addaxai-output").exists()


def test_create_pins_detection_threshold_to_inference_floor(client):
    """Folder runs pin the display threshold to the classification gate
    at creation (the setup step keeps the two in sync afterwards): the
    grid and counts show exactly what was classified, while data
    exports bypass the threshold entirely."""
    from app.ml.detection import DEFAULT_CLASSIFICATION_GATE

    resp = client.post(
        "/api/folder-runs",
        json={
            "source_folder": "/Volumes/Photos/Pinned_Floor",
            "image_count": 1,
            "video_count": 0,
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["project"]["detection_threshold"] == DEFAULT_CLASSIFICATION_GATE


def test_legacy_counts_and_summary_steps_resume_on_labels(client, db):
    """Runs persisted at the retired counts / summary steps resume on
    labels: the step right before save in the 3-step flow."""
    from app.models import Project

    for legacy in ("counts", "summary", "observations", "overview"):
        resp = client.post(
            "/api/folder-runs",
            json={"source_folder": f"/tmp/legacy-{legacy}"},
        )
        run_id = resp.json()["project"]["id"]
        proj = db.get(Project, run_id)
        proj.folder_run_state = {
            **(proj.folder_run_state or {}),
            "step": legacy,
        }
        db.commit()

        follow_up = client.get(f"/api/folder-runs/{run_id}")
        assert follow_up.json()["step"] == "labels", legacy
