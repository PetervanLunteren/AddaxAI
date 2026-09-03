"""Tests for /api/folder-runs.

The endpoint orchestrates a project (mode='folder_run') and a queue
entry. These tests pin the contract: create returns both, the project
has the right mode, the queue entry has no site, the step state
round-trips through GET, and lookups for non-folder-run project IDs
404 cleanly.
"""

from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import (
    make_deployment,
    make_detection,
    make_file,
    make_project,
)


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


def test_create_folder_run_file_mtime_fallback_defaults_off(client):
    """Omitting the field must never silently enable the fallback."""
    resp = client.post(
        "/api/folder-runs",
        json={"source_folder": "/Volumes/Photos/NoFlag", "image_count": 3},
    )
    assert resp.status_code == 201
    assert resp.json()["queue_entry"]["use_file_mtime_fallback"] is False


def test_create_folder_run_carries_file_mtime_fallback(client):
    """The opt-in has to reach the queue row, since that is what the
    shared worker reads. A folder run draws no charts, but its exported
    files table has a datetime column and its README reports the capture
    range, so both are empty without this."""
    resp = client.post(
        "/api/folder-runs",
        json={
            "source_folder": "/Volumes/Photos/Gabon_AVI",
            "image_count": 0,
            "video_count": 1,
            "use_file_mtime_fallback": True,
        },
    )
    assert resp.status_code == 201
    assert resp.json()["queue_entry"]["use_file_mtime_fallback"] is True


def test_create_folder_run_carries_datetime_offset(client):
    """The Adjust dates correction has to reach the queue row, since the
    shared worker applies it to every capture timestamp at ingest."""
    resp = client.post(
        "/api/folder-runs",
        json={
            "source_folder": "/Volumes/Photos/WrongClock",
            "image_count": 5,
            "datetime_offset_seconds": -2325283200,
        },
    )
    assert resp.status_code == 201
    assert resp.json()["queue_entry"]["datetime_offset_seconds"] == -2325283200


def test_create_folder_run_datetime_offset_defaults_null(client):
    resp = client.post(
        "/api/folder-runs",
        json={"source_folder": "/Volumes/Photos/GoodClock", "image_count": 3},
    )
    assert resp.status_code == 201
    assert resp.json()["queue_entry"]["datetime_offset_seconds"] is None


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


def test_output_preview_anonymise_reaches_the_estimate(client, db, tmp_path):
    """``anonymise`` must be declared on the request schema. Pydantic drops
    a field it does not know without a word (the `run_readme` story), and
    the footer would then quote the container size for a save that writes
    blurred stills."""
    from tests.conftest import make_deployment, make_file, make_project

    project = make_project(db, name="preview-blur", mode="folder_run")
    dep = make_deployment(db, project_id=project.id)
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"x" * 100)
    make_file(
        db,
        deployment_id=dep.id,
        file_path="/cam/clip.mp4",
        file_type="video",
        file_format="mp4",
        observation_type="blank",
        size_bytes=5000,
        best_frame_number=0,
        best_frame_path=str(frame),
    )
    url = f"/api/folder-runs/{project.id}/output-preview"

    whole = client.post(url, json={"include_empty": True})
    assert whole.status_code == 200
    assert whole.json()["in_scope_bytes"] == 5000

    blurred = client.post(url, json={"include_empty": True, "anonymise": True})
    assert blurred.status_code == 200
    assert blurred.json()["in_scope_bytes"] == 100


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


def _run_save_worker(db, monkeypatch, job_id: str) -> None:
    """Run the save-outputs worker synchronously on the test session."""
    import asyncio

    import app.workers.folder_run_save_outputs_worker as worker

    def _test_get_db():
        yield db

    monkeypatch.setattr(worker, "get_db", _test_get_db)
    asyncio.run(worker.process_save_outputs_job(job_id))


def test_save_outputs_marks_media_subdir_not_output_root(
    client, db, tmp_path, monkeypatch
):
    """The scan-skip marker goes on the addaxai-media subfolder only,
    and only the WORKER writes it.

    Root placement is pinned because a marker at the output root (which
    defaults to the source folder) would make every future re-scan skip
    the user's entire source. Worker-only writing is pinned because the
    marker is the wipe's ownership proof: the endpoint stamping it
    before the worker's check handed that proof to any pre-existing
    addaxai-media and got the user's own files deleted.
    """
    source = tmp_path / "src"
    source.mkdir()
    run_id = _create_run(client, str(source))

    resp = client.post(
        f"/api/folder-runs/{run_id}/save-outputs",
        json={"output_dir": str(source), "separate_folders": True},
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    # The endpoint alone must not have stamped anything.
    assert not (source / "addaxai-media" / ".addaxai-output").exists()

    _run_save_worker(db, monkeypatch, job_id)

    assert (source / "addaxai-media" / ".addaxai-output").is_file()
    assert not (source / ".addaxai-output").exists()


def test_save_outputs_leaves_foreign_media_dir_alone(
    client, db, tmp_path, monkeypatch
):
    """A pre-existing addaxai-media folder the app never created (no
    marker) must survive a media save through the real API path.

    Regression: the endpoint used to stamp the marker before spawning
    the job, so the worker's ownership check always found a marker and
    wiped the folder — including one holding the user's own files.
    """
    source = tmp_path / "src"
    source.mkdir()
    foreign = source / "addaxai-media" / "users-own-file.txt"
    foreign.parent.mkdir()
    foreign.write_text("keep me")
    run_id = _create_run(client, str(source))

    resp = client.post(
        f"/api/folder-runs/{run_id}/save-outputs",
        json={"output_dir": str(source), "separate_folders": True},
    )
    assert resp.status_code == 200

    _run_save_worker(db, monkeypatch, resp.json()["job_id"])

    assert foreign.read_text() == "keep me"
    # And not claimed: a marker here would let the NEXT save wipe it.
    assert not (source / "addaxai-media" / ".addaxai-output").exists()


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


def test_unticking_run_details_does_not_write_the_run_info_file(
    client, db, tmp_path, monkeypatch
):
    """The Save step's "Run details" checkbox, through the real path.

    This has to go through the endpoint. The bug lived entirely between
    the request and the job payload: `SaveOutputsRequest` did not declare
    `run_readme`, pydantic drops fields a model does not declare, and the
    router then hand-copied fourteen named fields into the job, so the
    flag was gone twice over before the worker could read it. The worker
    itself was always correct, which is why
    `tests/test_save_outputs_worker.py` passes `"run_readme": False` and
    is green: it hand-builds the payload and never crosses the boundary
    where the value was lost. A second worker-level test would have been
    green against the broken code too.

    Peter's report is the case in the first half: tick JSON only, and get
    the JSON plus a run-info file nobody asked for.
    """
    source = tmp_path / "src"
    source.mkdir()
    run_id = _create_run(client, str(source))

    resp = client.post(
        f"/api/folder-runs/{run_id}/save-outputs",
        json={
            "output_dir": str(source),
            "recognition_json": True,
            "run_readme": False,
        },
    )
    assert resp.status_code == 200
    _run_save_worker(db, monkeypatch, resp.json()["job_id"])

    assert (source / "addaxai-recognitions.json").is_file()
    assert not (source / "addaxai-run-info.txt").exists()


def test_ticking_run_details_still_writes_the_run_info_file(
    client, db, tmp_path, monkeypatch
):
    """The other half, so the fix cannot be "never write it"."""
    source = tmp_path / "src"
    source.mkdir()
    run_id = _create_run(client, str(source))

    resp = client.post(
        f"/api/folder-runs/{run_id}/save-outputs",
        json={"output_dir": str(source), "run_readme": True},
    )
    assert resp.status_code == 200
    _run_save_worker(db, monkeypatch, resp.json()["job_id"])

    assert (source / "addaxai-run-info.txt").is_file()


def test_save_outputs_job_payload_carries_the_whole_request(
    client, db, tmp_path
):
    """Every field of the request reaches the job, none renamed or lost.

    The guard on the spread that replaced the hand-copied dict. Without
    it, going back to naming fields one by one reintroduces exactly the
    omission this pair of tests exists for, and only for whichever flag
    the next person forgets.
    """
    from app.api.routers.folder_runs import SaveOutputsRequest
    from app.models import Job

    source = tmp_path / "src"
    source.mkdir()
    run_id = _create_run(client, str(source))

    resp = client.post(
        f"/api/folder-runs/{run_id}/save-outputs",
        json={"output_dir": str(source)},
    )
    assert resp.status_code == 200

    job = db.query(Job).filter(Job.id == resp.json()["job_id"]).one()
    expected = set(SaveOutputsRequest.model_fields) | {"run_id"}
    stored = set(job.payload)
    assert stored == expected


def test_create_sets_counting_threshold_to_counting_default(client):
    """Folder runs use the same single interpretation floor as projects
    mode: counting_threshold defaults to DEFAULT_COUNTING_THRESHOLD,
    decoupled from the classification gate. The grid, counts, and the
    verification pills all measure over it, so they agree; data exports
    bypass the threshold entirely."""
    from app.core.confidence import DEFAULT_COUNTING_THRESHOLD

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
    assert body["project"]["counting_threshold"] == DEFAULT_COUNTING_THRESHOLD


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


# ----------------------------------------------------------------------
# GET /api/folder-runs — the step-1 "Show recent runs" list
# ----------------------------------------------------------------------


def _make_run(db, folder_path: str, *, updated=None, **kw):
    """A folder run built straight in the DB.

    Built directly rather than through POST /api/folder-runs because that
    endpoint is create-or-resume (it returns the existing run for a folder,
    so it cannot produce the duplicates the dedupe test needs) and because
    an explicit ``updated_at_utc`` makes the ordering assertions
    deterministic.
    """
    from datetime import UTC, datetime

    from app.models import DeploymentQueue

    project = make_project(
        db,
        mode="folder_run",
        folder_run_state={"step": "setup"},
        updated_at_utc=updated or datetime(2026, 1, 1, tzinfo=UTC),
        **kw,
    )
    db.add(DeploymentQueue(project_id=project.id, folder_path=folder_path))
    db.flush()
    return project


def _add_detections(db, project, *specs: tuple[float, bool]):
    """Give a run one file carrying ``(confidence, verified)`` detections."""
    deployment = make_deployment(db, project_id=project.id)
    file = make_file(db, deployment_id=deployment.id)
    for confidence, verified in specs:
        make_detection(
            db, file_id=file.id, confidence=confidence, verified=verified
        )
    db.flush()


def test_list_folder_runs_newest_first(client, db, tmp_path):
    from datetime import UTC, datetime

    older = _make_run(db, str(tmp_path), updated=datetime(2026, 1, 1, tzinfo=UTC))
    newer = _make_run(
        db, str(tmp_path / "newer"), updated=datetime(2026, 6, 1, tzinfo=UTC)
    )
    (tmp_path / "newer").mkdir()

    body = client.get("/api/folder-runs").json()

    assert [r["id"] for r in body] == [newer.id, older.id]
    assert body[0]["folder_path"] == str(tmp_path / "newer")
    assert body[0]["file_count"] == 0


def test_list_folder_runs_dedupes_by_folder_keeping_newest(client, db, tmp_path):
    """Duplicate runs on one folder (possible for runs made before the
    resume logic) collapse to the most recent, mirroring _find_existing_run.
    Without this the list would surface duplicates that are invisible today."""
    from datetime import UTC, datetime

    _make_run(db, str(tmp_path), updated=datetime(2026, 1, 1, tzinfo=UTC))
    newest = _make_run(db, str(tmp_path), updated=datetime(2026, 5, 1, tzinfo=UTC))

    body = client.get("/api/folder-runs").json()

    assert len(body) == 1
    assert body[0]["id"] == newest.id


def test_list_folder_runs_returns_every_run(client, db, tmp_path):
    """No cap: a hidden run is one the user can neither open nor delete, and
    the dedupe needs the whole set anyway. The UI decides how many to show."""
    for i in range(25):
        _make_run(db, str(tmp_path / f"run-{i}"))

    body = client.get("/api/folder-runs").json()

    assert len(body) == 25


def test_list_folder_runs_flags_missing_folder(client, db, tmp_path):
    """A folder that moved / was deleted / is on an unplugged drive is
    reported so the UI can grey it out instead of resuming into nothing."""
    _make_run(db, str(tmp_path))
    _make_run(db, str(tmp_path / "does-not-exist"))

    by_path = {r["folder_path"]: r for r in client.get("/api/folder-runs").json()}

    assert by_path[str(tmp_path)]["folder_exists"] is True
    assert by_path[str(tmp_path / "does-not-exist")]["folder_exists"] is False


def test_list_folder_runs_reports_labels_verified(client, db, tmp_path):
    """The two halves of the row's "labels verified" fraction."""
    run = _make_run(db, str(tmp_path), counting_threshold=0.2)
    _add_detections(db, run, (0.9, True), (0.9, False), (0.9, False), (0.9, False))

    body = client.get("/api/folder-runs").json()

    assert body[0]["detection_count"] == 4
    assert body[0]["verified_detection_count"] == 1


def test_list_folder_runs_labels_verified_applies_threshold_override(
    client, db, tmp_path
):
    """The denominator counts what the run actually shows, so it carries the
    threshold + verified override: a low-confidence detection is excluded
    unless a human verified it (DEVELOPERS.md "Detection threshold and
    verified override"). Getting this wrong would report a percentage against
    a denominator the user never sees."""
    run = _make_run(db, str(tmp_path), counting_threshold=0.5)
    _add_detections(
        db,
        run,
        (0.9, False),  # passes on confidence
        (0.1, True),  # below threshold, but verified: still counted
        (0.1, False),  # below threshold, unverified: excluded
    )

    body = client.get("/api/folder-runs").json()

    assert body[0]["detection_count"] == 2
    assert body[0]["verified_detection_count"] == 1


def test_list_folder_runs_labels_verified_uses_each_runs_own_threshold(
    client, db, tmp_path
):
    """Thresholds are per-project, so a page of runs must not be counted
    against one shared value."""
    strict = _make_run(db, str(tmp_path / "strict"), counting_threshold=0.5)
    loose = _make_run(db, str(tmp_path / "loose"), counting_threshold=0.05)
    for run in (strict, loose):
        _add_detections(db, run, (0.1, False))

    by_id = {r["id"]: r for r in client.get("/api/folder-runs").json()}

    assert by_id[strict.id]["detection_count"] == 0
    assert by_id[loose.id]["detection_count"] == 1


def test_list_folder_runs_reports_zero_for_unanalysed_run(client, db, tmp_path):
    """A run with nothing analysed has no fraction to show; the UI renders
    "not analysed yet" off these zeros rather than a meaningless 0%."""
    _make_run(db, str(tmp_path))

    body = client.get("/api/folder-runs").json()

    assert body[0]["file_count"] == 0
    assert body[0]["detection_count"] == 0
    assert body[0]["verified_detection_count"] == 0


def test_list_folder_runs_distinguishes_empty_result_from_unanalysed(
    client, db, tmp_path
):
    """Two runs report zero detections for very different reasons, and the
    row must not call both "not analysed yet": a folder of empty images WAS
    analysed, and saying otherwise invites the user to re-run it.

    ``file_count`` is what separates them, because File rows are only written
    when results load (ml/json_pipeline.py). Pinned here so a future change to
    when files are created cannot silently break the distinction.
    """
    unanalysed = _make_run(db, str(tmp_path / "unanalysed"))
    empty_result = _make_run(db, str(tmp_path / "empty-result"))
    # Analysed, three images, every one of them blank: files, no detections.
    deployment = make_deployment(db, project_id=empty_result.id)
    for _ in range(3):
        make_file(db, deployment_id=deployment.id)
    db.flush()

    by_id = {r["id"]: r for r in client.get("/api/folder-runs").json()}

    assert by_id[unanalysed.id]["file_count"] == 0
    assert by_id[empty_result.id]["file_count"] == 3
    assert by_id[empty_result.id]["detection_count"] == 0


def test_list_folder_runs_carries_queue_status(client, db, tmp_path):
    """A run killed mid-analysis (crash, power cut) has zero files just like
    a run that never started, but it is a different fact: it must be run
    again, and the setup step says so. The list carries the queue status so
    it can say the same instead of "not analysed yet"."""
    from app.models import DeploymentQueue

    fresh = _make_run(db, str(tmp_path / "fresh"))
    failed = _make_run(db, str(tmp_path / "failed"))
    entry = db.query(DeploymentQueue).filter_by(project_id=failed.id).one()
    entry.status = "failed"
    db.flush()

    by_id = {r["id"]: r for r in client.get("/api/folder-runs").json()}

    assert by_id[fresh.id]["queue_status"] == "pending"
    assert by_id[failed.id]["queue_status"] == "failed"


def test_list_folder_runs_excludes_research_projects(client, db, tmp_path):
    _make_run(db, str(tmp_path))
    make_project(db, mode="research")

    body = client.get("/api/folder-runs").json()

    assert len(body) == 1


def test_list_folder_runs_drops_a_promoted_run(client, db, tmp_path):
    """Promoting a run flips mode to 'research' (PromoteDialog PATCHes the
    project), and it must leave this list: GET /{run_id} 404s for a research
    project, so a row left behind would navigate the user into a dead run.
    """
    run = _make_run(db, str(tmp_path))
    assert len(client.get("/api/folder-runs").json()) == 1

    # Exactly what the promote flow sends (PromoteDialog -> projectsApi.update).
    promoted = client.patch(
        f"/api/projects/{run.id}",
        json={"mode": "research", "folder_run_state": None},
    )
    assert promoted.status_code == 200, promoted.text

    assert client.get("/api/folder-runs").json() == []
    assert client.get(f"/api/folder-runs/{run.id}").status_code == 404


# ----------------------------------------------------------------------
# DELETE /api/folder-runs/{run_id}
# ----------------------------------------------------------------------


def test_delete_folder_run_removes_it(client, db, tmp_path):
    run = _make_run(db, str(tmp_path))

    assert client.delete(f"/api/folder-runs/{run.id}").status_code == 204
    assert client.get(f"/api/folder-runs/{run.id}").status_code == 404
    assert client.get("/api/folder-runs").json() == []


def test_delete_folder_run_unknown_id_404(client):
    assert client.delete("/api/folder-runs/does-not-exist").status_code == 404


def test_delete_folder_run_refuses_research_project(client, db):
    """A research project must not be deletable through the folder-run
    endpoint — it would take a real project's data with it."""
    project = make_project(db, mode="research")

    assert client.delete(f"/api/folder-runs/{project.id}").status_code == 404
    assert db.get(type(project), project.id) is not None


# --- detection checkpoints -------------------------------------------------


def _artifacts(folder, run_id):
    from app.ml.detection_checkpoint import artifacts_dir

    d = artifacts_dir(folder, run_id)
    d.mkdir(parents=True)
    return d


def _write_checkpoint_files(artifacts, *, images_done: int, meta):
    import json

    meta.write(artifacts)
    (artifacts / "md_checkpoint.json").write_text(
        json.dumps({"checkpoint": [{"file": f"{i}.jpg"} for i in range(images_done)]})
    )


def test_rerun_without_a_body_removes_the_whole_cache(client, tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    run_id = _create_run(client, str(folder))
    artifacts = _artifacts(folder, run_id)
    (artifacts / "md_checkpoint.json").write_text("{}")
    (artifacts / "results.json").write_text("{}")

    resp = client.post(f"/api/folder-runs/{run_id}/rerun")
    assert resp.status_code == 200
    assert not (folder / ".addaxai").exists()


def test_rerun_keeps_only_the_checkpoint_files_when_asked(client, tmp_path):
    """Continue keeps what the next run needs to pick detection up and
    nothing else: the rest of the cache describes the run being wiped."""
    from app.ml.detection_checkpoint import CHECKPOINT_FILES

    folder = tmp_path / "run"
    folder.mkdir()
    run_id = _create_run(client, str(folder))
    artifacts = _artifacts(folder, run_id)
    for name in CHECKPOINT_FILES:
        (artifacts / name).write_text("{}")
    (artifacts / "results.json").write_text("{}")
    (artifacts / "embeddings.npz").write_bytes(b"x")
    frames = artifacts / "video_frames" / "clip.mp4"
    frames.mkdir(parents=True)
    (frames / "frame000010.jpg").write_bytes(b"jpg")

    resp = client.post(
        f"/api/folder-runs/{run_id}/rerun", json={"keep_checkpoint": True}
    )
    assert resp.status_code == 200
    assert sorted(p.name for p in artifacts.iterdir()) == sorted(CHECKPOINT_FILES)
    assert resp.json()["queue_entry"]["status"] == "pending"


def test_lookup_reports_detection_resume_for_a_failed_run(client, db, tmp_path):
    """The re-run dialog's Continue button is fed by the lookup: how far
    detection got, plus the saved detection settings the checkpoint is
    valid for. Only a failed entry is asked, and only a checkpoint made
    under the saved settings counts."""
    from app.ml.detection_checkpoint import CheckpointMeta
    from app.models import DeploymentQueue, Project

    folder = tmp_path / "run"
    folder.mkdir()
    resp = client.post(
        "/api/folder-runs",
        json={"source_folder": str(folder), "image_count": 4, "video_count": 0},
    )
    run_id = resp.json()["project"]["id"]
    entry_id = resp.json()["queue_entry"]["id"]
    project = db.get(Project, run_id)
    project.detection_model_id = "MD5A-0-0"
    entry = db.get(DeploymentQueue, entry_id)
    entry.status = "failed"
    db.commit()

    artifacts = _artifacts(folder, run_id)
    matching = CheckpointMeta(
        detection_model_id="MD5A-0-0",
        image_size=project.detection_image_size,
        augment=project.detection_augment,
        image_count=4,
    )
    _write_checkpoint_files(artifacts, images_done=3, meta=matching)

    body = client.get("/api/folder-runs/lookup", params={"folder": str(folder)}).json()
    assert body["detection_resume"] == {"images_done": 3, "images_total": 4}
    assert body["detection_image_size"] == project.detection_image_size
    assert body["detection_augment"] == project.detection_augment

    # Made under other settings: nothing to continue from.
    other = CheckpointMeta(
        detection_model_id="MD5A-0-0",
        image_size=1280,
        augment=project.detection_augment,
        image_count=4,
    )
    _write_checkpoint_files(artifacts, images_done=3, meta=other)
    body = client.get("/api/folder-runs/lookup", params={"folder": str(folder)}).json()
    assert body["detection_resume"] is None

    # A completed run is never asked, whatever sits in the cache.
    _write_checkpoint_files(artifacts, images_done=3, meta=matching)
    entry.status = "completed"
    db.commit()
    body = client.get("/api/folder-runs/lookup", params={"folder": str(folder)}).json()
    assert body["detection_resume"] is None


def test_lookup_uses_the_live_image_count_when_given(client, db, tmp_path):
    """The picker knows how many images the folder holds right now. A
    folder that lost or gained files since the crash gets no Continue,
    which is also what the worker will decide."""
    from app.ml.detection_checkpoint import CheckpointMeta
    from app.models import DeploymentQueue, Project

    folder = tmp_path / "run"
    folder.mkdir()
    resp = client.post(
        "/api/folder-runs",
        json={"source_folder": str(folder), "image_count": 4, "video_count": 0},
    )
    run_id = resp.json()["project"]["id"]
    project = db.get(Project, run_id)
    project.detection_model_id = "MD5A-0-0"
    entry = db.get(DeploymentQueue, resp.json()["queue_entry"]["id"])
    entry.status = "failed"
    db.commit()
    _write_checkpoint_files(
        _artifacts(folder, run_id),
        images_done=3,
        meta=CheckpointMeta("MD5A-0-0", project.detection_image_size,
                            project.detection_augment, 4),
    )

    params = {"folder": str(folder)}
    assert client.get("/api/folder-runs/lookup", params=params).json()[
        "detection_resume"
    ] == {"images_done": 3, "images_total": 4}
    assert client.get(
        "/api/folder-runs/lookup", params={**params, "image_count": 4}
    ).json()["detection_resume"] == {"images_done": 3, "images_total": 4}
    assert client.get(
        "/api/folder-runs/lookup", params={**params, "image_count": 3}
    ).json()["detection_resume"] is None


def test_folder_run_paths_match_with_or_without_a_trailing_slash(client):
    """Create with a slash, look up without, and the other way round: one
    run, found both ways. The queue stores the normalised form."""
    created = client.post(
        "/api/folder-runs", json={"source_folder": "/tmp/slash-run/"}
    ).json()
    assert created["queue_entry"]["folder_path"] == "/tmp/slash-run"
    for folder in ("/tmp/slash-run", "/tmp/slash-run/"):
        body = client.get("/api/folder-runs/lookup", params={"folder": folder}).json()
        assert body is not None and body["id"] == created["project"]["id"]
    # And create-or-resume returns the same run rather than a second one.
    again = client.post("/api/folder-runs", json={"source_folder": "/tmp/slash-run"}).json()
    assert again["project"]["id"] == created["project"]["id"]
