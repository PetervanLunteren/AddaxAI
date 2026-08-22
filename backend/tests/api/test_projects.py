"""Tests for the /api/projects endpoints."""

from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import make_project


@pytest.fixture(autouse=True)
def mock_manifest_manager():
    """Patch ManifestManager so model validation always succeeds."""
    mock_mgr = MagicMock()
    mock_mgr.get_model.return_value = MagicMock(model_id="MD5A-0-0")
    with patch("app.ml.manifest_manager.ManifestManager", return_value=mock_mgr):
        yield mock_mgr


# --- List / Create / Get / Update / Delete ---


def test_list_projects_empty(client):
    resp = client.get("/api/projects")
    assert resp.status_code == 200
    assert resp.json() == []


def test_create_project(client):
    resp = client.post(
        "/api/projects",
        json={"name": "My Project", "timezone": "UTC"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "My Project"
    assert data["timezone"] == "UTC"
    assert "id" in data
    assert "created_at_utc" in data


def test_create_project_duplicate_name(client, db):
    make_project(db, name="dup")
    resp = client.post("/api/projects", json={"name": "dup", "timezone": "UTC"})
    assert resp.status_code == 409


def test_create_project_without_timezone_starts_unset(client):
    """Timezone is optional: a new project starts with none and later
    auto-derives one from its first site's coordinates."""
    resp = client.post("/api/projects", json={"name": "no-tz"})
    assert resp.status_code == 201
    assert resp.json()["timezone"] is None


def test_create_project_rejects_invalid_timezone(client):
    """Bogus IANA strings get rejected by the field validator."""
    resp = client.post(
        "/api/projects",
        json={"name": "bad-tz", "timezone": "Moon/Crater"},
    )
    assert resp.status_code == 422


def test_create_project_accepts_valid_iana_timezone(client):
    resp = client.post(
        "/api/projects",
        json={"name": "amsterdam", "timezone": "Europe/Amsterdam"},
    )
    assert resp.status_code == 201
    assert resp.json()["timezone"] == "Europe/Amsterdam"


def test_update_project_timezone(client, db):
    p = make_project(db)
    resp = client.patch(
        f"/api/projects/{p.id}",
        json={"timezone": "America/New_York"},
    )
    assert resp.status_code == 200
    assert resp.json()["timezone"] == "America/New_York"


def test_update_project_rejects_invalid_timezone(client, db):
    p = make_project(db)
    resp = client.patch(
        f"/api/projects/{p.id}",
        json={"timezone": "Foo/Bar"},
    )
    assert resp.status_code == 422


def test_project_media_filter_defaults_to_all(client, db):
    """Existing projects and new ones analyse everything: the setting only
    does something once a user opts in."""
    p = make_project(db)
    resp = client.get(f"/api/projects/{p.id}")
    assert resp.status_code == 200
    assert resp.json()["media_filter"] == "all"


def test_update_project_media_filter(client, db):
    p = make_project(db)
    resp = client.patch(
        f"/api/projects/{p.id}",
        json={"media_filter": "images"},
    )
    assert resp.status_code == 200
    assert resp.json()["media_filter"] == "images"
    db.refresh(p)
    assert p.media_filter == "images"


def test_update_project_rejects_invalid_media_filter(client, db):
    """The worker treats an unrecognised filter as "analyse everything", so a
    typo must be rejected at the door rather than silently ignored later."""
    p = make_project(db)
    resp = client.patch(
        f"/api/projects/{p.id}",
        json={"media_filter": "photos"},
    )
    assert resp.status_code == 422


def test_get_project(client, db):
    p = make_project(db)
    resp = client.get(f"/api/projects/{p.id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == p.id


def test_get_project_not_found(client):
    resp = client.get("/api/projects/nonexistent")
    assert resp.status_code == 404


def test_update_project_name(client, db):
    p = make_project(db)
    resp = client.patch(f"/api/projects/{p.id}", json={"name": "new-name"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "new-name"


def test_update_project_not_found(client):
    resp = client.patch("/api/projects/nonexistent", json={"name": "x"})
    assert resp.status_code == 404


def test_an_explicit_null_clears_the_description_and_an_omitted_one_does_not(
    client, db
):
    """The contract the Edit project dialog relies on to clear a description.

    Both halves matter and they are the whole bug. `update_project` uses
    `model_dump(exclude_unset=True)`, so a key the client did not send
    means "leave this alone", which is correct PATCH behaviour and is
    what kept the description alive. The dialog's zod schema turned an
    emptied box into `undefined`, `JSON.stringify` drops undefined keys,
    and the field therefore never reached this endpoint at all: the
    request was literally `{"name": "ENA24"}`. Sending null is how the
    client says "clear it" out loud.

    The frontend has no test framework, so the line that was actually
    broken cannot be covered. This pins the server side of the contract
    instead, so the value the dialog now sends cannot stop working
    without a red test.
    """
    p = make_project(db, description="Test project for the tutorial")

    resp = client.patch(f"/api/projects/{p.id}", json={"name": "renamed"})
    assert resp.status_code == 200
    assert resp.json()["description"] == "Test project for the tutorial"

    resp = client.patch(f"/api/projects/{p.id}", json={"description": None})
    assert resp.status_code == 200
    assert resp.json()["description"] is None


def test_a_description_over_500_characters_is_refused(client, db):
    """The cap the dialogs enforce, enforced where it cannot be skipped.

    Every dialog caps at 500 twice, in its zod schema and as `maxLength`
    on the textarea, so nothing reachable through the UI changes here.
    Before this the API took any length at all, which left those two
    client-side caps as the only thing between a script and an unbounded
    column. Both entry points are checked because `ProjectCreate`,
    `ProjectUpdate` and `ProjectDuplicate` each declare the field
    separately, and capping only the base class would have missed two of
    the three.
    """
    p = make_project(db)

    resp = client.patch(
        f"/api/projects/{p.id}", json={"description": "a" * 501}
    )
    assert resp.status_code == 422

    resp = client.patch(
        f"/api/projects/{p.id}", json={"description": "a" * 500}
    )
    assert resp.status_code == 200
    assert resp.json()["description"] == "a" * 500


def test_delete_project(client, db):
    p = make_project(db)
    resp = client.delete(f"/api/projects/{p.id}")
    assert resp.status_code == 204


def test_delete_project_not_found(client):
    resp = client.delete("/api/projects/nonexistent")
    assert resp.status_code == 404


def test_delete_project_with_folder_less_deployment(client, db):
    """A deployment with no folder_path must not break the delete.

    `folder_path` is nullable (a deployment can exist before its folder is
    linked) and the handler builds `Path(...)` from it to clean up the
    on-disk cache. `Path(None)` raises, so this used to 500.
    """
    from tests.conftest import make_deployment

    p = make_project(db)
    make_deployment(db, project_id=p.id, folder_path=None)
    db.commit()

    resp = client.delete(f"/api/projects/{p.id}")
    assert resp.status_code == 204


def test_delete_project_removes_jobs_referenced_by_detections(client, db):
    """Jobs must be deleted after the detections that point at them.

    `detections.job_id` and `detection_embeddings.job_id` are NO ACTION
    foreign keys, so deleting a job while a detection still references it is
    an IntegrityError. Nothing pinned that ordering before.
    """
    import uuid

    import numpy as np

    from app.models import DetectionEmbedding, Job
    from tests.conftest import make_deployment, make_detection, make_file, make_job

    p = make_project(db)
    job = make_job(db, payload={"project_id": p.id})
    dep = make_deployment(db, project_id=p.id, folder_path="/fake/jobs-run")
    f = make_file(db, deployment_id=dep.id)
    det = make_detection(db, file_id=f.id, job_id=job.id)
    db.add(
        DetectionEmbedding(
            id=str(uuid.uuid4()),
            detection_id=det.id,
            job_id=job.id,
            embedding_model_id="TEST-EMB",
            vector=np.zeros(8, dtype=np.float16).tobytes(),
            dimension=8,
            l2_norm=0.0,
        )
    )
    db.commit()
    # Hold the id, not the object: after the delete the instance is gone and
    # touching an expired attribute would raise instead of reporting.
    job_id = job.id

    resp = client.delete(f"/api/projects/{p.id}")
    assert resp.status_code == 204

    db.expire_all()
    assert db.query(Job).filter(Job.id == job_id).count() == 0


# --- Stats ---


def test_get_project_stats_empty(client, db):
    p = make_project(db)
    resp = client.get(f"/api/projects/{p.id}/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["site_count"] == 0
    assert data["deployment_count"] == 0


def test_get_project_stats_not_found(client):
    resp = client.get("/api/projects/nonexistent/stats")
    assert resp.status_code == 404


def test_get_detection_stats_empty(client, db):
    p = make_project(db)
    resp = client.get(f"/api/projects/{p.id}/detection-stats")
    assert resp.status_code == 200
    assert resp.json() == {}


def test_get_detection_count_zero(client, db):
    p = make_project(db)
    resp = client.get(f"/api/projects/{p.id}/detection-count")
    assert resp.status_code == 200
    assert resp.json()["count"] == 0


def test_get_label_stats_empty(client, db):
    p = make_project(db)
    resp = client.get(f"/api/projects/{p.id}/label-stats")
    assert resp.status_code == 200
    assert resp.json() == []


def _add_observation(db, *, event_id, label, category="animal", max_n=1, human_count=None):
    from app.models.event_observation import EventObservation

    db.add(
        EventObservation(
            event_id=event_id,
            label=label,
            label_taxonomy_id=None,
            category=category,
            max_n=max_n,
            human_count=human_count,
        )
    )
    db.flush()


def test_independent_observation_stats_honours_human_count(client, db):
    """Abundance uses effective_count (human_count when set, not raw max_n),
    counts human-only species, and excludes person/vehicle."""
    from datetime import datetime

    from tests.conftest import make_deployment, make_event_with_files

    p = make_project(db)
    dep = make_deployment(db, project_id=p.id)
    ev = make_event_with_files(
        db, deployment_id=dep.id, event_start_local=datetime(2024, 1, 1, 12, 0, 0)
    )
    # AI said 2 lions, human confirmed 5.
    _add_observation(db, event_id=ev.id, label="lion", max_n=2, human_count=5)
    # Human added a species the AI missed (max_n=0).
    _add_observation(db, event_id=ev.id, label="fox", max_n=0, human_count=1)
    # Person is not part of label refinement and must be excluded.
    _add_observation(db, event_id=ev.id, label="person", category="person", max_n=4)

    resp = client.get(f"/api/projects/{p.id}/independent-observation-stats")
    assert resp.status_code == 200
    data = resp.json()
    labels = {row["label"]: row["count"] for row in data["labels"]}
    assert labels == {"lion": 5, "fox": 1}
    assert data["total"] == 6


def test_regroup_preview_empty(client, db):
    p = make_project(db)
    resp = client.get(
        f"/api/projects/{p.id}/regroup-preview?independence_interval=1800"
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "confirmed_at_risk": 0,
        "counts_at_risk": 0,
        "total_confirmed": 0,
        "example": None,
    }


# --- Reprocess / Re-embed ---


def test_reprocess_not_found(client):
    resp = client.post("/api/projects/nonexistent/reprocess")
    assert resp.status_code == 404


def test_reprocess_success(client, db):
    p = make_project(db)
    with patch("app.api.routers.projects.ws_manager"):
        resp = client.post(f"/api/projects/{p.id}/reprocess")
    assert resp.status_code == 202
    assert "job_id" in resp.json()


def test_re_embed_no_model(client, db):
    p = make_project(db)
    # Must explicitly set after creation since column default overrides None
    p.embedding_model_id = None
    db.flush()
    resp = client.post(f"/api/projects/{p.id}/re-embed")
    assert resp.status_code == 202
    assert resp.json()["job_id"] is None


def test_re_embed_with_model(client, db):
    p = make_project(db, embedding_model_id="DINOV2-VITB14")
    with patch("app.api.routers.projects.ws_manager"):
        resp = client.post(f"/api/projects/{p.id}/re-embed")
    assert resp.status_code == 202
    assert resp.json()["job_id"] is not None


# --- Postprocessing status ---


def test_postprocessing_status_no_classifications(client, db):
    p = make_project(db)
    resp = client.get(f"/api/projects/{p.id}/postprocessing-status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["has_classifications"] is False


# --- Batch size overrides ---


def test_create_project_default_batch_sizes_are_null(client):
    """A new project should have NULL for all three batch_size fields."""
    resp = client.post(
        "/api/projects", json={"name": "bs-default", "timezone": "UTC"}
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["detection_batch_size"] is None
    assert data["classification_batch_size"] is None
    assert data["embedding_batch_size"] is None


def test_update_project_batch_sizes_persist(client, db):
    p = make_project(db)
    resp = client.patch(
        f"/api/projects/{p.id}",
        json={
            "detection_batch_size": 16,
            "classification_batch_size": 64,
            "embedding_batch_size": 128,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["detection_batch_size"] == 16
    assert data["classification_batch_size"] == 64
    assert data["embedding_batch_size"] == 128

    # Reload via GET to confirm persistence
    resp = client.get(f"/api/projects/{p.id}")
    data = resp.json()
    assert data["detection_batch_size"] == 16
    assert data["classification_batch_size"] == 64
    assert data["embedding_batch_size"] == 128


def test_update_project_batch_size_to_null(client, db):
    """Setting a batch_size back to null reverts to the default."""
    p = make_project(db)
    client.patch(f"/api/projects/{p.id}", json={"detection_batch_size": 16})
    resp = client.patch(f"/api/projects/{p.id}", json={"detection_batch_size": None})
    assert resp.status_code == 200
    assert resp.json()["detection_batch_size"] is None


def test_update_project_batch_size_zero_is_rejected(client, db):
    p = make_project(db)
    resp = client.patch(f"/api/projects/{p.id}", json={"detection_batch_size": 0})
    assert resp.status_code == 422


def test_update_project_batch_size_negative_is_rejected(client, db):
    p = make_project(db)
    resp = client.patch(f"/api/projects/{p.id}", json={"classification_batch_size": -1})
    assert resp.status_code == 422


def test_create_project_default_detection_inference_options(client):
    """A new project has augmentation off and no image-size override."""
    resp = client.post(
        "/api/projects", json={"name": "det-default", "timezone": "UTC"}
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["detection_augment"] is False
    assert data["detection_image_size"] is None


def test_update_project_detection_inference_options_persist(client, db):
    p = make_project(db)
    resp = client.patch(
        f"/api/projects/{p.id}",
        json={"detection_augment": True, "detection_image_size": 1920},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["detection_augment"] is True
    assert data["detection_image_size"] == 1920

    # Reload via GET to confirm persistence
    data = client.get(f"/api/projects/{p.id}").json()
    assert data["detection_augment"] is True
    assert data["detection_image_size"] == 1920


def test_update_project_detection_image_size_to_null(client, db):
    """Clearing the override reverts to the model default (null)."""
    p = make_project(db)
    client.patch(f"/api/projects/{p.id}", json={"detection_image_size": 2560})
    resp = client.patch(f"/api/projects/{p.id}", json={"detection_image_size": None})
    assert resp.status_code == 200
    assert resp.json()["detection_image_size"] is None


def test_update_project_detection_image_size_out_of_range_is_rejected(client, db):
    p = make_project(db)
    resp = client.patch(f"/api/projects/{p.id}", json={"detection_image_size": 10})
    assert resp.status_code == 422


def test_update_project_batch_size_too_large_is_rejected(client, db):
    p = make_project(db)
    resp = client.patch(f"/api/projects/{p.id}", json={"embedding_batch_size": 999})
    assert resp.status_code == 422


def test_reprocess_accepts_folder_run_projects(client, db):
    """The folder-run Labels step's analysis panel PATCHes the project
    and reprocesses through the same endpoints as the Settings page;
    neither may gate on project mode."""
    p = make_project(db, mode="folder_run")

    resp = client.patch(
        f"/api/projects/{p.id}",
        json={"independence_interval": 900, "taxonomic_rollup": False},
    )
    assert resp.status_code == 200
    assert resp.json()["independence_interval"] == 900

    with patch("app.api.routers.projects.ws_manager"):
        resp = client.post(f"/api/projects/{p.id}/reprocess")
    assert resp.status_code == 202
    assert "job_id" in resp.json()


def test_re_embed_accepts_min_confidence_override(client, db):
    """The labels grid's unprocessed-range banner backfills embeddings
    below the classification gate; the override rides on the job
    payload."""
    from app.models import Job

    p = make_project(db, embedding_model_id="DINOV2-VITB14")
    with patch("app.api.routers.projects.ws_manager"):
        resp = client.post(
            f"/api/projects/{p.id}/re-embed",
            json={"min_confidence": 0.02},
        )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    job = db.get(Job, job_id)
    assert job.payload["min_confidence"] == 0.02


# ---------------------------------------------------------------------------
# Counts are gated to what the user can reach
# ---------------------------------------------------------------------------


def _video_with_off_frame_boxes(db):
    """A project holding one video whose best frame is 3, with one box on
    that frame and two on frames nobody can open."""
    from tests.conftest import make_deployment, make_detection, make_file, make_site

    p = make_project(db)
    site = make_site(db, project_id=p.id)
    dep = make_deployment(db, site_id=site.id, project_id=p.id)
    f = make_file(
        db,
        deployment_id=dep.id,
        file_type="video",
        file_format="mp4",
        best_frame_number=3,
    )
    for frame in (3, 7, 11):
        make_detection(
            db, file_id=f.id, confidence=0.9, label="deer", frame_number=frame,
        )
    db.flush()
    return p


def test_detection_count_ignores_off_best_frame_boxes(client, db):
    """A video stores boxes on every sampled frame, but only the best frame
    is written to disk, so the other boxes have no picture to open. Counting
    them made this endpoint report 220 where the Labels grid held 32, and the
    reprocess summary built on it promised changes to unreachable boxes."""
    p = _video_with_off_frame_boxes(db)
    resp = client.get(f"/api/projects/{p.id}/detection-count?threshold=0.2")
    assert resp.status_code == 200
    assert resp.json()["count"] == 1


def test_label_stats_ignores_off_best_frame_boxes(client, db):
    """Same gate, because these counts drive the "Effect on statistics"
    summary shown after a settings change."""
    p = _video_with_off_frame_boxes(db)
    resp = client.get(f"/api/projects/{p.id}/label-stats?threshold=0.2")
    assert resp.status_code == 200
    assert resp.json() == [{"label": "deer", "count": 1}]


def test_verified_off_frame_box_still_counts(client, db):
    """The verified override outranks the frame gate: a human decision must
    never drop out of the numbers, even on a frame with no picture."""
    from app.models import Detection

    p = _video_with_off_frame_boxes(db)
    off = (
        db.query(Detection)
        .filter(Detection.frame_number == 7)
        .one()
    )
    off.verified = True
    db.flush()

    resp = client.get(f"/api/projects/{p.id}/detection-count?threshold=0.2")
    assert resp.json()["count"] == 2


def test_custom_label_reuses_the_builtin_row(client, db):
    """"animal", "person" and "vehicle" already exist as builtin taxonomy
    rows. Creating a custom label of the same name used to add a second,
    rank-less row displaying the identical name, which the label filter then
    showed as two entries nobody could tell apart."""
    from app.ml.taxonomy_db import BUILTIN_MODEL_ID, ensure_builtin_labels
    from app.models.label_taxonomy import LabelTaxonomy

    builtin = ensure_builtin_labels(db)
    p = make_project(db)

    resp = client.post(
        f"/api/projects/{p.id}/custom-labels", json={"name": "vehicle"}
    )
    assert resp.status_code in (200, 201)
    assert resp.json()["id"] == builtin["vehicle"]

    rows = (
        db.query(LabelTaxonomy)
        .filter(LabelTaxonomy.name == "vehicle")
        .all()
    )
    assert len(rows) == 1
    assert rows[0].classification_model_id == BUILTIN_MODEL_ID
