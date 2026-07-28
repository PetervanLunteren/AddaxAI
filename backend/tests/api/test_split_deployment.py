"""Tests for POST /api/deployments/{id}/split and its preview endpoint."""

import json
from datetime import date, datetime
from pathlib import Path

from sqlalchemy import func, select

from app.api.crud.deployment_split import split_deployment as crud_split_deployment
from app.models import Deployment, DeploymentQueue, Event, EventObservation, File
from tests.conftest import (
    make_deployment,
    make_file,
    make_job,
    make_project,
    make_site,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_tree(tmp_path: Path, layout: dict[str, int]) -> Path:
    """
    Create a folder tree under tmp_path.

    layout maps relative subfolder path -> number of image files to create
    directly inside that subfolder. Parents are created as needed.

    Returns the root folder (tmp_path / "parent").
    """
    root = tmp_path / "parent"
    root.mkdir()
    for rel, count in layout.items():
        sub = root / rel
        sub.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            (sub / f"img_{i:03d}.jpg").write_bytes(b"\x00")
    return root


def _seed_deployment_with_files(
    db, tmp_path: Path, layout: dict[str, int]
) -> tuple[Path, Deployment]:
    """
    Create a project + deployment whose folder_path matches an on-disk tree
    built from `layout`. Registers one File row per on-disk image.

    Returns (root_folder, deployment).
    """
    root = _build_tree(tmp_path, layout)
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(
        db,
        site_id=s.id,
        folder_path=str(root),
        start_date_local=date(2024, 1, 1),
        end_date_local=date(2024, 1, 10),
        camera_model="CamX",
        camera_serial="SN-42",
        notes="shared note",
        tags={"region": "NL"},
        datetime_offset_seconds=123,
    )
    # Point File rows at the on-disk files. Timestamps are spaced by 1 hour
    # so straddling tests can rely on deterministic bounds.
    ts_base = datetime(2024, 1, 5, 10, 0, 0)
    tick = 0
    for rel, count in layout.items():
        for i in range(count):
            make_file(
                db,
                deployment_id=d.id,
                file_path=str(root / rel / f"img_{i:03d}.jpg"),
                captured_at_local=ts_base.replace(hour=10 + (tick % 10)),
            )
            tick += 1
    db.commit()
    return root, d


def _seed_parent_results_json(
    root: Path, project_id: str, files: list[File]
) -> Path:
    """Write a minimal parent results.json that lists every file."""
    artifacts = root / ".addaxai" / "projects" / project_id
    artifacts.mkdir(parents=True, exist_ok=True)
    images = []
    for f in files:
        rel = Path(f.file_path).relative_to(root)
        images.append({"file": str(rel), "detections": []})
    payload = {
        "info": {"detector": "md5a"},
        "detection_categories": {"1": "animal"},
        "classification_categories": {"1": "deer"},
        "images": images,
    }
    json_path = artifacts / "results.json"
    json_path.write_text(json.dumps(payload))
    return json_path


# ---------------------------------------------------------------------------
# Preview endpoint
# ---------------------------------------------------------------------------


def test_split_preview_happy_path(client, db, tmp_path):
    root, d = _seed_deployment_with_files(
        db, tmp_path, {"siteA": 3, "siteB": 2, "siteC": 1}
    )

    resp = client.get(f"/api/deployments/{d.id}/split-preview?depth=1")
    assert resp.status_code == 200
    data = resp.json()

    assert data["original_folder"] == str(root)
    assert data["depth"] == 1
    assert data["max_depth"] == 1
    assert data["can_decrease"] is False
    assert data["can_increase"] is False
    assert data["blocked_reason"] is None

    names = sorted(t["name"] for t in data["targets"])
    assert names == ["siteA", "siteB", "siteC"]
    counts = {t["name"]: t["image_count"] for t in data["targets"]}
    assert counts == {"siteA": 3, "siteB": 2, "siteC": 1}


def test_split_preview_empty_subfolder_omitted(client, db, tmp_path):
    root, d = _seed_deployment_with_files(
        db, tmp_path, {"siteA": 3, "siteB": 0}
    )
    # siteB exists on disk (dir created) but has no files in DB.
    (root / "siteB").mkdir(exist_ok=True)

    resp = client.get(f"/api/deployments/{d.id}/split-preview?depth=1")
    data = resp.json()
    assert [t["name"] for t in data["targets"]] == ["siteA"]
    # One non-empty child => targets reported as-is. No scary blocked_reason;
    # the frontend disables OK for targets <= 1.
    assert data["blocked_reason"] is None


def test_split_preview_clamp_per_branch(client, db, tmp_path):
    root, d = _seed_deployment_with_files(
        db,
        tmp_path,
        {"siteA/cam1": 2, "siteA/cam2": 2, "siteB": 3},
    )

    resp = client.get(f"/api/deployments/{d.id}/split-preview?depth=2")
    data = resp.json()
    names = sorted(t["name"] for t in data["targets"])
    # siteA descends to cam1/cam2 (depth 2). siteB has no children so it
    # clamps at its own leaf (siteB).
    assert names == ["cam1", "cam2", "siteB"]


def test_split_preview_linear_chain_reports_one_target(client, db, tmp_path):
    """Linear chain (a/b/c): preview reports the single target at whatever
    depth. No blocked_reason — the frontend disables OK via target count.
    The user can see what would happen, and no warning implies data is broken."""
    root, d = _seed_deployment_with_files(
        db, tmp_path, {"a/b/c": 3}
    )

    for depth in (1, 2, 3):
        resp = client.get(
            f"/api/deployments/{d.id}/split-preview?depth={depth}"
        )
        data = resp.json()
        assert data["blocked_reason"] is None
        assert len(data["targets"]) == 1


def test_split_preview_one_target_when_fork_is_deeper(client, db, tmp_path):
    """Tree forks below depth 1: at shallow depths targets.length==1,
    deeper depths reveal the fork. No scary warning at shallow depths."""
    root, d = _seed_deployment_with_files(
        db, tmp_path, {"only_branch/cam1": 1, "only_branch/cam2": 1}
    )

    resp = client.get(f"/api/deployments/{d.id}/split-preview?depth=1")
    data = resp.json()
    assert data["blocked_reason"] is None
    assert len(data["targets"]) == 1

    resp = client.get(f"/api/deployments/{d.id}/split-preview?depth=2")
    data = resp.json()
    assert data["blocked_reason"] is None
    assert sorted(t["name"] for t in data["targets"]) == ["cam1", "cam2"]


def test_split_preview_clamps_when_files_live_at_parent_level(
    client, db, tmp_path
):
    """
    A branch whose files live directly at depth N (not inside subfolders)
    must clamp at N. If we blindly descended into empty subfolders we'd
    orphan those files and report fewer deployments at deeper depths.

    Layout: siteA has subfolders cam1/cam2 holding all its files, siteB
    has files directly in it. At depth 2, siteA descends into cam1/cam2
    (2 targets), siteB clamps at itself (1 target). Total 3 — same as
    depth 3, 4, 5, ... because siteB can never descend further.
    """
    root, d = _seed_deployment_with_files(
        db,
        tmp_path,
        {"siteA/cam1": 2, "siteA/cam2": 2, "siteB": 3},
    )

    # Depth 1 → [siteA, siteB] = 2 targets.
    resp = client.get(f"/api/deployments/{d.id}/split-preview?depth=1")
    depth1_names = sorted(t["name"] for t in resp.json()["targets"])
    assert depth1_names == ["siteA", "siteB"]

    # Depth 2 → [cam1, cam2, siteB] = 3 targets.
    resp = client.get(f"/api/deployments/{d.id}/split-preview?depth=2")
    depth2_names = sorted(t["name"] for t in resp.json()["targets"])
    assert depth2_names == ["cam1", "cam2", "siteB"]

    # Depth 3 must NOT regress: siteA/cam1 and siteA/cam2 have no children,
    # so they also clamp at themselves. siteB stays clamped. Same 3 targets.
    resp = client.get(f"/api/deployments/{d.id}/split-preview?depth=3")
    depth3_names = sorted(t["name"] for t in resp.json()["targets"])
    assert depth3_names == ["cam1", "cam2", "siteB"]


def test_split_honours_clamp_when_executing(client, db, tmp_path):
    """End-to-end: a depth beyond one branch's clamp must still succeed and
    preserve every file, with the stuck branch landing at its clamp point."""
    root, d = _seed_deployment_with_files(
        db,
        tmp_path,
        {"siteA/cam1": 1, "siteA/cam2": 1, "siteB": 2},
    )
    files = list(db.execute(select(File)).scalars())
    _seed_parent_results_json(root, d.project_id, files)

    resp = client.post(
        f"/api/deployments/{d.id}/split", json={"depth": 3}
    )
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["created_deployment_ids"]) == 3

    db.expire_all()
    children = list(db.execute(select(Deployment)).scalars())
    folders = sorted(c.folder_path for c in children)
    assert folders == sorted(
        [str(root / "siteA" / "cam1"),
         str(root / "siteA" / "cam2"),
         str(root / "siteB")]
    )
    # No files lost.
    assert db.scalar(select(func.count(File.id))) == 4


def test_split_preview_blocked_when_needs_relink(client, db, tmp_path):
    root, d = _seed_deployment_with_files(db, tmp_path, {"a": 1, "b": 1})
    d.folder_status = "needs_relink"
    db.commit()

    resp = client.get(f"/api/deployments/{d.id}/split-preview?depth=1")
    assert resp.status_code == 200
    data = resp.json()
    assert "needs_relink" in (data["blocked_reason"] or "")


def test_split_preview_blocked_by_active_job(client, db, tmp_path):
    root, d = _seed_deployment_with_files(db, tmp_path, {"a": 1, "b": 1})
    make_job(
        db,
        job_type="postprocessing",
        status="running",
        payload={"project_id": d.project_id},
    )
    db.commit()

    resp = client.get(f"/api/deployments/{d.id}/split-preview?depth=1")
    data = resp.json()
    assert data["blocked_reason"] is not None
    assert "postprocessing" in data["blocked_reason"]


# ---------------------------------------------------------------------------
# Split endpoint — happy paths
# ---------------------------------------------------------------------------


def test_split_creates_children_and_removes_parent(client, db, tmp_path):
    root, d = _seed_deployment_with_files(
        db, tmp_path, {"siteA": 2, "siteB": 3}
    )
    files = list(db.execute(select(File)).scalars())
    _seed_parent_results_json(root, d.project_id, files)

    original_id = d.id
    parent_project_id = d.project_id
    parent_site_id = d.site_id

    resp = client.post(
        f"/api/deployments/{d.id}/split", json={"depth": 1}
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data["created_deployment_ids"]) == 2

    db.expire_all()
    assert db.get(Deployment, original_id) is None

    children = list(
        db.execute(
            select(Deployment).where(
                Deployment.project_id == parent_project_id
            )
        ).scalars()
    )
    assert len(children) == 2
    child_by_folder = {c.folder_path: c for c in children}
    assert str(root / "siteA") in child_by_folder
    assert str(root / "siteB") in child_by_folder

    # Files reassigned.
    for child in children:
        child_files = (
            db.execute(
                select(File).where(File.deployment_id == child.id)
            )
            .scalars()
            .all()
        )
        for f in child_files:
            assert f.file_path.startswith(child.folder_path + "/")

    # Inherited metadata; site carries over from parent; dates recomputed.
    for child in children:
        assert child.site_id == parent_site_id
        assert child.camera_model == "CamX"
        assert child.camera_serial == "SN-42"
        assert child.notes == "shared note"
        assert child.tags == {"region": "NL"}
        assert child.datetime_offset_seconds == 123
        assert child.folder_status == "valid"
        assert child.start_date_local == date(2024, 1, 5)
        assert child.end_date_local == date(2024, 1, 5)


def test_split_inherits_no_site_from_parent(client, db, tmp_path):
    """Parent with site_id=NULL produces children with site_id=NULL too.
    Split always carries the parent's site across, whatever it is."""
    # Seed a deployment with no site.
    root = _build_tree(tmp_path, {"siteA": 1, "siteB": 1})
    p = make_project(db)
    d = make_deployment(
        db,
        project_id=p.id,
        site_id=None,
        folder_path=str(root),
        start_date_local=date(2024, 1, 1),
    )
    make_file(
        db,
        deployment_id=d.id,
        file_path=str(root / "siteA" / "img_000.jpg"),
        captured_at_local=datetime(2024, 1, 5, 10),
    )
    make_file(
        db,
        deployment_id=d.id,
        file_path=str(root / "siteB" / "img_000.jpg"),
        captured_at_local=datetime(2024, 1, 5, 11),
    )
    db.commit()
    files = list(db.execute(select(File)).scalars())
    _seed_parent_results_json(root, d.project_id, files)

    resp = client.post(
        f"/api/deployments/{d.id}/split", json={"depth": 1}
    )
    assert resp.status_code == 200, resp.text

    db.expire_all()
    children = list(db.execute(select(Deployment)).scalars())
    assert len(children) == 2
    for child in children:
        assert child.site_id is None


def test_split_slices_results_json_and_removes_parent_dir(
    client, db, tmp_path
):
    root, d = _seed_deployment_with_files(
        db, tmp_path, {"siteA": 2, "siteB": 1}
    )
    files = list(db.execute(select(File)).scalars())
    parent_json = _seed_parent_results_json(root, d.project_id, files)
    assert parent_json.exists()

    resp = client.post(
        f"/api/deployments/{d.id}/split", json={"depth": 1}
    )
    assert resp.status_code == 200

    # Parent's .addaxai has been cleaned up.
    assert not (root / ".addaxai").exists()

    # Each child has its own slice with rewritten relative paths.
    for sub in ("siteA", "siteB"):
        child_json_path = (
            root / sub / ".addaxai" / "projects" / d.project_id
            / "results.json"
        )
        assert child_json_path.exists()
        payload = json.loads(child_json_path.read_text())
        for img in payload["images"]:
            assert not img["file"].startswith(sub + "/")
            assert img["file"].startswith("img_")
        # Top-level metadata is preserved.
        assert payload["info"] == {"detector": "md5a"}
        assert payload["classification_categories"] == {"1": "deer"}


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def test_split_reassigns_single_group_event(client, db, tmp_path):
    root, d = _seed_deployment_with_files(
        db, tmp_path, {"siteA": 2, "siteB": 2}
    )
    # Event linking two siteA files only.
    a_files = list(
        db.execute(
            select(File).where(File.file_path.like(f"{root}/siteA/%"))
        ).scalars()
    )
    from app.models.event import event_files
    ev = Event(
        deployment_id=d.id,
        event_start_local=datetime(2024, 1, 5, 10, 0),
        event_end_local=datetime(2024, 1, 5, 10, 30),
        file_count=len(a_files),
    )
    db.add(ev)
    db.flush()
    for seq, f in enumerate(a_files):
        db.execute(
            event_files.insert().values(
                event_id=ev.id, file_id=f.id, sequence_number=seq
            )
        )
    db.add(
        EventObservation(
            event_id=ev.id, label="deer", category="animal", max_n=1
        )
    )
    db.commit()

    files = list(db.execute(select(File)).scalars())
    _seed_parent_results_json(root, d.project_id, files)

    resp = client.post(
        f"/api/deployments/{d.id}/split", json={"depth": 1}
    )
    assert resp.status_code == 200

    # The event moved to the siteA child.
    children = list(db.execute(select(Deployment)).scalars())
    site_a_child = next(
        c for c in children if c.folder_path == str(root / "siteA")
    )
    events_for_a = list(
        db.execute(
            select(Event).where(Event.deployment_id == site_a_child.id)
        ).scalars()
    )
    assert len(events_for_a) == 1
    obs = list(
        db.execute(
            select(EventObservation).where(
                EventObservation.event_id == events_for_a[0].id
            )
        ).scalars()
    )
    assert len(obs) == 1
    assert obs[0].label == "deer"


def test_split_duplicates_straddling_event(client, db, tmp_path):
    root, d = _seed_deployment_with_files(
        db, tmp_path, {"siteA": 2, "siteB": 2}
    )
    files = list(db.execute(select(File)).scalars())
    # Attach an event that straddles both children.
    from app.models.event import event_files
    ev = Event(
        deployment_id=d.id,
        event_start_local=datetime(2024, 1, 5, 10, 0),
        event_end_local=datetime(2024, 1, 5, 10, 30),
        file_count=len(files),
    )
    db.add(ev)
    db.flush()
    for seq, f in enumerate(files):
        db.execute(
            event_files.insert().values(
                event_id=ev.id, file_id=f.id, sequence_number=seq
            )
        )
    peak_file = next(
        f for f in files if f.file_path.startswith(str(root / "siteA"))
    )
    db.add(
        EventObservation(
            event_id=ev.id,
            label="deer",
            category="animal",
            max_n=3,
            max_n_file_id=peak_file.id,
        )
    )
    db.commit()

    _seed_parent_results_json(root, d.project_id, files)

    resp = client.post(
        f"/api/deployments/{d.id}/split", json={"depth": 1}
    )
    assert resp.status_code == 200

    children = list(db.execute(select(Deployment)).scalars())
    by_folder = {c.folder_path: c for c in children}
    ev_a = list(
        db.execute(
            select(Event).where(
                Event.deployment_id == by_folder[str(root / "siteA")].id
            )
        ).scalars()
    )
    ev_b = list(
        db.execute(
            select(Event).where(
                Event.deployment_id == by_folder[str(root / "siteB")].id
            )
        ).scalars()
    )
    assert len(ev_a) == 1
    assert len(ev_b) == 1

    obs_a = list(
        db.execute(
            select(EventObservation).where(
                EventObservation.event_id == ev_a[0].id
            )
        ).scalars()
    )
    obs_b = list(
        db.execute(
            select(EventObservation).where(
                EventObservation.event_id == ev_b[0].id
            )
        ).scalars()
    )
    assert len(obs_a) == 1 and obs_a[0].max_n == 3
    assert len(obs_b) == 1 and obs_b[0].max_n == 3
    # Peak-file pointer follows the file it references.
    assert obs_a[0].max_n_file_id == peak_file.id
    assert obs_b[0].max_n_file_id is None


def test_split_preserves_human_count_and_confirmation(client, db, tmp_path):
    """Human counts and the confirmed flag are holy: a split copies
    observations, so a confirmed event with a human count override must
    survive on every child."""
    root, d = _seed_deployment_with_files(
        db, tmp_path, {"siteA": 2, "siteB": 2}
    )
    files = list(db.execute(select(File)).scalars())
    from app.models.event import event_files
    ev = Event(
        deployment_id=d.id,
        event_start_local=datetime(2024, 1, 5, 10, 0),
        event_end_local=datetime(2024, 1, 5, 10, 30),
        file_count=len(files),
        confirmed=True,
    )
    db.add(ev)
    db.flush()
    for seq, f in enumerate(files):
        db.execute(
            event_files.insert().values(
                event_id=ev.id, file_id=f.id, sequence_number=seq
            )
        )
    # AI counted 3, human overrode to 7.
    db.add(
        EventObservation(
            event_id=ev.id,
            label="deer",
            category="animal",
            max_n=3,
            human_count=7,
        )
    )
    db.commit()

    _seed_parent_results_json(root, d.project_id, files)

    resp = client.post(f"/api/deployments/{d.id}/split", json={"depth": 1})
    assert resp.status_code == 200

    children = list(db.execute(select(Deployment)).scalars())
    by_folder = {c.folder_path: c for c in children}
    for folder in (str(root / "siteA"), str(root / "siteB")):
        evs = list(
            db.execute(
                select(Event).where(
                    Event.deployment_id == by_folder[folder].id
                )
            ).scalars()
        )
        assert len(evs) == 1
        assert evs[0].confirmed is True
        obs = list(
            db.execute(
                select(EventObservation).where(
                    EventObservation.event_id == evs[0].id
                )
            ).scalars()
        )
        assert len(obs) == 1
        assert obs[0].human_count == 7
        assert obs[0].effective_count == 7


# ---------------------------------------------------------------------------
# Best frame path rewrite
# ---------------------------------------------------------------------------


def test_split_handles_video_only_project(client, db, tmp_path):
    """
    Video-only projects no longer hold `file_type='frame'` rows
    post-2026-05: detections live on the parent video File and the only
    artifact under `.addaxai/` is the single best-frame JPEG per video.
    Splitting a video-only project must rewrite each video's
    `best_frame_path` into its new child layout and move the JPEG on disk.
    """
    root = tmp_path / "parent"
    root.mkdir()
    (root / "siteA").mkdir()
    (root / "siteB").mkdir()
    vid_a = root / "siteA" / "REC001.mp4"
    vid_b = root / "siteB" / "REC002.mp4"
    vid_a.write_bytes(b"\x00")
    vid_b.write_bytes(b"\x00")

    p = make_project(db)
    d = make_deployment(
        db,
        project_id=p.id,
        folder_path=str(root),
        start_date_local=date(2024, 1, 1),
    )
    # Seed best-frame JPEGs under parent's .addaxai.
    parent_artifacts = root / ".addaxai" / "projects" / d.project_id
    bf_a = parent_artifacts / "video_frames" / "siteA" / "REC001.mp4" / "frame000042.jpg"
    bf_b = parent_artifacts / "video_frames" / "siteB" / "REC002.mp4" / "frame000017.jpg"
    for bf in (bf_a, bf_b):
        bf.parent.mkdir(parents=True, exist_ok=True)
        bf.write_bytes(b"\x00")

    v_a = make_file(
        db,
        deployment_id=d.id,
        file_path=str(vid_a),
        file_type="video",
        file_format="mp4",
        captured_at_local=datetime(2024, 1, 5, 10),
        best_frame_number=42,
        best_frame_path=str(bf_a),
    )
    v_b = make_file(
        db,
        deployment_id=d.id,
        file_path=str(vid_b),
        file_type="video",
        file_format="mp4",
        captured_at_local=datetime(2024, 1, 5, 11),
        best_frame_number=17,
        best_frame_path=str(bf_b),
    )
    db.commit()

    # Minimal parent results.json covering just the two videos.
    _seed_parent_results_json(
        root, d.project_id, [v_a, v_b]
    )

    # Preview should now report the two video-only subfolders.
    resp = client.get(f"/api/deployments/{d.id}/split-preview?depth=1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["blocked_reason"] is None
    counts = {t["name"]: (t["image_count"], t["video_count"]) for t in data["targets"]}
    assert counts == {"siteA": (0, 1), "siteB": (0, 1)}

    # Execute split.
    resp = client.post(
        f"/api/deployments/{d.id}/split", json={"depth": 1}
    )
    assert resp.status_code == 200, resp.text

    # Best-frame JPEGs moved on disk to each child's .addaxai.
    for sub, expected_video, expected_frame in (
        ("siteA", "REC001.mp4", "frame000042.jpg"),
        ("siteB", "REC002.mp4", "frame000017.jpg"),
    ):
        child_frame = (
            root / sub / ".addaxai" / "projects" / d.project_id
            / "video_frames" / expected_video / expected_frame
        )
        assert child_frame.exists(), f"missing best frame {child_frame}"

    # No more `file_type='frame'` rows are produced anywhere.
    db.expire_all()
    frames = db.execute(
        select(File).where(File.file_type == "frame")
    ).scalars().all()
    assert frames == []

    # Video best_frame_path rewritten to point under siteA's child layout.
    reloaded_va = db.get(File, v_a.id)
    assert reloaded_va.best_frame_path is not None
    assert "siteA" in reloaded_va.best_frame_path
    assert Path(reloaded_va.best_frame_path).exists()


def test_split_rewrites_best_frame_path(client, db, tmp_path):
    root, d = _seed_deployment_with_files(
        db, tmp_path, {"siteA": 1, "siteB": 1}
    )
    # Promote siteA's image to a "video" by faking best_frame_number +
    # best_frame_path under the parent's .addaxai layout.
    site_a_file = db.execute(
        select(File).where(File.file_path.like(f"{root}/siteA/%"))
    ).scalar_one()
    site_a_file.file_type = "video"
    site_a_file.best_frame_number = 42
    frame_relative = (
        Path(site_a_file.file_path).relative_to(root)
    )
    frame_dir = (
        root / ".addaxai" / "projects" / d.project_id / "video_frames"
        / frame_relative
    )
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_file = frame_dir / "frame000042.jpg"
    frame_file.write_bytes(b"\x00")
    site_a_file.best_frame_path = str(frame_file)
    db.commit()

    files = list(db.execute(select(File)).scalars())
    _seed_parent_results_json(root, d.project_id, files)

    resp = client.post(
        f"/api/deployments/{d.id}/split", json={"depth": 1}
    )
    assert resp.status_code == 200, resp.text

    db.expire_all()
    reloaded = db.execute(
        select(File).where(File.id == site_a_file.id)
    ).scalar_one()
    expected = (
        root / "siteA" / ".addaxai" / "projects" / d.project_id
        / "video_frames" / Path(reloaded.file_path).name / "frame000042.jpg"
    )
    assert reloaded.best_frame_path == str(expected)
    assert Path(reloaded.best_frame_path).exists()


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_split_nulls_out_completed_queue_entry_deployment_id(
    client, db, tmp_path
):
    """A completed DeploymentQueue row that points at the parent must have
    its deployment_id nulled after split — the referenced row no longer
    exists, and the FK is only a plain string column (no cascade)."""
    root, d = _seed_deployment_with_files(
        db, tmp_path, {"siteA": 1, "siteB": 1}
    )
    files = list(db.execute(select(File)).scalars())
    _seed_parent_results_json(root, d.project_id, files)

    q = DeploymentQueue(
        project_id=d.project_id,
        folder_path=str(root),
        status="completed",
        deployment_id=d.id,
    )
    db.add(q)
    db.commit()
    q_id = q.id

    resp = client.post(
        f"/api/deployments/{d.id}/split", json={"depth": 1}
    )
    assert resp.status_code == 200

    db.expire_all()
    reloaded = db.get(DeploymentQueue, q_id)
    assert reloaded is not None
    assert reloaded.deployment_id is None
    # Rest of the queue row preserved for audit.
    assert reloaded.status == "completed"
    assert reloaded.folder_path == str(root)


def test_split_blocked_by_queue_entry(client, db, tmp_path):
    root, d = _seed_deployment_with_files(
        db, tmp_path, {"siteA": 1, "siteB": 1}
    )
    db.add(
        DeploymentQueue(
            project_id=d.project_id,
            folder_path=str(root),
            status="pending",
        )
    )
    db.commit()

    resp = client.post(
        f"/api/deployments/{d.id}/split", json={"depth": 1}
    )
    assert resp.status_code == 409


def test_split_blocked_when_needs_relink(client, db, tmp_path):
    root, d = _seed_deployment_with_files(
        db, tmp_path, {"siteA": 1, "siteB": 1}
    )
    d.folder_status = "needs_relink"
    db.commit()

    resp = client.post(
        f"/api/deployments/{d.id}/split", json={"depth": 1}
    )
    assert resp.status_code == 409


def test_split_rejects_single_non_empty_branch(client, db, tmp_path):
    root, d = _seed_deployment_with_files(db, tmp_path, {"siteA": 2})

    resp = client.post(
        f"/api/deployments/{d.id}/split", json={"depth": 1}
    )
    assert resp.status_code == 400


def test_split_tolerates_duplicate_entries_in_parent_json(
    client, db, tmp_path
):
    """
    Parent results.json in the wild sometimes carries duplicate entries for
    the same physical file (re-runs merged into the same JSON). The DB
    deduplicates via uniqueness on (file_path, deployment_id) but the JSON
    does not. The split must succeed as long as every File row is covered
    by at least one entry; extra / duplicate entries pass through to the
    child slice verbatim.
    """
    root, d = _seed_deployment_with_files(
        db, tmp_path, {"siteA": 2, "siteB": 1}
    )
    files = list(db.execute(select(File)).scalars())
    parent_json_path = _seed_parent_results_json(root, d.project_id, files)
    # Inject a duplicate for the first siteA file.
    parent_json = json.loads(parent_json_path.read_text())
    dup = dict(parent_json["images"][0])
    parent_json["images"].append(dup)
    parent_json_path.write_text(json.dumps(parent_json))

    resp = client.post(
        f"/api/deployments/{d.id}/split", json={"depth": 1}
    )
    assert resp.status_code == 200, resp.text

    # siteA child's results.json still has the duplicate.
    a_json_path = (
        root / "siteA" / ".addaxai" / "projects" / d.project_id
        / "results.json"
    )
    a_images = json.loads(a_json_path.read_text())["images"]
    assert len(a_images) == 3  # 2 unique siteA files + 1 duplicate
    unique_files = {img["file"] for img in a_images}
    assert len(unique_files) == 2


def test_split_fails_when_file_missing_from_parent_json(
    client, db, tmp_path
):
    """If a File row has no corresponding entry in parent results.json we
    would otherwise silently produce a child whose DB and JSON disagree.
    Catch that here."""
    root, d = _seed_deployment_with_files(
        db, tmp_path, {"siteA": 2, "siteB": 1}
    )
    files = list(db.execute(select(File)).scalars())
    parent_json_path = _seed_parent_results_json(root, d.project_id, files)
    # Remove the first siteA image from results.json while leaving the File row.
    parent_json = json.loads(parent_json_path.read_text())
    parent_json["images"] = parent_json["images"][1:]
    parent_json_path.write_text(json.dumps(parent_json))

    resp = client.post(
        f"/api/deployments/{d.id}/split", json={"depth": 1}
    )
    assert resp.status_code == 400
    assert "missing" in resp.json()["detail"].lower()


def test_split_rolls_back_on_validation_failure(
    client, db, tmp_path, monkeypatch
):
    """If child artifact validation fails, DB stays unchanged and child
    .addaxai dirs are scrubbed."""
    root, d = _seed_deployment_with_files(
        db, tmp_path, {"siteA": 2, "siteB": 1}
    )
    files = list(db.execute(select(File)).scalars())
    _seed_parent_results_json(root, d.project_id, files)

    from app.api.crud import deployment_split

    def _broken_validate(*_args, **_kwargs):
        raise deployment_split.SplitError("forced failure for test")

    monkeypatch.setattr(
        deployment_split, "_validate_child_artifacts", _broken_validate
    )

    resp = client.post(
        f"/api/deployments/{d.id}/split", json={"depth": 1}
    )
    assert resp.status_code == 400

    # Parent still exists, no child rows created.
    db.expire_all()
    assert db.get(Deployment, d.id) is not None
    assert len(db.execute(select(Deployment)).scalars().all()) == 1

    # Children's .addaxai dirs were cleaned up.
    assert not (root / "siteA" / ".addaxai").exists()
    assert not (root / "siteB" / ".addaxai").exists()
    # Parent's .addaxai is intact.
    assert (root / ".addaxai" / "projects" / d.project_id / "results.json").exists()


def test_crud_split_not_found(db):
    from app.api.crud.deployment_split import SplitError

    try:
        crud_split_deployment(db, "nonexistent", 1)
    except SplitError as exc:
        assert exc.status_code == 404
    else:
        raise AssertionError("expected SplitError")
