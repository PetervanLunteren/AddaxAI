"""Unit tests for the folder-aware trap-nights helper."""

from datetime import date, datetime

from app.api.crud.trap_nights import (
    compute_trap_nights_for_deployment,
    compute_trap_nights_for_deployments,
)
from tests.conftest import (
    make_deployment,
    make_file,
    make_project,
    make_site,
)


def _dt(y: int, m: int, d: int, hour: int = 12) -> datetime:
    return datetime(y, m, d, hour, 0, 0)


def test_single_folder_matches_old_formula(db):
    """Clean single-folder case: (max - min + 1). Same as old formula."""
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id, folder_path="/data/deploy")
    for i in range(5):
        make_file(
            db,
            deployment_id=d.id,
            file_path=f"/data/deploy/img_{i:03d}.jpg",
            captured_at_local=_dt(2024, 6, 1 + i),
        )
    db.commit()

    # June 1 .. June 5 = 5 inclusive days
    assert compute_trap_nights_for_deployment(db, d.id) == 5


def test_multi_folder_sums_each_range(db):
    """Backlog across three SD cards spaced a year apart. Each card runs
    10 days. Naive (end - start) would count 375+. Correct answer is 30."""
    p = make_project(db)
    d = make_deployment(
        db, project_id=p.id, folder_path="/data/backlog"
    )
    for i in range(10):
        make_file(
            db,
            deployment_id=d.id,
            file_path=f"/data/backlog/card_a/img_{i:03d}.jpg",
            captured_at_local=_dt(2022, 1, 1 + i),
        )
    for i in range(10):
        make_file(
            db,
            deployment_id=d.id,
            file_path=f"/data/backlog/card_b/img_{i:03d}.jpg",
            captured_at_local=_dt(2023, 1, 1 + i),
        )
    for i in range(10):
        make_file(
            db,
            deployment_id=d.id,
            file_path=f"/data/backlog/card_c/img_{i:03d}.jpg",
            captured_at_local=_dt(2024, 1, 1 + i),
        )
    db.commit()
    assert compute_trap_nights_for_deployment(db, d.id) == 30


def test_empty_deployment_returns_none(db):
    """No files → None, so callers can distinguish 'not analysed yet' from
    'zero nights after clipping'."""
    p = make_project(db)
    d = make_deployment(db, project_id=p.id, folder_path="/data/empty")
    db.commit()
    assert compute_trap_nights_for_deployment(db, d.id) is None


def test_clip_window_narrows_to_zero(db):
    """Files exist but the clip window excludes them → 0, not None."""
    p = make_project(db)
    d = make_deployment(db, project_id=p.id, folder_path="/data/deploy")
    for i in range(3):
        make_file(
            db,
            deployment_id=d.id,
            file_path=f"/data/deploy/img_{i:03d}.jpg",
            captured_at_local=_dt(2024, 6, 1 + i),
        )
    db.commit()
    nights = compute_trap_nights_for_deployment(
        db,
        d.id,
        clip_start=date(2025, 1, 1),
        clip_end=date(2025, 12, 31),
    )
    assert nights == 0


def test_clip_window_partial_overlap(db):
    """Window catches part of the deployment; counts just that slice
    (inclusive, +1 day)."""
    p = make_project(db)
    d = make_deployment(db, project_id=p.id, folder_path="/data/deploy")
    for i in range(10):
        make_file(
            db,
            deployment_id=d.id,
            file_path=f"/data/deploy/img_{i:03d}.jpg",
            captured_at_local=_dt(2024, 6, 1 + i),
        )
    db.commit()
    # Clip to June 3 .. June 7 = 5 days
    nights = compute_trap_nights_for_deployment(
        db,
        d.id,
        clip_start=date(2024, 6, 3),
        clip_end=date(2024, 6, 7),
    )
    assert nights == 5


def test_frame_rows_do_not_contribute(db):
    """Frame rows live inside .addaxai/ and represent pipeline artifacts.
    They should be ignored — only image and video rows drive trap nights."""
    p = make_project(db)
    d = make_deployment(db, project_id=p.id, folder_path="/data/deploy")

    # One video spanning a week
    video = make_file(
        db,
        deployment_id=d.id,
        file_path="/data/deploy/clip.mp4",
        file_type="video",
        file_format="mp4",
        captured_at_local=_dt(2024, 6, 1),
    )
    # Frames attached to the video at later dates should not count
    # (they're pipeline artifacts, not capture events).
    for n in range(5):
        make_file(
            db,
            deployment_id=d.id,
            file_path=(
                f"/data/deploy/.addaxai/projects/{p.id}/video_frames"
                f"/clip.mp4/frame{n:06d}.jpg"
            ),
            file_type="frame",
            file_format="jpg",
            captured_at_local=_dt(2025, 12, 1),  # way later
            source_video_id=video.id,
            source_frame_number=n,
        )
    db.commit()
    # Only the one video contributes → 1 trap night
    assert compute_trap_nights_for_deployment(db, d.id) == 1


def test_video_only_deployment_counts(db):
    """Video-only deployment with multiple video folders — trap nights
    sums per-folder spans, same as images."""
    p = make_project(db)
    d = make_deployment(db, project_id=p.id, folder_path="/data/vids")
    for folder, day in (("loc_a", 1), ("loc_a", 2), ("loc_b", 10), ("loc_b", 12)):
        make_file(
            db,
            deployment_id=d.id,
            file_path=f"/data/vids/{folder}/REC_{day:03d}.mp4",
            file_type="video",
            file_format="mp4",
            captured_at_local=_dt(2024, 6, day),
        )
    db.commit()
    # loc_a: June 1 .. June 2 → 2; loc_b: June 10 .. June 12 → 3; total 5
    assert compute_trap_nights_for_deployment(db, d.id) == 5


def test_rollover_shared_boundary_merges(db):
    """
    Bushnell / Reconyx rollover: `100MEDIA` and `101MEDIA` are the same
    camera running continuously, and the day the camera rolls over sits
    in both folders. Simple per-folder summation with +1 would double-
    count that day. Interval merging collapses it.

    Camera actually runs Jan 1 .. Jan 30 = 30 nights. 100MEDIA covers
    Jan 1 .. Jan 15, 101MEDIA covers Jan 15 .. Jan 30. Pre-merge sum
    would be 15 + 16 = 31; merge should give 30.
    """
    p = make_project(db)
    d = make_deployment(db, project_id=p.id, folder_path="/data/cam")
    for day in range(1, 16):
        make_file(
            db,
            deployment_id=d.id,
            file_path=f"/data/cam/100MEDIA/IMG_{day:04d}.jpg",
            captured_at_local=_dt(2024, 1, day),
        )
    for day in range(15, 31):
        make_file(
            db,
            deployment_id=d.id,
            file_path=f"/data/cam/101MEDIA/IMG_{day:04d}.jpg",
            captured_at_local=_dt(2024, 1, day),
        )
    db.commit()
    assert compute_trap_nights_for_deployment(db, d.id) == 30


def test_overlapping_intervals_merge(db):
    """Two folders whose date ranges overlap by a few days get merged
    into one, not summed. Mirrors the unusual case where a user crammed
    two cameras worth of files into one deployment with partial date
    overlap — we report the calendar span, not the camera-nights total."""
    p = make_project(db)
    d = make_deployment(db, project_id=p.id, folder_path="/data/cam")
    # Folder A: Jan 1 .. Jan 10
    for day in range(1, 11):
        make_file(
            db,
            deployment_id=d.id,
            file_path=f"/data/cam/alpha/img_{day:02d}.jpg",
            captured_at_local=_dt(2024, 1, day),
        )
    # Folder B: Jan 5 .. Jan 15 (overlaps Jan 5..Jan 10 with alpha)
    for day in range(5, 16):
        make_file(
            db,
            deployment_id=d.id,
            file_path=f"/data/cam/beta/img_{day:02d}.jpg",
            captured_at_local=_dt(2024, 1, day),
        )
    db.commit()
    # Merged range [Jan 1, Jan 15] = 15 inclusive days
    assert compute_trap_nights_for_deployment(db, d.id) == 15


def test_adjacent_non_overlapping_intervals_stay_separate(db):
    """Two folders whose intervals are adjacent but don't share a day
    (A = Jan 1..10, B = Jan 11..20) should count 20, not 19 or 21.
    Separate or merged doesn't matter here — either way the count is
    the sum of inclusive spans."""
    p = make_project(db)
    d = make_deployment(db, project_id=p.id, folder_path="/data/cam")
    for day in range(1, 11):
        make_file(
            db,
            deployment_id=d.id,
            file_path=f"/data/cam/a/img_{day:02d}.jpg",
            captured_at_local=_dt(2024, 1, day),
        )
    for day in range(11, 21):
        make_file(
            db,
            deployment_id=d.id,
            file_path=f"/data/cam/b/img_{day:02d}.jpg",
            captured_at_local=_dt(2024, 1, day),
        )
    db.commit()
    assert compute_trap_nights_for_deployment(db, d.id) == 20


def test_bulk_matches_per_deployment(db):
    """Bulk helper agrees with the single-deployment helper for each id."""
    p = make_project(db)
    d1 = make_deployment(db, project_id=p.id, folder_path="/data/d1")
    d2 = make_deployment(db, project_id=p.id, folder_path="/data/d2")
    for i in range(3):
        make_file(
            db,
            deployment_id=d1.id,
            file_path=f"/data/d1/img_{i}.jpg",
            captured_at_local=_dt(2024, 1, 1 + i),
        )
    for i in range(4):
        make_file(
            db,
            deployment_id=d2.id,
            file_path=f"/data/d2/img_{i}.jpg",
            captured_at_local=_dt(2024, 2, 1 + i),
        )
    db.commit()

    bulk = compute_trap_nights_for_deployments(db, [d1.id, d2.id])
    assert bulk == {
        d1.id: compute_trap_nights_for_deployment(db, d1.id),
        d2.id: compute_trap_nights_for_deployment(db, d2.id),
    }
    assert bulk[d1.id] == 3
    assert bulk[d2.id] == 4


def test_bulk_empty_deployment_returns_zero(db):
    """Bulk helper returns 0 (not None / missing key) for empty deployments."""
    p = make_project(db)
    d1 = make_deployment(db, project_id=p.id, folder_path="/data/d1")
    d2 = make_deployment(db, project_id=p.id, folder_path="/data/d2")  # no files
    make_file(
        db,
        deployment_id=d1.id,
        file_path="/data/d1/img.jpg",
        captured_at_local=_dt(2024, 6, 1),
    )
    db.commit()
    bulk = compute_trap_nights_for_deployments(db, [d1.id, d2.id])
    assert bulk == {d1.id: 1, d2.id: 0}
