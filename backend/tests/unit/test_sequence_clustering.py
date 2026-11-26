"""Unit tests for the shared event-clustering primitive."""

from datetime import datetime, timedelta

from app.services.event_clustering import cluster_files_into_events
from tests.conftest import make_deployment, make_file, make_project


def _paired_layout(db):
    """Two subfolders whose captures interleave one second apart, plus
    one later file well past the interval. Returns (deployment, files)."""
    p = make_project(db)
    d = make_deployment(db, project_id=p.id, folder_path="/data/station")
    base = datetime(2024, 6, 15, 10, 0, 0)
    schedule = [
        ("cam_a/img_000.jpg", base),
        ("cam_b/img_000.jpg", base + timedelta(seconds=1)),
        ("cam_a/img_001.jpg", base + timedelta(seconds=2)),
        ("cam_b/img_001.jpg", base + timedelta(seconds=3)),
        ("cam_a/img_002.jpg", base + timedelta(hours=2)),
    ]
    files = [
        make_file(
            db,
            deployment_id=d.id,
            file_path=f"/data/station/{rel}",
            captured_at_local=ts,
        )
        for rel, ts in schedule
    ]
    return d, files


def test_unpaired_buckets_by_folder(db):
    """Default rule: subfolders never share an event."""
    _d, files = _paired_layout(db)
    clusters = cluster_files_into_events(files, 1800, paired_cameras=False)
    assert sorted(len(c) for c in clusters) == [1, 2, 2]
    for c in clusters:
        assert len({f.file_path.rsplit("/", 1)[0] for f in c}) == 1


def test_paired_clusters_across_folders_in_time_order(db):
    """Paired cameras: both subfolders are one camera, so the interleaved
    captures form one event ordered by capture time, and the file two
    hours later still starts a new event."""
    _d, files = _paired_layout(db)
    clusters = cluster_files_into_events(files, 1800, paired_cameras=True)
    assert [len(c) for c in clusters] == [4, 1]
    first = clusters[0]
    assert [f.captured_at_local for f in first] == sorted(
        f.captured_at_local for f in first
    )
    assert {f.file_path.rsplit("/", 1)[0] for f in first} == {
        "/data/station/cam_a",
        "/data/station/cam_b",
    }


def test_paired_keeps_dateless_files_as_singletons(db):
    """A file without a capture time cannot be time-grouped, paired or not."""
    d, files = _paired_layout(db)
    dateless = make_file(
        db, deployment_id=d.id, file_path="/data/station/cam_b/img_nodate.jpg"
    )
    dateless.captured_at_local = None  # the factory fills None with a default
    clusters = cluster_files_into_events(files + [dateless], 1800, paired_cameras=True)
    assert [len(c) for c in clusters] == [4, 1, 1]
    assert clusters[-1] == [dateless]
