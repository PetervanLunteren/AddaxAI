"""Unit tests for the shared sequence-clustering primitive."""

from datetime import datetime, timedelta

from app.services.sequence_clustering import (
    SEQUENCE_GAP_SECONDS,
    cluster_files_into_sequences,
)
from tests.conftest import make_deployment, make_file, make_project


def _paired_layout(db):
    """Two subfolders whose captures interleave one second apart, plus
    one file two hours later. Returns (deployment, files)."""
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
    """Default rule: subfolders never share a sequence."""
    _d, files = _paired_layout(db)
    clusters = cluster_files_into_sequences(files, paired_cameras=False)
    assert sorted(len(c) for c in clusters) == [1, 2, 2]
    for c in clusters:
        assert len({f.file_path.rsplit("/", 1)[0] for f in c}) == 1


def test_paired_clusters_across_folders_in_time_order(db):
    """Paired cameras: both subfolders are one camera, so the interleaved
    captures form one sequence ordered by capture time, and the file two
    hours later still starts a new sequence."""
    _d, files = _paired_layout(db)
    clusters = cluster_files_into_sequences(files, paired_cameras=True)
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
    clusters = cluster_files_into_sequences(files + [dateless], paired_cameras=True)
    assert [len(c) for c in clusters] == [4, 1, 1]
    assert clusters[-1] == [dateless]


def _photos(db, seconds: list[float], **file_kw):
    """Photos of one camera folder at the given offsets in seconds."""
    p = make_project(db)
    d = make_deployment(db, project_id=p.id, folder_path="/data/cam")
    base = datetime(2024, 6, 15, 10, 0, 0)
    return [
        make_file(
            db,
            deployment_id=d.id,
            file_path=f"/data/cam/img_{i:03d}.jpg",
            captured_at_local=base + timedelta(seconds=s),
            **file_kw,
        )
        for i, s in enumerate(seconds)
    ]


def _sizes(files) -> list[int]:
    return [len(c) for c in cluster_files_into_sequences(files, paired_cameras=False)]


def test_a_gap_of_exactly_the_sequence_gap_joins_and_one_second_more_splits(db):
    assert SEQUENCE_GAP_SECONDS == 120
    assert _sizes(_photos(db, [0, 120])) == [2]
    assert _sizes(_photos(db, [0, 121])) == [1, 1]


def test_a_time_lapse_cadence_never_chains_into_one_sequence(db):
    """Three photos every five minutes for a day: the case that made one
    30-minute event of a whole folder. Each burst is its own sequence."""
    offsets = [burst * 300 + shot for burst in range(288) for shot in (0, 1, 2)]
    assert _sizes(_photos(db, offsets)) == [3] * 288


def test_a_video_is_measured_from_its_end(db):
    """GoPro chapters of one drop start minutes apart but follow each other
    with no gap, so they are one sequence. Measured start to start they
    would be three."""
    chapters = _photos(db, [0, 530, 1060], file_type="video", duration_seconds=530.0)
    assert _sizes(chapters) == [3]


def test_a_video_without_a_length_counts_as_a_moment(db):
    clips = _photos(db, [0, 530], file_type="video", duration_seconds=None)
    assert _sizes(clips) == [1, 1]


def test_the_gap_runs_from_the_latest_end_not_the_previous_file(db):
    """With paired cameras a long clip on one camera outlasts a photo on the
    other; a photo after that photo but inside the clip joins the sequence."""
    p = make_project(db)
    d = make_deployment(db, project_id=p.id, folder_path="/data/pair")
    base = datetime(2024, 6, 15, 10, 0, 0)
    clip = make_file(
        db, deployment_id=d.id, file_path="/data/pair/a/clip.mp4", file_type="video",
        captured_at_local=base, duration_seconds=600.0,
    )
    early = make_file(
        db, deployment_id=d.id, file_path="/data/pair/b/1.jpg",
        captured_at_local=base + timedelta(seconds=10),
    )
    late = make_file(
        db, deployment_id=d.id, file_path="/data/pair/b/2.jpg",
        captured_at_local=base + timedelta(seconds=500),
    )
    clusters = cluster_files_into_sequences([clip, early, late], paired_cameras=True)
    assert clusters == [[clip, early, late]]
