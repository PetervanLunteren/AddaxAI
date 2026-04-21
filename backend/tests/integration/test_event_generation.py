"""
Integration tests: event generation (clustering from files).

Tests generate_events_for_project() after loading JSON to DB.
No subprocess mocks needed — cv2 works on 1x1 JPEGs.
"""

from datetime import datetime, timedelta
from unittest.mock import patch

from app.api.crud.event import generate_events_for_project
from app.ml.json_pipeline import load_json_to_database
from app.models import Event

from .conftest import (
    build_detection_json,
    create_tiny_jpeg,
    create_video_frames,
    write_json,
)


def _load_images_with_timestamps(s: dict, timestamps: list[datetime]) -> None:
    """Create tiny JPEGs with EXIF timestamps and load to DB."""
    db, deploy_dir = s["db"], s["deploy_dir"]

    images = []
    for i, ts in enumerate(timestamps):
        img_path = create_tiny_jpeg(deploy_dir / f"timed_{i:03d}.jpg")
        rel = str(img_path.relative_to(deploy_dir))
        images.append({
            "file": rel,
            "exif_metadata": {
                "DateTimeOriginal": ts.strftime("%Y:%m:%d %H:%M:%S"),
            },
            "detections": [
                {
                    "category": "1",
                    "conf": 0.8 + (i * 0.01),
                    "bbox": [0.1, 0.2, 0.3, 0.4],
                    "classifications": [[1, 0.9]],
                },
            ],
        })

    md_json = build_detection_json(
        images, classification_categories={"1": "zebra"}
    )
    json_path = write_json(s["artifacts"] / "results.json", md_json)

    with patch("app.ml.json_pipeline.extract_video_dates", return_value={}):
        load_json_to_database(
            json_path=json_path,
            deployment_id=s["deployment"].id,
            deployment_folder=deploy_dir,
            job_id=s["job"].id,
            db=db,
            artifacts_folder=s["artifacts"],
        )


def test_full_pipeline_to_events(deployment_scaffold):
    """JSON → load → generate events: correct event count, times, file_count, MaxN observations."""
    s = deployment_scaffold
    db = s["db"]

    # 3 images close together → 1 event
    base = datetime(2024, 6, 15, 10, 0, 0)
    timestamps = [base, base + timedelta(minutes=5), base + timedelta(minutes=10)]
    _load_images_with_timestamps(s, timestamps)

    total = generate_events_for_project(db, s["project"].id)
    assert total == 1

    event = db.query(Event).one()
    assert event.file_count == 3
    assert event.event_start_local == base
    assert event.event_end_local == timestamps[-1]
    # MaxN observations should be computed
    assert len(event.observations) > 0


def test_events_with_mixed_content(deployment_scaffold):
    """Frame files included in events; video files excluded."""
    s = deployment_scaffold
    db, deploy_dir = s["db"], s["deploy_dir"]

    create_video_frames(s["artifacts"], "videos/clip.mp4", [0, 30])

    # Video + image at similar times
    base = datetime(2024, 6, 15, 10, 0, 0)
    images = [
        {
            "file": "videos/clip.mp4",
            "frame_rate": 30.0,
            "exif_metadata": {"DateTimeOriginal": base.strftime("%Y:%m:%d %H:%M:%S")},
            "detections": [
                {"category": "1", "conf": 0.9, "bbox": [0.1, 0.2, 0.3, 0.4],
                 "frame_number": 0, "classifications": [[1, 0.8]]},
            ],
        },
        {
            "file": "subdir/img_001.jpg",
            "exif_metadata": {
                "DateTimeOriginal": (base + timedelta(minutes=2)).strftime("%Y:%m:%d %H:%M:%S"),
            },
            "detections": [
                {"category": "1", "conf": 0.85, "bbox": [0.2, 0.3, 0.4, 0.5],
                 "classifications": [[1, 0.7]]},
            ],
        },
    ]
    md_json = build_detection_json(
        images, classification_categories={"1": "zebra"}
    )
    json_path = write_json(s["artifacts"] / "results.json", md_json)

    with patch("app.ml.json_pipeline.extract_video_dates", return_value={}):
        load_json_to_database(
            json_path=json_path,
            deployment_id=s["deployment"].id,
            deployment_folder=deploy_dir,
            job_id=s["job"].id,
            db=db,
            artifacts_folder=s["artifacts"],
        )

    total = generate_events_for_project(db, s["project"].id)
    assert total >= 1

    # Events should include frame files and image files, not video files
    event = db.query(Event).first()
    event_file_types = {f.file_type for f in event.files}
    assert "video" not in event_file_types
    assert "image" in event_file_types or "frame" in event_file_types


def test_events_idempotent_regeneration(deployment_scaffold):
    """Generate twice → same result (delete + rebuild)."""
    s = deployment_scaffold
    db = s["db"]

    base = datetime(2024, 6, 15, 10, 0, 0)
    timestamps = [base, base + timedelta(minutes=5)]
    _load_images_with_timestamps(s, timestamps)

    total1 = generate_events_for_project(db, s["project"].id)
    events1 = db.query(Event).all()
    assert total1 == 1

    # Regenerate
    total2 = generate_events_for_project(db, s["project"].id)
    events2 = db.query(Event).all()

    assert total2 == total1
    assert len(events2) == len(events1)
    # IDs differ (regenerated) but counts match
    assert events2[0].file_count == events1[0].file_count


def test_events_temporal_clustering(deployment_scaffold):
    """6 files in 2 clusters (3h apart) with independence_interval=1800 → 2 events."""
    s = deployment_scaffold
    db = s["db"]

    # Update project independence interval
    s["project"].independence_interval = 1800  # 30 min
    db.flush()

    base = datetime(2024, 6, 15, 8, 0, 0)
    # Cluster 1: 3 files close together
    # Cluster 2: 3 files 3 hours later
    timestamps = [
        base,
        base + timedelta(minutes=5),
        base + timedelta(minutes=10),
        base + timedelta(hours=3),
        base + timedelta(hours=3, minutes=5),
        base + timedelta(hours=3, minutes=10),
    ]
    _load_images_with_timestamps(s, timestamps)

    total = generate_events_for_project(db, s["project"].id)
    assert total == 2

    events = db.query(Event).order_by(Event.event_start_local.asc()).all()
    assert events[0].file_count == 3
    assert events[1].file_count == 3

    assert events[0].event_start_local == base
    assert events[0].event_end_local == base + timedelta(minutes=10)

    assert events[1].event_start_local == base + timedelta(hours=3)
    assert events[1].event_end_local == base + timedelta(hours=3, minutes=10)


def test_events_split_at_folder_boundary(deployment_scaffold):
    """
    Files from different folders must not cluster together, even when
    their timestamps fall inside the independence interval. Models the
    backlog case where one deployment wraps multiple SD-card folders.
    """
    s = deployment_scaffold
    db, deploy_dir = s["db"], s["deploy_dir"]
    s["project"].independence_interval = 1800  # 30 min
    db.flush()

    # Two folders, each with two images, timestamps 1 minute apart
    # everywhere. Under time-only clustering this is one event of 4.
    # With folder awareness, two events of two.
    base = datetime(2024, 6, 15, 10, 0, 0)
    (deploy_dir / "card_a").mkdir()
    (deploy_dir / "card_b").mkdir()
    images = []
    schedule = [
        ("card_a/img_000.jpg", base),
        ("card_a/img_001.jpg", base + timedelta(minutes=1)),
        ("card_b/img_000.jpg", base + timedelta(minutes=2)),
        ("card_b/img_001.jpg", base + timedelta(minutes=3)),
    ]
    for rel, ts in schedule:
        create_tiny_jpeg(deploy_dir / rel)
        images.append({
            "file": rel,
            "exif_metadata": {
                "DateTimeOriginal": ts.strftime("%Y:%m:%d %H:%M:%S"),
            },
            "detections": [
                {"category": "1", "conf": 0.9,
                 "bbox": [0.1, 0.2, 0.3, 0.4],
                 "classifications": [[1, 0.8]]},
            ],
        })
    md_json = build_detection_json(
        images, classification_categories={"1": "zebra"}
    )
    json_path = write_json(s["artifacts"] / "results.json", md_json)
    with patch("app.ml.json_pipeline.extract_video_dates", return_value={}):
        load_json_to_database(
            json_path=json_path,
            deployment_id=s["deployment"].id,
            deployment_folder=deploy_dir,
            job_id=s["job"].id,
            db=db,
            artifacts_folder=s["artifacts"],
        )

    total = generate_events_for_project(db, s["project"].id)
    assert total == 2
    events = db.query(Event).order_by(Event.event_start_local.asc()).all()
    assert events[0].file_count == 2
    assert events[1].file_count == 2
    # First event's files all live under card_a.
    for f in events[0].files:
        assert "/card_a/" in f.file_path
    for f in events[1].files:
        assert "/card_b/" in f.file_path


def test_events_interleaved_parallel_folders(deployment_scaffold):
    """
    Two cameras in two folders firing in parallel — timestamps interleave
    one second apart. The naive single-walk would break on every folder
    change and produce one event per file. Correct behaviour: bucket by
    folder first, then cluster, yielding one event per folder.
    """
    s = deployment_scaffold
    db, deploy_dir = s["db"], s["deploy_dir"]
    s["project"].independence_interval = 1800
    db.flush()

    base = datetime(2024, 6, 15, 10, 0, 0)
    (deploy_dir / "cam_a").mkdir()
    (deploy_dir / "cam_b").mkdir()
    schedule = [
        ("cam_a/img_000.jpg", base),
        ("cam_b/img_000.jpg", base + timedelta(seconds=1)),
        ("cam_a/img_001.jpg", base + timedelta(seconds=2)),
        ("cam_b/img_001.jpg", base + timedelta(seconds=3)),
        ("cam_a/img_002.jpg", base + timedelta(seconds=4)),
        ("cam_b/img_002.jpg", base + timedelta(seconds=5)),
    ]
    images = []
    for rel, ts in schedule:
        create_tiny_jpeg(deploy_dir / rel)
        images.append({
            "file": rel,
            "exif_metadata": {
                "DateTimeOriginal": ts.strftime("%Y:%m:%d %H:%M:%S"),
            },
            "detections": [
                {"category": "1", "conf": 0.9,
                 "bbox": [0.1, 0.2, 0.3, 0.4],
                 "classifications": [[1, 0.8]]},
            ],
        })
    md_json = build_detection_json(
        images, classification_categories={"1": "zebra"}
    )
    json_path = write_json(s["artifacts"] / "results.json", md_json)
    with patch("app.ml.json_pipeline.extract_video_dates", return_value={}):
        load_json_to_database(
            json_path=json_path,
            deployment_id=s["deployment"].id,
            deployment_folder=deploy_dir,
            job_id=s["job"].id,
            db=db,
            artifacts_folder=s["artifacts"],
        )

    total = generate_events_for_project(db, s["project"].id)
    assert total == 2
    events = db.query(Event).all()
    assert sorted(e.file_count for e in events) == [3, 3]
    for e in events:
        folders = {f.file_path.rsplit("/", 1)[0] for f in e.files}
        assert len(folders) == 1, "event spans multiple folders"


def test_events_same_folder_still_cluster(deployment_scaffold):
    """Sanity: files that live in the SAME folder still cluster by time
    as before. The folder constraint is additive, not replacing the gap
    rule."""
    s = deployment_scaffold
    db, deploy_dir = s["db"], s["deploy_dir"]
    s["project"].independence_interval = 1800
    db.flush()

    base = datetime(2024, 6, 15, 10, 0, 0)
    (deploy_dir / "card_a").mkdir()
    images = []
    for i in range(3):
        rel = f"card_a/img_{i:03d}.jpg"
        create_tiny_jpeg(deploy_dir / rel)
        images.append({
            "file": rel,
            "exif_metadata": {
                "DateTimeOriginal": (base + timedelta(minutes=i * 2)).strftime(
                    "%Y:%m:%d %H:%M:%S"
                ),
            },
            "detections": [
                {"category": "1", "conf": 0.9,
                 "bbox": [0.1, 0.2, 0.3, 0.4],
                 "classifications": [[1, 0.8]]},
            ],
        })
    md_json = build_detection_json(
        images, classification_categories={"1": "zebra"}
    )
    json_path = write_json(s["artifacts"] / "results.json", md_json)
    with patch("app.ml.json_pipeline.extract_video_dates", return_value={}):
        load_json_to_database(
            json_path=json_path,
            deployment_id=s["deployment"].id,
            deployment_folder=deploy_dir,
            job_id=s["job"].id,
            db=db,
            artifacts_folder=s["artifacts"],
        )

    total = generate_events_for_project(db, s["project"].id)
    assert total == 1
    event = db.query(Event).one()
    assert event.file_count == 3
