"""
Integration tests: event generation (clustering from files).

Tests generate_events_for_project() after loading JSON to DB.
No subprocess mocks needed — cv2 works on 1x1 JPEGs.
"""

from datetime import datetime, timedelta
from unittest.mock import patch

from app.api.crud.event import generate_events_for_project
from app.ml.json_pipeline import load_json_to_database
from app.models import Event, File

from .conftest import build_detection_json, create_tiny_jpeg, write_json


def _load_images_with_timestamps(s, timestamps):
    """Helper: create tiny JPEGs with EXIF timestamps and load to DB."""
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
    """JSON → load → generate events: correct event count, times, file_count, representative."""
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
    assert event.start_time == base
    assert event.end_time == timestamps[-1]
    assert event.representative_file_id is not None
    # Representative must be a valid file
    rep = db.query(File).filter(File.id == event.representative_file_id).one()
    assert rep.deployment_id == s["deployment"].id


def test_events_with_mixed_content(deployment_scaffold):
    """Frame files included in events; video files excluded."""
    s = deployment_scaffold
    db, deploy_dir = s["db"], s["deploy_dir"]

    from .conftest import create_video_frames

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

    events = db.query(Event).order_by(Event.start_time.asc()).all()
    assert events[0].file_count == 3
    assert events[1].file_count == 3

    assert events[0].start_time == base
    assert events[0].end_time == base + timedelta(minutes=10)

    assert events[1].start_time == base + timedelta(hours=3)
    assert events[1].end_time == base + timedelta(hours=3, minutes=10)
