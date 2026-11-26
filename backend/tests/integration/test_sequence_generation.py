"""
Integration tests: sequence generation (clustering from files).

Tests generate_sequences_for_project() after loading JSON to DB.
No subprocess mocks needed — cv2 works on 1x1 JPEGs.
"""

from datetime import datetime, timedelta
from unittest.mock import patch

from app.api.crud.sequence import generate_sequences_for_project
from app.ml.json_pipeline import load_json_to_database
from app.models import Sequence

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


def test_full_pipeline_to_sequences(deployment_scaffold):
    """JSON → load → generate sequences: correct sequence count, times, file_count,
    MaxN observations."""
    s = deployment_scaffold
    db = s["db"]

    # 3 images close together → 1 sequence
    base = datetime(2024, 6, 15, 10, 0, 0)
    timestamps = [base, base + timedelta(minutes=1), base + timedelta(minutes=2)]
    _load_images_with_timestamps(s, timestamps)

    total = generate_sequences_for_project(db, s["project"].id)
    assert total == 1

    sequence = db.query(Sequence).one()
    assert sequence.file_count == 3
    assert sequence.sequence_start_local == base
    assert sequence.sequence_end_local == timestamps[-1]
    # MaxN observations should be computed
    assert len(sequence.observations) > 0


def test_sequences_with_mixed_content(deployment_scaffold):
    """Frame files included in sequences; video files excluded."""
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

    total = generate_sequences_for_project(db, s["project"].id)
    assert total >= 1

    # Sequences should include frame files and image files, not video files
    sequence = db.query(Sequence).first()
    sequence_file_types = {f.file_type for f in sequence.files}
    assert "video" not in sequence_file_types
    assert "image" in sequence_file_types or "frame" in sequence_file_types


def test_sequences_idempotent_regeneration(deployment_scaffold):
    """Generate twice → same result (delete + rebuild)."""
    s = deployment_scaffold
    db = s["db"]

    base = datetime(2024, 6, 15, 10, 0, 0)
    timestamps = [base, base + timedelta(minutes=1)]
    _load_images_with_timestamps(s, timestamps)

    total1 = generate_sequences_for_project(db, s["project"].id)
    events1 = db.query(Sequence).all()
    assert total1 == 1

    # Regenerate
    total2 = generate_sequences_for_project(db, s["project"].id)
    events2 = db.query(Sequence).all()

    assert total2 == total1
    assert len(events2) == len(events1)
    # IDs differ (regenerated) but counts match
    assert events2[0].file_count == events1[0].file_count


def test_every_file_is_in_one_sequence_at_most(deployment_scaffold):
    """The Files tab's sequence sort joins files to their sequence, so a file in
    two sequences would be listed twice (`get_labels_files`). Generation
    assigns each file once, also when it runs again over the same files."""
    from sqlalchemy import func, select

    from app.models.sequence import sequence_files

    s = deployment_scaffold
    db = s["db"]
    base = datetime(2024, 6, 15, 10, 0, 0)
    # Two bursts more than the 30-minute interval apart: two sequences.
    _load_images_with_timestamps(
        s, [base, base + timedelta(minutes=1), base + timedelta(hours=3)]
    )

    for _ in range(2):
        assert generate_sequences_for_project(db, s["project"].id) == 2
        per_file = db.execute(
            select(sequence_files.c.file_id, func.count())
            .group_by(sequence_files.c.file_id)
        ).all()
        assert len(per_file) == 3
        assert {n for _, n in per_file} == {1}


def test_sequences_temporal_clustering(deployment_scaffold):
    """6 files in 2 bursts (3h apart) → 2 sequences. The project's
    independence interval plays no part."""
    s = deployment_scaffold
    db = s["db"]

    base = datetime(2024, 6, 15, 8, 0, 0)
    # Cluster 1: 3 files close together
    # Cluster 2: 3 files 3 hours later
    timestamps = [
        base,
        base + timedelta(minutes=1),
        base + timedelta(minutes=2),
        base + timedelta(hours=3),
        base + timedelta(hours=3, minutes=1),
        base + timedelta(hours=3, minutes=2),
    ]
    _load_images_with_timestamps(s, timestamps)

    total = generate_sequences_for_project(db, s["project"].id)
    assert total == 2

    sequences = db.query(Sequence).order_by(Sequence.sequence_start_local.asc()).all()
    assert sequences[0].file_count == 3
    assert sequences[1].file_count == 3

    assert sequences[0].sequence_start_local == base
    assert sequences[0].sequence_end_local == base + timedelta(minutes=2)

    assert sequences[1].sequence_start_local == base + timedelta(hours=3)
    assert sequences[1].sequence_end_local == base + timedelta(hours=3, minutes=2)


def test_sequences_split_at_folder_boundary(deployment_scaffold):
    """
    Files from different folders must not cluster together, even when
    their timestamps fall inside the independence interval. Models the
    backlog case where one deployment wraps multiple SD-card folders.
    """
    s = deployment_scaffold
    db, deploy_dir = s["db"], s["deploy_dir"]
    db.flush()

    # Two folders, each with two images, timestamps 1 minute apart
    # everywhere. Under time-only clustering this is one sequence of 4.
    # With folder awareness, two sequences of two.
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

    total = generate_sequences_for_project(db, s["project"].id)
    assert total == 2
    sequences = db.query(Sequence).order_by(Sequence.sequence_start_local.asc()).all()
    assert sequences[0].file_count == 2
    assert sequences[1].file_count == 2
    # First sequence's files all live under card_a.
    for f in sequences[0].files:
        assert "/card_a/" in f.file_path
    for f in sequences[1].files:
        assert "/card_b/" in f.file_path


def test_sequences_interleaved_parallel_folders(deployment_scaffold):
    """
    Two cameras in two folders firing in parallel — timestamps interleave
    one second apart. The naive single-walk would break on every folder
    change and produce one sequence per file. Correct behaviour: bucket by
    folder first, then cluster, yielding one sequence per folder.
    """
    s = deployment_scaffold
    db, deploy_dir = s["db"], s["deploy_dir"]
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

    total = generate_sequences_for_project(db, s["project"].id)
    assert total == 2
    sequences = db.query(Sequence).all()
    assert sorted(e.file_count for e in sequences) == [3, 3]
    for e in sequences:
        folders = {f.file_path.rsplit("/", 1)[0] for f in e.files}
        assert len(folders) == 1, "sequence spans multiple folders"


def test_sequences_and_smoother_input_agree_on_boundaries(deployment_scaffold):
    """
    The UI's Sequence rows and the smoother's input must group files into
    the same clusters. Both call `cluster_files_into_sequences` — if that
    invariant ever broke, downstream stats would disagree with what the
    smoother operates on.
    """
    from app.ml.postprocessing import build_smoother_input

    s = deployment_scaffold
    db, deploy_dir = s["db"], s["deploy_dir"]
    db.flush()

    (deploy_dir / "card_a").mkdir()
    (deploy_dir / "card_b").mkdir()
    base = datetime(2024, 6, 15, 10, 0, 0)
    schedule = [
        ("card_a/img_000.jpg", base),
        ("card_a/img_001.jpg", base + timedelta(minutes=1)),
        ("card_b/img_000.jpg", base + timedelta(minutes=2)),
        ("card_b/img_001.jpg", base + timedelta(hours=3)),  # gap → new sequence
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

    # Generate sequences the way the UI would.
    generate_sequences_for_project(db, s["project"].id)
    sequences = db.query(Sequence).order_by(Sequence.sequence_start_local.asc()).all()
    sequence_file_groups = [
        frozenset(f.file_path for f in ev.files) for ev in sequences
    ]

    # Ask the smoother for its input on the same deployment.
    smoother_rows = build_smoother_input(s["deployment"].id, db)
    smoother_groups: dict[str, set[str]] = {}
    for row in smoother_rows:
        smoother_groups.setdefault(row["seq_id"], set()).add(
            str(deploy_dir / row["file_name"])
        )
    smoother_file_groups = [frozenset(v) for v in smoother_groups.values()]

    assert sorted(sequence_file_groups, key=sorted) == sorted(
        smoother_file_groups, key=sorted
    )


def test_sequences_same_folder_still_cluster(deployment_scaffold):
    """Sanity: files that live in the SAME folder still cluster by time
    as before. The folder constraint is additive, not replacing the gap
    rule."""
    s = deployment_scaffold
    db, deploy_dir = s["db"], s["deploy_dir"]
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

    total = generate_sequences_for_project(db, s["project"].id)
    assert total == 1
    sequence = db.query(Sequence).one()
    assert sequence.file_count == 3


def _load_paired_station(s: dict) -> None:
    """Two cameras at one station, one subfolder each, firing one second
    apart at the same animals. cam_a sees 2 zebras on its second frame,
    cam_b sees 1 on both of its frames."""
    db, deploy_dir = s["db"], s["deploy_dir"]
    base = datetime(2024, 6, 15, 10, 0, 0)
    (deploy_dir / "cam_a").mkdir()
    (deploy_dir / "cam_b").mkdir()
    box = {"category": "1", "conf": 0.9, "bbox": [0.1, 0.2, 0.3, 0.4],
           "classifications": [[1, 0.8]]}
    box2 = {**box, "bbox": [0.5, 0.5, 0.2, 0.2]}
    schedule = [
        ("cam_a/img_000.jpg", base, [box]),
        ("cam_b/img_000.jpg", base + timedelta(seconds=1), [box]),
        ("cam_a/img_001.jpg", base + timedelta(seconds=2), [box, box2]),
        ("cam_b/img_001.jpg", base + timedelta(seconds=3), [box]),
    ]
    images = []
    for rel, ts, dets in schedule:
        create_tiny_jpeg(deploy_dir / rel)
        images.append({
            "file": rel,
            "exif_metadata": {"DateTimeOriginal": ts.strftime("%Y:%m:%d %H:%M:%S")},
            "detections": dets,
        })
    md_json = build_detection_json(images, classification_categories={"1": "zebra"})
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


def test_paired_cameras_form_one_sequence_with_max_not_sum(deployment_scaffold):
    """A paired station groups both cameras' frames into one sequence in
    capture order, and MaxN is the largest count in any single frame (2),
    not the sum over both cameras (5)."""
    from app.ml.postprocessing import build_smoother_input

    s = deployment_scaffold
    db = s["db"]
    s["deployment"].paired_cameras = True
    db.flush()
    _load_paired_station(s)

    assert generate_sequences_for_project(db, s["project"].id) == 1
    sequence = db.query(Sequence).one()
    assert sequence.file_count == 4
    ordered = sorted(sequence.files, key=lambda f: f.captured_at_local)
    assert [f.file_path.rsplit("/", 2)[1] for f in ordered] == [
        "cam_a", "cam_b", "cam_a", "cam_b",
    ]
    (obs,) = sequence.observations
    assert obs.label == "zebra"
    assert obs.max_n == 2

    # The smoother sees the same grouping: one seq_id across both cameras.
    smoother = build_smoother_input(s["deployment"].id, db)
    assert len(smoother) == 4
    assert len({row["seq_id"] for row in smoother}) == 1


def test_unpaired_station_layout_stays_two_sequences(deployment_scaffold):
    """Same layout with the flag off keeps the per-subfolder rule."""
    s = deployment_scaffold
    db = s["db"]
    db.flush()
    _load_paired_station(s)

    assert generate_sequences_for_project(db, s["project"].id) == 2
