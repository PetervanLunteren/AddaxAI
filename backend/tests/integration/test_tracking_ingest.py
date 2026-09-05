"""A tracked video's JSON becomes track rows and boxes that point at them.

The tracking script writes MegaDetector JSON with a `track_id` on every
box (and only tracked boxes, see the script). The ingest turns each id
into one `tracks` row per video and links the boxes; a run without
tracking leaves every `track_id` NULL and makes no rows.
"""

from app.ml.json_pipeline import load_json_to_database
from app.models import Detection, File, Track

from .conftest import build_detection_json, create_video_frames, write_json


def _box(frame, conf, track=None, bbox=(0.1, 0.1, 0.2, 0.2)):
    det = {"category": "0", "conf": conf, "bbox": list(bbox), "frame_number": frame}
    if track is not None:
        det["track_id"] = track
    return det


def _tracked_json(detections):
    return build_detection_json(
        images=[{
            "file": "videos/clip.mp4",
            "frame_rate": 30.0,
            "frames_processed": [0, 30, 60, 90, 300, 330, 600],
            "best_frame_number": 60,
            "detections": detections,
        }],
        detection_categories={"0": "elasmobranch"},
    )


def _load(s, data):
    json_path = write_json(s["artifacts"] / "results.json", data)
    return load_json_to_database(
        json_path,
        s["deployment"].id,
        s["deploy_dir"],
        s["job"].id,
        s["db"],
        artifacts_folder=s["artifacts"],
    )


def test_tracked_boxes_become_track_rows(deployment_scaffold):
    s = deployment_scaffold
    db = s["db"]
    # The frame passes wrote the best frame and the representative frames
    # of tracks 1 and 2; track 3's frame never decoded.
    create_video_frames(s["artifacts"], "videos/clip.mp4", [60, 300])

    _load(s, _tracked_json([
        _box(30, 0.5, 1),
        _box(60, 0.9, 1),
        _box(90, 0.7, 1),
        _box(300, 0.4, 2),
        _box(330, 0.4, 2),
        _box(600, 0.8, 3),
    ]))

    video = db.query(File).filter(File.file_type == "video").one()
    tracks = {t.track_key: t for t in db.query(Track).filter(Track.file_id == video.id)}
    assert sorted(tracks) == [1, 2, 3]

    one = tracks[1]
    assert (one.start_frame, one.end_frame, one.frame_count) == (30, 90, 3)
    assert one.max_confidence == 0.9
    assert one.representative_frame_number == 60
    assert one.frame_path is not None and one.frame_path.endswith("frame000060.jpg")
    # Same folder and name scheme as the best frame, so they share a file.
    assert one.frame_path == video.best_frame_path

    assert tracks[2].representative_frame_number == 300
    assert tracks[2].frame_path is not None and tracks[2].frame_path.endswith("frame000300.jpg")
    # No JPEG on disk: NULL, never a path to a file that is not there.
    assert tracks[3].frame_path is None

    boxes = db.query(Detection).filter(Detection.file_id == video.id).all()
    assert len(boxes) == 6
    by_frame = {d.frame_number: d for d in boxes}
    assert by_frame[30].track_id == one.id
    assert by_frame[60].track_id == one.id
    assert by_frame[300].track_id == tracks[2].id
    assert by_frame[600].track_id == tracks[3].id
    # Categories pass through untranslated.
    assert {d.category for d in boxes} == {"elasmobranch"}


def test_a_run_without_tracking_makes_no_track_rows(deployment_scaffold):
    s = deployment_scaffold
    db = s["db"]
    create_video_frames(s["artifacts"], "videos/clip.mp4", [60])

    _load(s, _tracked_json([_box(30, 0.5), _box(60, 0.9)]))

    assert db.query(Track).count() == 0
    assert all(d.track_id is None for d in db.query(Detection).all())


def test_a_second_ingest_finds_its_track_rows_again(deployment_scaffold):
    """Keyed by (file, track key): a re-ingest onto an existing file row
    refreshes the rows instead of tripping the unique constraint."""
    s = deployment_scaffold
    db = s["db"]
    create_video_frames(s["artifacts"], "videos/clip.mp4", [60])
    data = _tracked_json([_box(30, 0.5, 1), _box(60, 0.9, 1)])

    _load(s, data)
    _load(s, data)

    video = db.query(File).filter(File.file_type == "video").one()
    tracks = db.query(Track).filter(Track.file_id == video.id).all()
    assert len(tracks) == 1
    assert all(
        d.track_id == tracks[0].id
        for d in db.query(Detection).filter(Detection.file_id == video.id)
    )
