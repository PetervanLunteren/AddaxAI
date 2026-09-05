"""`GET /api/files/{id}/image?frame=N` serves a track's still.

A video has one still per track beside its best frame. The endpoint
serves the best frame by default, the track's still for its
representative frame, and says so for any other frame rather than
handing out the best frame under a wrong label.
"""

from PIL import Image

from tests.conftest import (
    make_deployment,
    make_file,
    make_project,
    make_site,
    make_track,
)


def _jpeg(path, colour):
    Image.new("RGB", (32, 24), colour).save(path, "JPEG")
    return path


def _tracked_video(db, tmp_path):
    project = make_project(db)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)
    best = _jpeg(tmp_path / "frame000060.jpg", (10, 10, 10))
    still = _jpeg(tmp_path / "frame000330.jpg", (200, 200, 200))
    video = make_file(
        db, deployment_id=dep.id, file_type="video", file_format="mp4",
        file_path="/fake/clip.mp4", best_frame_number=60, best_frame_path=str(best),
    )
    make_track(db, file_id=video.id, track_key=2, start_frame=300, end_frame=360,
               representative_frame_number=330, frame_path=str(still))
    make_track(db, file_id=video.id, track_key=3, start_frame=600, end_frame=630,
               representative_frame_number=600, frame_path=None)
    db.commit()
    return video, best, still


def test_the_default_is_the_best_frame(client, db, tmp_path):
    video, best, _ = _tracked_video(db, tmp_path)
    resp = client.get(f"/api/files/{video.id}/image")
    assert resp.status_code == 200
    assert resp.content == best.read_bytes()
    # Asking for the best frame by number is the same picture.
    resp = client.get(f"/api/files/{video.id}/image?frame=60")
    assert resp.content == best.read_bytes()


def test_a_tracks_representative_frame_has_its_own_still(client, db, tmp_path):
    video, _, still = _tracked_video(db, tmp_path)
    resp = client.get(f"/api/files/{video.id}/image?frame=330")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/jpeg")
    assert resp.content == still.read_bytes()


def test_a_frame_without_a_still_is_a_404_not_the_best_frame(client, db, tmp_path):
    video, _, _ = _tracked_video(db, tmp_path)
    # A frame nobody wrote, and a track whose still could not be decoded.
    for frame in (90, 600):
        resp = client.get(f"/api/files/{video.id}/image?frame={frame}")
        assert resp.status_code == 404, frame
        assert "no still" in resp.json()["detail"]


def test_the_file_detail_lists_its_tracks(client, db, tmp_path):
    video, _, _ = _tracked_video(db, tmp_path)
    resp = client.get(f"/api/files/{video.id}")
    assert resp.status_code == 200
    tracks = sorted(resp.json()["tracks"], key=lambda t: t["track_key"])
    assert [(t["track_key"], t["representative_frame_number"], t["has_frame"]) for t in tracks] == [
        (2, 330, True),
        (3, 600, False),
    ]
