"""`GET /api/files/{id}/image?frame=N` decodes any frame on request.

A video stores its cover frame and one crop per track, nothing per
frame. The endpoint serves the cover by default and decodes any other
frame with ffmpeg, saying so when the frame cannot be decoded rather
than handing out the cover under a wrong label.
"""

from PIL import Image

from app.api.routers import files as files_router
from tests.conftest import (
    make_deployment,
    make_file,
    make_project,
    make_site,
    make_track,
)


def _jpeg(path, colour, size=(32, 24)):
    Image.new("RGB", size, colour).save(path, "JPEG")
    return path


def _tracked_video(db, tmp_path):
    project = make_project(db)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)
    best = _jpeg(tmp_path / "frame000060.jpg", (10, 10, 10))
    video = make_file(
        db, deployment_id=dep.id, file_type="video", file_format="mp4",
        file_path="/fake/clip.mp4", frame_rate=30.0, best_frame_number=60,
        best_frame_path=str(best),
    )
    make_track(db, file_id=video.id, track_key=2, start_frame=300, end_frame=360,
               representative_frame_number=330, crop_path=str(tmp_path / "track000002.jpg"))
    make_track(db, file_id=video.id, track_key=3, start_frame=600, end_frame=630,
               representative_frame_number=600, crop_path=None)
    db.commit()
    return video, best


def test_the_default_is_the_cover_frame(client, db, tmp_path):
    video, best = _tracked_video(db, tmp_path)
    resp = client.get(f"/api/files/{video.id}/image")
    assert resp.status_code == 200
    assert resp.content == best.read_bytes()
    # Asking for the cover by number is the same stored picture.
    resp = client.get(f"/api/files/{video.id}/image?frame=60")
    assert resp.content == best.read_bytes()


def test_any_other_frame_is_decoded_on_request(client, db, tmp_path, monkeypatch):
    video, _ = _tracked_video(db, tmp_path)
    decoded = _jpeg(tmp_path / "decoded.jpg", (200, 200, 200), size=(1600, 900)).read_bytes()
    asked: list[tuple[str, int, float | None]] = []

    def fake_decode(path, frame, rate):
        asked.append((path, frame, rate))
        return decoded

    monkeypatch.setattr(files_router, "decode_frame_jpeg", fake_decode)

    resp = client.get(f"/api/files/{video.id}/image?frame=330")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/jpeg")
    assert resp.content == decoded
    # A frame no track names is decoded just the same.
    resp = client.get(f"/api/files/{video.id}/image?frame=90")
    assert resp.status_code == 200 and resp.content == decoded
    assert asked == [("/fake/clip.mp4", 330, 30.0), ("/fake/clip.mp4", 90, 30.0)]

    # The thumbnail size applies to a decoded frame too.
    resp = client.get(f"/api/files/{video.id}/image?frame=330&size=thumb")
    assert resp.status_code == 200
    assert Image.open(__import__("io").BytesIO(resp.content)).width <= 768


def test_a_frame_that_cannot_be_decoded_is_a_404_not_the_cover(
    client, db, tmp_path, monkeypatch
):
    video, _ = _tracked_video(db, tmp_path)
    monkeypatch.setattr(files_router, "decode_frame_jpeg", lambda *a: None)
    resp = client.get(f"/api/files/{video.id}/image?frame=600")
    assert resp.status_code == 404
    assert "Could not decode frame 600" in resp.json()["detail"]


def test_the_file_detail_lists_its_tracks(client, db, tmp_path):
    video, _ = _tracked_video(db, tmp_path)
    resp = client.get(f"/api/files/{video.id}")
    assert resp.status_code == 200
    tracks = sorted(resp.json()["tracks"], key=lambda t: t["track_key"])
    assert [(t["track_key"], t["representative_frame_number"]) for t in tracks] == [
        (2, 330),
        (3, 600),
    ]
    assert "has_frame" not in tracks[0]
