"""Tests for the /api/files endpoints."""

import io
import tempfile
import uuid
from pathlib import Path

from PIL import Image

from app.models import LabelTaxonomy
from tests.conftest import (
    make_deployment,
    make_detection,
    make_file,
    make_project,
    make_site,
    make_track,
    make_video_box,
)


def _setup_deployment(db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    return make_deployment(db, site_id=s.id), p


def test_list_files_empty(client):
    resp = client.get("/api/files")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_files_by_deployment(client, db):
    d, _ = _setup_deployment(db)
    make_file(db, deployment_id=d.id)
    resp = client.get(f"/api/files?deployment_id={d.id}")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_get_file(client, db):
    d, _ = _setup_deployment(db)
    f = make_file(db, deployment_id=d.id)
    resp = client.get(f"/api/files/{f.id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == f.id


def test_get_file_not_found(client):
    resp = client.get("/api/files/nonexistent")
    assert resp.status_code == 404


def test_update_file_verified(client, db):
    d, _ = _setup_deployment(db)
    f = make_file(db, deployment_id=d.id)
    resp = client.patch(f"/api/files/{f.id}", json={"verified": True})
    assert resp.status_code == 200
    assert resp.json()["verified"] is True


def test_update_file_flagged_sets_timestamp(client, db):
    d, _ = _setup_deployment(db)
    f = make_file(db, deployment_id=d.id)
    resp = client.patch(f"/api/files/{f.id}", json={"flagged": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["flagged"] is True
    assert body["flagged_at_utc"] is not None
    resp = client.patch(f"/api/files/{f.id}", json={"flagged": False})
    assert resp.status_code == 200
    body = resp.json()
    assert body["flagged"] is False
    assert body["flagged_at_utc"] is None


def test_get_file_image_success(client, db):
    d, _ = _setup_deployment(db)
    # Create a real temp JPEG file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        # Minimal JPEG header
        tmp.write(b"\xff\xd8\xff\xe0\x00\x10JFIF")
        tmp_path = tmp.name
    f = make_file(db, deployment_id=d.id, file_path=tmp_path, file_format="jpg")
    resp = client.get(f"/api/files/{f.id}/image")
    assert resp.status_code == 200
    Path(tmp_path).unlink(missing_ok=True)


def test_get_file_image_file_missing_on_disk(client, db):
    d, _ = _setup_deployment(db)
    f = make_file(db, deployment_id=d.id, file_path="/nonexistent/photo.jpg")
    resp = client.get(f"/api/files/{f.id}/image")
    assert resp.status_code == 404


def test_get_file_image_not_in_db(client):
    resp = client.get("/api/files/nonexistent/image")
    assert resp.status_code == 404


def test_a_video_that_never_decoded_says_so_instead_of_serving_the_container(
    client, db
):
    """A video with no best frame must not fall through to its own bytes.

    `best_frame.py` skips a clip it cannot open, so `best_frame_number`
    and `best_frame_path` both stay NULL, and such a clip always lands in
    the Empties tab (no visible surface means no passing detection). Its
    tile then asks this endpoint for a picture. Serving the container
    answered `image/avi` for the full size and, for `size=thumb`, handed
    the file to PIL, which cannot open an AVI: an unhandled
    `UnidentifiedImageError` and a 500 per tile.

    The file exists on disk here on purpose. The existing missing-file
    test would pass against the old code too, since that path never got
    as far as opening anything.
    """
    d, _ = _setup_deployment(db)
    with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as tmp:
        tmp.write(b"RIFF\x00\x00\x00\x00AVI ")
        tmp_path = tmp.name
    f = make_file(
        db,
        deployment_id=d.id,
        file_path=tmp_path,
        file_type="video",
        file_format="avi",
        best_frame_number=None,
        best_frame_path=None,
    )
    try:
        for url in (
            f"/api/files/{f.id}/image",
            f"/api/files/{f.id}/image?size=thumb",
        ):
            resp = client.get(url)
            assert resp.status_code == 404, url
            assert "could not be decoded" in resp.json()["detail"]
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_label_tree_count_by_file(client, db):
    p = make_project(db, counting_threshold=0.5)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    # Build a tiny taxonomy: one species "wolf" under mammalia/carnivora/canidae/canis
    cls_model_id = p.classification_model_id or str(uuid.uuid4())
    if not p.classification_model_id:
        # Ensure taxonomy lookup works even without a real model.
        p.classification_model_id = cls_model_id
    tax = LabelTaxonomy(
        id=str(uuid.uuid4()),
        classification_model_id=cls_model_id,
        name="wolf",
        taxon_class="mammalia",
        taxon_order="carnivora",
        taxon_family="canidae",
        taxon_genus="canis",
        taxon_species="lupus",
        level="species",
    )
    db.add(tax)
    db.flush()

    # Image 1 with two wolf detections
    img1 = make_file(db, deployment_id=d.id, file_type="image")
    make_detection(
        db, file_id=img1.id, confidence=0.9,
        label="wolf", label_taxonomy_id=tax.id,
    )
    make_detection(
        db, file_id=img1.id, confidence=0.8,
        label="wolf", label_taxonomy_id=tax.id,
    )
    # Image 2 with one wolf detection
    img2 = make_file(db, deployment_id=d.id, file_type="image")
    make_detection(
        db, file_id=img2.id, confidence=0.95,
        label="wolf", label_taxonomy_id=tax.id,
    )
    db.commit()

    resp = client.get(
        f"/api/events/label-tree?project_id={p.id}&count_by=file"
    )
    assert resp.status_code == 200
    body = resp.json()
    # Two distinct files; three detections.
    counts = body["label_event_counts"]
    assert counts.get("wolf") == 2
    assert body["count_unit"] == "file"


def test_a_video_whose_cover_is_gone_is_shown_from_its_decoded_frame(
    client, db, tmp_path, monkeypatch
):
    """The cover lives in the deployment's `.addaxai` folder. A bundle
    without photos, connected to a copy of the videos that was never
    analysed, has the rows and not that folder: the tile then showed no
    picture. The cover frame is decoded from the clip instead, at both
    sizes, and the stored cover is still used when it is there."""
    from app.api.routers.files import _THUMB_MAX_WIDTH
    from app.services import video_frame_service

    asked = []

    def fake_decode(path, frame_number, rate):
        asked.append(frame_number)
        buf = io.BytesIO()
        Image.new("RGB", (1024, 576), (20, 200, 20)).save(buf, "JPEG")
        return buf.getvalue()

    monkeypatch.setattr(video_frame_service, "decode_frame_jpeg", fake_decode)
    d, _ = _setup_deployment(db)
    f = make_file(
        db, deployment_id=d.id, file_path=str(tmp_path / "clip.mp4"),
        file_type="video", file_format="mp4", frame_rate=30.0,
        best_frame_number=60, best_frame_path=str(tmp_path / "gone" / "frame000060.jpg"),
    )

    full = client.get(f"/api/files/{f.id}/image")
    thumb = client.get(f"/api/files/{f.id}/image?size=thumb")

    assert full.status_code == thumb.status_code == 200
    assert Image.open(io.BytesIO(full.content)).size == (1024, 576)
    assert Image.open(io.BytesIO(thumb.content)).width == _THUMB_MAX_WIDTH
    assert asked == [60, 60]

    stored = tmp_path / "gone" / "frame000060.jpg"
    stored.parent.mkdir()
    Image.new("RGB", (64, 48), (200, 20, 20)).save(stored)
    resp = client.get(f"/api/files/{f.id}/image")
    assert Image.open(io.BytesIO(resp.content)).size == (64, 48)
    assert asked == [60, 60]


def test_label_tree_count_by_file_reads_a_video_on_its_best_frame(client, db):
    """A video counts for a label only when the box sits on the frame the
    app shows. The count query used to leave File unjoined, so the frame
    rule was checked against every file in the project (a cross join) and
    any image made the video's off-frame box count."""
    p = make_project(db, counting_threshold=0.5)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    cls_model_id = p.classification_model_id or str(uuid.uuid4())
    p.classification_model_id = cls_model_id
    tax = LabelTaxonomy(
        id=str(uuid.uuid4()),
        classification_model_id=cls_model_id,
        name="wolf",
        taxon_class="mammalia",
        level="species",
    )
    db.add(tax)
    db.flush()

    img = make_file(db, deployment_id=d.id, file_type="image")
    make_detection(
        db, file_id=img.id, confidence=0.9,
        label="wolf", label_taxonomy_id=tax.id,
    )
    video = make_file(
        db, deployment_id=d.id, file_type="video", file_format="mp4",
        best_frame_number=3,
    )
    make_detection(
        db, file_id=video.id, confidence=0.9, frame_number=1,
        label="wolf", label_taxonomy_id=tax.id,
    )
    db.commit()

    resp = client.get(
        f"/api/events/label-tree?project_id={p.id}&count_by=file"
    )
    assert resp.status_code == 200
    # The image only: the video's wolf is not on its best frame.
    assert resp.json()["label_event_counts"].get("wolf") == 1


def test_label_tree_rejects_invalid_count_by(client, db):
    p = make_project(db)
    resp = client.get(
        f"/api/events/label-tree?project_id={p.id}&count_by=bogus"
    )
    assert resp.status_code == 400


def _bulk_verify(client, ids, verified=True):
    return client.post(
        "/api/files/bulk-verify", json={"file_ids": ids, "verified": verified}
    )


def test_bulk_verify_signs_off_every_file_in_one_request(client, db):
    """The Files tab's bulk action. Same rule per file as the PATCH: the
    visible boxes are verified, the weak ones rejected, the file flagged."""
    from app.models import Detection

    d, _ = _setup_deployment(db)
    a = make_file(db, deployment_id=d.id)
    b = make_file(db, deployment_id=d.id)
    strong_id = make_detection(db, file_id=a.id, confidence=0.9).id
    weak_id = make_detection(db, file_id=a.id, confidence=0.05).id
    db.commit()

    resp = _bulk_verify(client, [a.id, b.id])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["updated_count"] == 2
    # The before-state for undo: both files, and the boxes the sign-off
    # touched (both of a's, b has none).
    assert {s["file_id"] for s in body["before"]["files"]} == {a.id, b.id}
    assert {s["detection_id"] for s in body["before"]["detections"]} == {
        strong_id,
        weak_id,
    }

    db.expire_all()
    assert a.verified is True
    assert b.verified is True
    assert db.get(Detection, strong_id).verified is True
    weak = db.get(Detection, weak_id)
    assert weak.label == "false detection"
    assert weak.verified is True


def test_bulk_unverify_takes_the_sign_off_back(client, db):

    d, _ = _setup_deployment(db)
    a = make_file(db, deployment_id=d.id)
    det = make_detection(db, file_id=a.id, confidence=0.9)
    db.commit()
    assert _bulk_verify(client, [a.id]).status_code == 200

    resp = _bulk_verify(client, [a.id], verified=False)
    assert resp.status_code == 200, resp.text

    db.expire_all()
    assert a.verified is False
    assert det.verified is False


def test_bulk_verify_of_nothing_is_a_no_op_and_of_unknown_ids_a_404(client, db):
    assert _bulk_verify(client, []).json() == {
        "updated_count": 0,
        "before": {"detections": [], "files": []},
    }
    assert _bulk_verify(client, [str(uuid.uuid4())]).status_code == 404


def _tracked_clip(db, *, counting_threshold=0.5):
    """A clip with its cover on frame 60 and two tracks: one of three
    frames whose card sits on frame 90, one of one frame on the cover."""
    p = make_project(db, counting_threshold=counting_threshold)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    clip = make_file(
        db, deployment_id=d.id, file_type="video", file_format="mp4",
        file_path="/fake/clip.mp4", frame_rate=30.0, best_frame_number=60,
    )
    long = make_track(db, file_id=clip.id, track_key=1, start_frame=30, end_frame=90,
                      representative_frame_number=90)
    box = dict(file_id=clip.id, category="fish", confidence=0.9)
    make_detection(db, frame_number=30, track_id=long.id, **box)
    make_detection(db, frame_number=60, track_id=long.id, **box)
    card = make_detection(db, frame_number=90, track_id=long.id, **box)
    on_cover = make_video_box(db, file_id=clip.id, frame_number=60, category="fish", confidence=0.9)
    db.commit()
    return clip, card, on_cover


def test_cards_scope_carries_the_cover_boxes_and_one_box_per_track(client, db):
    """A tile draws the cover and names the species of the cards, and a
    viewer needs every frame: two scopes of one endpoint. On a BRUV clip
    "all" is 59,000 boxes and 28 MB, "cards" about 2,300 and 1 MB."""
    clip, card, on_cover = _tracked_clip(db)

    cards = client.get(f"/api/files/{clip.id}?scope=cards").json()
    everything = client.get(f"/api/files/{clip.id}").json()

    # Frame 60 twice (the cover: the long track's middle box and the
    # one-frame track's card) and frame 90 (the long track's card);
    # frame 30 stays in the database.
    assert sorted((d["frame_number"], d["id"] == card.id) for d in cards["detections"]) == [
        (60, False), (60, False), (90, True)
    ]
    assert {d["id"] for d in cards["detections"]} >= {card.id, on_cover.id}
    assert len(everything["detections"]) == 4
    assert len(cards["tracks"]) == len(everything["tracks"]) == 2
    assert client.get(f"/api/files/{clip.id}?scope=frames").status_code == 422


def test_cards_scope_of_a_photo_is_every_box(client, db):
    d, _ = _setup_deployment(db)
    photo = make_file(db, deployment_id=d.id)
    make_detection(db, file_id=photo.id)
    make_detection(db, file_id=photo.id)
    db.commit()
    assert len(client.get(f"/api/files/{photo.id}?scope=cards").json()["detections"]) == 2


def test_max_at_once_is_the_peak_per_species_at_the_first_frame_it_reached(client, db):
    """Five fish on frame 60 and five on frame 90 is 5, at 60; a box below
    the counting threshold does not count unless verified; a rejected
    box never counts; species count apart. The same rule as the event
    MaxN, per file, so a tile chip and a count row say one number."""
    clip, _, _ = _tracked_clip(db)  # frame 60: 2 fish, frame 90: 1 fish so far
    fish = dict(file_id=clip.id, category="fish", confidence=0.9)
    # Frame 60 reaches 5 with three more; frame 90 reaches 5 with three
    # more plus the verified weak box, the unverified weak box and the
    # rejected box not counting.
    for frame, n in ((60, 3), (90, 3)):
        for _ in range(n):
            make_detection(db, frame_number=frame, **fish)
    make_detection(db, frame_number=90, file_id=clip.id, category="fish", confidence=0.2)
    make_detection(db, frame_number=90, file_id=clip.id, category="fish", confidence=0.2,
                   verified=True)
    make_detection(db, frame_number=90, file_id=clip.id, category="fish", confidence=0.9,
                   label="false detection", verified=True)
    make_detection(db, frame_number=30, file_id=clip.id, category="fish", confidence=0.9,
                   label="Shark")
    db.commit()

    by_species = {
        (m["label_taxonomy_id"], m["label"], m["category"]): m
        for m in client.get(f"/api/files/{clip.id}?scope=cards").json()["max_at_once"]
    }
    assert by_species[(None, None, "fish")] == {
        "label_taxonomy_id": None, "label": None, "category": "fish",
        "count": 5, "frame_number": 60,
    }
    assert by_species[(None, "Shark", "fish")]["count"] == 1


def test_max_at_once_of_a_photo_has_no_frame(client, db):
    d, _ = _setup_deployment(db)
    photo = make_file(db, deployment_id=d.id)
    make_detection(db, file_id=photo.id)
    make_detection(db, file_id=photo.id)
    db.commit()
    assert client.get(f"/api/files/{photo.id}").json()["max_at_once"] == [
        {"label_taxonomy_id": None, "label": None, "category": "animal",
         "count": 2, "frame_number": None}
    ]


def test_thumbnail_of_a_rotated_photo_comes_out_upright(client, db, tmp_path):
    """The thumbnail is re-encoded without EXIF, so the browser cannot
    rotate it the way it rotates the original; the pixels are rotated
    here instead, matching the boxes drawn over them."""
    d, _ = _setup_deployment(db)
    exif = Image.Exif()
    exif[0x0112] = 6
    path = tmp_path / "sideways.jpg"
    Image.new("RGB", (120, 200), (90, 90, 90)).save(path, "JPEG", exif=exif.tobytes())
    f = make_file(db, deployment_id=d.id, file_path=str(path), file_format="jpg")

    resp = client.get(f"/api/files/{f.id}/image?size=thumb")

    assert resp.status_code == 200
    assert Image.open(io.BytesIO(resp.content)).size == (200, 120)
