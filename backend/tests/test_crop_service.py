"""A card's crop comes from the picture the box is actually in.

Every image box is a card and is cut from the photo. A video box is a
card only on its track's representative frame, and its picture is the
crop the tracking script stored, the cover frame when the card has no
crop and sits on it, or that frame decoded on request. A box that is
not a card gets no crop: cutting the cover at a box from another moment
produced a confident picture of the wrong place (31 of 32 tiles of a
walking person were leaf litter), and it stayed hidden because a slow
subject makes it look right.
"""

import io

from PIL import Image

from app.services import crop_service
from app.services.crop_service import _resolve_source, get_or_create_crop
from tests.conftest import (
    make_deployment,
    make_detection,
    make_file,
    make_project,
    make_track,
    make_video_box,
)


def _jpeg(tmp_path, name="frame000024.jpg", size=(640, 480), colour=(120, 120, 120)):
    path = tmp_path / name
    Image.new("RGB", size, colour).save(path, "JPEG")
    return path


def _video(db, tmp_path, best=24):
    cover = _jpeg(tmp_path)
    dep = make_deployment(db, project_id=make_project(db).id)
    f = make_file(
        db,
        deployment_id=dep.id,
        file_type="video",
        file_format="mp4",
        file_path="/fake/clip.mp4",
        frame_rate=30.0,
        best_frame_number=best,
        best_frame_path=str(cover),
    )
    return f, cover


def _box_of(source):
    return source[1]


def test_image_detection_resolves_to_the_file_itself(db, tmp_path):
    photo = _jpeg(tmp_path, "photo.jpg")
    dep = make_deployment(db, project_id=make_project(db).id)
    f = make_file(db, deployment_id=dep.id, file_path=str(photo))
    d = make_detection(db, file_id=f.id, bbox_x=0.1, bbox_y=0.2, bbox_width=0.3, bbox_height=0.4)

    assert _resolve_source(f, d) == (photo, [0.1, 0.2, 0.3, 0.4])


def test_a_card_on_the_cover_is_cut_from_the_cover(db, tmp_path):
    """A one-frame track (a drawn box, a legacy box) sitting on the cover
    frame has no crop of its own; the cover holds its pixels."""
    f, cover = _video(db, tmp_path)
    d = make_video_box(db, file_id=f.id, frame_number=24)

    picture, bbox = _resolve_source(f, d)
    assert picture == cover and bbox == [0.1, 0.1, 0.2, 0.2]
    assert get_or_create_crop(d.id, 200, db) is not None


def test_a_box_that_is_not_a_card_has_no_crop(db, tmp_path):
    """An untracked box, on the cover or off it, and a track's box on
    any frame but its representative one: no picture, no crop."""
    f, _ = _video(db, tmp_path)
    untracked_on_cover = make_detection(db, file_id=f.id, frame_number=24)
    untracked_off = make_detection(db, file_id=f.id, frame_number=144)
    track = make_track(db, file_id=f.id, start_frame=270, end_frame=330,
                       representative_frame_number=300)
    sibling = make_detection(db, file_id=f.id, frame_number=330, track_id=track.id)
    no_frame = make_detection(db, file_id=f.id, frame_number=None)
    db.flush()

    for d in (untracked_on_cover, untracked_off, sibling, no_frame):
        assert _resolve_source(f, d) is None
        assert get_or_create_crop(d.id, 200, db) is None


def test_a_tracks_card_is_the_stored_crop_as_it_is(db, tmp_path):
    """The tracking script cut the padded square already; the service
    serves it resized, pixel for pixel, with no second geometry pass."""
    f, _ = _video(db, tmp_path)
    crop = _jpeg(tmp_path, "track000001.jpg", size=(300, 300), colour=(200, 30, 30))
    track = make_track(db, file_id=f.id, start_frame=270, end_frame=330,
                       representative_frame_number=300, crop_path=str(crop))
    card = make_detection(db, file_id=f.id, frame_number=300, track_id=track.id)
    db.commit()

    assert _resolve_source(f, card) == (crop, [0.0, 0.0, 1.0, 1.0])
    served = Image.open(io.BytesIO(get_or_create_crop(card.id, 100, db)))
    assert served.size == (100, 100)
    r, g, b = served.getpixel((50, 50))
    assert r > 180 and g < 60 and b < 60


def test_a_card_without_a_crop_off_the_cover_is_cut_from_a_decoded_frame(
    db, tmp_path, monkeypatch
):
    """A legacy verified box or a drawn box on another frame: the frame
    is decoded on request and the card cut from it."""
    f, _ = _video(db, tmp_path)
    frame = _jpeg(tmp_path, "decoded.jpg", size=(640, 480), colour=(20, 200, 20)).read_bytes()
    asked = []

    def fake_decode(path, frame_number, rate):
        asked.append((path, frame_number, rate))
        return frame

    monkeypatch.setattr(crop_service, "decode_frame_jpeg", fake_decode)
    d = make_video_box(db, file_id=f.id, frame_number=144, verified=True)
    db.commit()

    picture, bbox = _resolve_source(f, d)
    assert picture == frame and bbox == [0.1, 0.1, 0.2, 0.2]
    served = Image.open(io.BytesIO(get_or_create_crop(d.id, 100, db)))
    assert served.getpixel((50, 50))[1] > 150
    assert set(asked) == {("/fake/clip.mp4", 144, 30.0)}

    # When the frame cannot be decoded there is no crop, not the cover.
    monkeypatch.setattr(crop_service, "decode_frame_jpeg", lambda *a: None)
    crop_service.invalidate_crop_cache(d.id)
    assert _resolve_source(f, d) is None
    assert get_or_create_crop(d.id, 100, db) is None
