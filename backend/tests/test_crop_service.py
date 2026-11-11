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

import pytest
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


@pytest.fixture(autouse=True)
def warmed(monkeypatch):
    """Rung 3 decodes the chunk of the track around the frame before
    the frame itself. Recorded here, never run: the clips are fake."""
    calls: list[tuple] = []
    monkeypatch.setattr(
        crop_service, "warm_track_frames", lambda *args: calls.append(args)
    )
    return calls


def _jpeg(tmp_path, name="frame000024.jpg", size=(640, 480), colour=(120, 120, 120)):
    path = tmp_path / name
    Image.new("RGB", size, colour).save(path, "JPEG")
    return path


def _jpeg_bytes(size=(640, 480), colour=(120, 120, 120)):
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, "JPEG")
    return buf.getvalue()


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


def test_a_box_with_no_track_or_no_frame_has_no_crop(db, tmp_path):
    """Untracked boxes, on the cover or off it, and a box with no frame
    number: visible nowhere, so no picture and no crop. A thumbnail
    would be the only surface they ever reached."""
    f, _ = _video(db, tmp_path)
    untracked_on_cover = make_detection(db, file_id=f.id, frame_number=24)
    untracked_off = make_detection(db, file_id=f.id, frame_number=144)
    no_frame = make_detection(db, file_id=f.id, frame_number=None)
    db.flush()

    for d in (untracked_on_cover, untracked_off, no_frame):
        assert _resolve_source(f, d) is None
        assert get_or_create_crop(d.id, 200, db) is None


def test_a_sibling_frame_is_cut_from_its_own_frame_at_its_own_box(db, tmp_path, monkeypatch):
    """The opened track's cards.

    A box on any frame of a track other than the card decodes that
    frame and is cut at its own bbox. It must not borrow the track's
    stored crop: that would render one picture for every frame of the
    animal, which looks entirely plausible and is wrong.
    """
    f, _ = _video(db, tmp_path)
    crop = _jpeg(tmp_path, "track000001.jpg", size=(300, 300), colour=(200, 30, 30))
    track = make_track(db, file_id=f.id, start_frame=270, end_frame=330,
                       representative_frame_number=300, crop_path=str(crop))
    sibling = make_detection(
        db, file_id=f.id, frame_number=330, track_id=track.id,
        bbox_x=0.5, bbox_y=0.25, bbox_width=0.2, bbox_height=0.3,
    )
    db.commit()

    frame_bytes = _jpeg_bytes(size=(640, 480), colour=(10, 220, 40))
    asked: list[tuple] = []

    def fake_decode(path, frame_number, frame_rate):
        asked.append((path, frame_number, frame_rate))
        return frame_bytes

    monkeypatch.setattr(crop_service, "decode_frame_jpeg", fake_decode)

    picture, bbox = _resolve_source(f, sibling)

    assert picture == frame_bytes
    assert bbox == [0.5, 0.25, 0.2, 0.3]
    assert asked == [(f.file_path, 330, f.frame_rate)]

    out = get_or_create_crop(sibling.id, 200, db)
    assert out is not None
    img = Image.open(io.BytesIO(out))
    assert img.size == (200, 200)
    # The decoded frame's green, not the stored crop's red. Loose bounds:
    # the pixels went through JPEG twice.
    r, g, b = img.convert("RGB").getpixel((100, 100))
    assert g >= 200 and r <= 40 and b <= 70


def test_a_tracks_card_is_the_stored_crop_as_it_is(db, tmp_path):
    """The tracking script cut the padded square already; the service
    serves it resized, pixel for pixel, with no second geometry pass."""
    f, _ = _video(db, tmp_path)
    crop = _jpeg(tmp_path, "track000001.jpg", size=(300, 300), colour=(200, 30, 30))
    track = make_track(db, file_id=f.id, start_frame=270, end_frame=330,
                       representative_frame_number=300, crop_path=str(crop))
    card = make_detection(db, file_id=f.id, frame_number=300, track_id=track.id)
    db.commit()

    assert _resolve_source(f, card) == (crop, None)
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


def test_a_sibling_frame_warms_its_tracks_chunk_from_that_frame(
    db, tmp_path, monkeypatch, warmed
):
    """The opened track asks for a screenful of frames at once. The
    first request for a frame decodes the chunk of the track's frames
    from that frame on with one seek; the card's own frame never does,
    it has a stored crop; a photo has no track to warm."""
    f, _ = _video(db, tmp_path)
    crop = _jpeg(tmp_path, "track000001.jpg", size=(300, 300))
    track = make_track(db, file_id=f.id, start_frame=270, end_frame=330,
                       representative_frame_number=300, crop_path=str(crop))
    box = dict(file_id=f.id, track_id=track.id, bbox_x=0.5, bbox_y=0.25,
               bbox_width=0.2, bbox_height=0.3)
    make_detection(db, frame_number=270, **box)
    card = make_detection(db, frame_number=300, **box)
    sibling = make_detection(db, frame_number=330, **box)
    db.commit()
    monkeypatch.setattr(crop_service, "decode_frame_jpeg", lambda *a: _jpeg_bytes())

    assert get_or_create_crop(sibling.id, 200, db) is not None
    assert warmed == [(f.file_path, [270, 300, 330], 330, f.frame_rate)]

    assert get_or_create_crop(card.id, 200, db) is not None
    assert len(warmed) == 1


# ---------------------------------------------------------------------------
# Shaped crops: the window takes the frame's shape and slides inside the
# photo, so the dashboard's wide hero fills with the scene, not with blur.
# ---------------------------------------------------------------------------


from app.services.crop_service import compute_expanded_crop_region  # noqa: E402

W, H = 1000, 600


def _contains_box(region, bbox):
    left, top, right, bottom = region
    bx, by, bw, bh = bbox
    return (
        left <= bx * W and right >= (bx + bw) * W
        and top <= by * H and bottom >= (by + bh) * H
    )


def test_a_square_crop_is_unchanged_and_still_centred():
    """The Labels grid overlay is computed from a centred square; the
    shaped path must not have moved it, even off the photo's edge."""
    region = compute_expanded_crop_region(0.0, 0.4, 0.1, 0.2, W, H)

    # box 100x120 px, padding 12, side 144, centre (50, 300)
    assert region == (-22, 228, 122, 372)


def test_a_shaped_crop_has_the_asked_shape_and_holds_the_whole_box():
    bbox = (0.4, 0.3, 0.1, 0.3)
    left, top, right, bottom = compute_expanded_crop_region(
        *bbox, W, H, aspect=1.6
    )

    assert abs((right - left) / (bottom - top) - 1.6) < 0.02
    assert _contains_box((left, top, right, bottom), bbox)


def test_a_box_at_the_edge_slides_the_window_into_the_photo():
    """No blur where the photo has pixels: the frame pans inward."""
    bbox = (0.0, 0.4, 0.1, 0.2)
    region = compute_expanded_crop_region(*bbox, W, H, aspect=1.6)

    left, top, right, bottom = region
    assert left == 0 and top >= 0 and right <= W and bottom <= H
    assert _contains_box(region, bbox)


def test_a_window_larger_than_the_photo_stays_centred():
    """A tall box in a wide frame needs more width than the photo has;
    nothing can slide, so it falls back to the square's blurred edges."""
    bbox = (0.45, 0.0, 0.1, 1.0)
    left, top, right, bottom = compute_expanded_crop_region(
        *bbox, W, H, aspect=2.0
    )

    assert right - left > W
    assert abs((left + right) / 2 - 500) <= 1
    assert _contains_box((left, top, right, bottom), bbox)


def test_a_shaped_crop_comes_out_at_that_shape(db, tmp_path):
    photo = _jpeg(tmp_path, "wide.jpg", size=(1000, 600))
    dep = make_deployment(db, project_id=make_project(db).id)
    f = make_file(db, deployment_id=dep.id, file_path=str(photo))
    d = make_detection(db, file_id=f.id, bbox_x=0.0, bbox_y=0.4,
                       bbox_width=0.1, bbox_height=0.2)

    square = Image.open(io.BytesIO(get_or_create_crop(d.id, 200, db)))
    wide = Image.open(io.BytesIO(get_or_create_crop(d.id, 320, db, aspect=1.6)))

    assert square.size == (200, 200)
    assert wide.size == (320, 200)


def _card_with_stored_crop(db, tmp_path, crop_exists=True):
    """A track's card off the cover, with a red stored crop; the clip's
    frames decode to green."""
    f, _ = _video(db, tmp_path)
    crop = tmp_path / "track000001.jpg"
    if crop_exists:
        Image.new("RGB", (300, 300), (220, 30, 30)).save(crop, "JPEG")
    track = make_track(db, file_id=f.id, start_frame=270, end_frame=330,
                       representative_frame_number=300, crop_path=str(crop))
    card = make_detection(db, file_id=f.id, frame_number=300, track_id=track.id,
                          bbox_x=0.4, bbox_y=0.1, bbox_width=0.2, bbox_height=0.8)
    db.commit()
    return f, card


def test_a_shaped_crop_of_a_tracks_card_is_cut_from_its_frame(db, tmp_path, monkeypatch):
    """The stored track crop is the padded square. A shape cut out of its
    centre lost the animal's head and feet (calibration at 4:3, the
    dashboard hero at 16:10), so a shaped request decodes the card's
    frame and is cut like a photo, with the whole box in it."""
    f, card = _card_with_stored_crop(db, tmp_path)
    asked = []

    def fake_decode(path, frame_number, rate):
        asked.append(frame_number)
        return _jpeg_bytes(colour=(20, 200, 20))

    monkeypatch.setattr(crop_service, "decode_frame_jpeg", fake_decode)

    picture, bbox = _resolve_source(f, card, square=False)
    assert isinstance(picture, bytes) and bbox == [0.4, 0.1, 0.2, 0.8]
    wide = Image.open(io.BytesIO(get_or_create_crop(card.id, 320, db, aspect=1.6)))
    assert wide.size == (320, 200)
    assert wide.getpixel((160, 100))[1] > 150
    assert asked == [300, 300]

    # The square card is still the stored crop, with nothing decoded.
    square = Image.open(io.BytesIO(get_or_create_crop(card.id, 100, db)))
    assert square.getpixel((50, 50))[0] > 180
    assert asked == [300, 300]


def test_a_card_whose_stored_crop_is_gone_is_cut_from_its_frame(db, tmp_path, monkeypatch):
    """A bundle without photos, connected to a copy of the videos that was
    never analysed, has the tracks and no `.addaxai` folder. The card is
    decoded from the clip instead of left without a picture."""
    f, card = _card_with_stored_crop(db, tmp_path, crop_exists=False)
    monkeypatch.setattr(
        crop_service, "decode_frame_jpeg", lambda *a: _jpeg_bytes(colour=(20, 200, 20))
    )

    picture, bbox = _resolve_source(f, card)
    assert isinstance(picture, bytes) and bbox == [0.4, 0.1, 0.2, 0.8]
    served = Image.open(io.BytesIO(get_or_create_crop(card.id, 100, db)))
    assert served.size == (100, 100) and served.getpixel((50, 50))[1] > 150


def test_a_box_that_spans_the_photo_is_not_squeezed_into_the_square(db, tmp_path):
    """A whole-image classifier stamps a box over the full photo. It is a
    box like any other: the card is a square cut around it, not the 5:3
    photo pressed into a square."""
    photo = tmp_path / "wide.jpg"
    img = Image.new("RGB", (1000, 600), (20, 20, 200))
    img.paste((220, 30, 30), (400, 200, 600, 400))  # a red square in the middle
    img.save(photo, "JPEG", quality=95)
    dep = make_deployment(db, project_id=make_project(db).id)
    f = make_file(db, deployment_id=dep.id, file_path=str(photo))
    d = make_detection(db, file_id=f.id, bbox_x=0.0, bbox_y=0.0,
                       bbox_width=1.0, bbox_height=1.0)

    assert _resolve_source(f, d) == (photo, [0.0, 0.0, 1.0, 1.0])
    card = Image.open(io.BytesIO(get_or_create_crop(d.id, 200, db)))

    red = [x for x in range(200) if card.getpixel((x, 100))[0] > 150]
    red_rows = [y for y in range(200) if card.getpixel((100, y))[0] > 150]
    assert abs(len(red) - len(red_rows)) <= 2


def test_a_crop_from_a_rotated_photo_shows_the_same_pixels(db, tmp_path):
    """MegaDetector reports boxes on the EXIF-rotated pixels, so the tile
    must be cut from those as well. A photo stored on its side with
    orientation 6 and the upright photo give the same crop for the same
    box; without the rotation the tile showed a sideways slice of the
    wrong corner."""
    upright = Image.new("RGB", (400, 240), (30, 30, 30))
    upright.paste((220, 40, 40), (300, 0, 400, 120))  # red block, top right
    upright.save(tmp_path / "upright.jpg", "JPEG", quality=95)
    exif = Image.Exif()
    exif[0x0112] = 6
    upright.rotate(90, expand=True).save(
        tmp_path / "sideways.jpg", "JPEG", quality=95, exif=exif.tobytes()
    )
    dep = make_deployment(db, project_id=make_project(db).id)
    box = dict(bbox_x=0.75, bbox_y=0.0, bbox_width=0.25, bbox_height=0.5)
    crops = []
    for name in ("upright.jpg", "sideways.jpg"):
        f = make_file(db, deployment_id=dep.id, file_path=str(tmp_path / name))
        d = make_detection(db, file_id=f.id, **box)
        crops.append(Image.open(io.BytesIO(get_or_create_crop(d.id, 100, db))))

    for crop in crops:
        r, g, b = crop.getpixel((50, 50))
        assert r > 180 and g < 80 and b < 80, "tile should be the red block"
    assert crops[0].size == crops[1].size
