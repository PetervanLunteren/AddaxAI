"""`vote_tracks`: every frame of a track carries the species its frames
agree on, weighted by how clearly each frame saw the animal.

Pure, like `test_track_summaries.py`: the input is a results dict as the
tracking script writes it; tracked boxes change, nothing else does.
"""

import copy

import pytest

from app.ml.track_vote import tracks_follow_their_cards, vote_tracks

FOX, WOLF, BOAR, DEER = "1", "2", "3", "4"


def _box(frame, conf, track=None, classifications=None):
    det = {
        "category": "1",
        "conf": conf,
        "bbox": [0.1, 0.1, 0.2, 0.2],
        "frame_number": frame,
    }
    if track is not None:
        det["track_id"] = track
    if classifications is not None:
        det["classifications"] = classifications
    return det


def _results(*dets, file="clip.mp4"):
    return {
        "classification_categories": {FOX: "fox", WOLF: "wolf", BOAR: "boar", DEER: "deer"},
        "images": [{"file": file, "detections": list(dets)}],
    }


def _top1(det):
    return det["classifications"][0][0]


def test_a_misread_card_frame_is_outvoted_by_its_track():
    """The card is the detector's most confident frame, not the
    classifier's most reliable one. Four clear frames saying wolf beat
    the card's own fox."""
    card = _box(60, 0.9, 1, [[FOX, 0.6], [WOLF, 0.4]])
    siblings = [_box(f, 0.8, 1, [[WOLF, 0.95], [FOX, 0.05]]) for f in (30, 90, 120, 150)]
    res = _results(card, *siblings)

    assert vote_tracks(res) == 1
    assert _top1(card) == WOLF
    # 0.9 * 0.4 + 4 * 0.8 * 0.95 over 0.9 + 3.2
    assert card["classifications"][0][1] == pytest.approx((0.36 + 3.04) / 4.1)


def test_blurry_tail_frames_weigh_less_than_sharp_ones():
    """Three half-visible frames read boar, two clear ones read deer. A
    plain mean would say boar; weighted by detection confidence, the
    clear views win."""
    card = _box(60, 0.9, 1, [[DEER, 0.9], [BOAR, 0.1]])
    sharp = _box(90, 0.9, 1, [[DEER, 0.9], [BOAR, 0.1]])
    tail = [_box(f, 0.12, 1, [[BOAR, 0.9], [DEER, 0.1]]) for f in (120, 150, 180)]
    res = _results(card, sharp, *tail)
    # A plain count of the top-1s would have been 3 boar to 2 deer.
    assert sum(_top1(d) == BOAR for d in tail) == 3

    vote_tracks(res)
    assert _top1(card) == DEER


def test_every_frame_of_the_track_carries_the_vote():
    """One animal, one species: a frame the classifier misread shows the
    track's species like the rest, on every surface (2026-11-05: a wolf
    read as a jackal on two frames showed a jackal there)."""
    card = _box(60, 0.9, 1, [[WOLF, 0.6], [FOX, 0.4]])
    misread = _box(90, 0.8, 1, [[FOX, 0.7], [WOLF, 0.3]])
    clear = _box(30, 0.8, 1, [[WOLF, 0.95], [FOX, 0.05]])
    res = _results(card, misread, clear)

    vote_tracks(res)
    assert _top1(misread) == WOLF
    assert card["classifications"] == misread["classifications"] == clear["classifications"]


def test_unclassified_frames_take_the_vote_without_voting_and_the_rest_is_untouched():
    """A box below the gate carries no scores and has no say, but it is
    the same animal, so it takes the track's species; a person track has
    no voters at all; a photo has no track. None of them count as a
    changed card."""
    weak = _box(30, 0.05, 1)
    card = _box(60, 0.9, 1, [[WOLF, 0.7], [FOX, 0.3]])
    person_track = [_box(f, 0.8, 2) for f in (200, 230)]
    photo = {
        "category": "1",
        "conf": 0.9,
        "bbox": [0, 0, 0.5, 0.5],
        "classifications": [[FOX, 0.9]],
    }
    res = {
        "classification_categories": {FOX: "fox", WOLF: "wolf"},
        "images": [
            {"file": "clip.mp4", "detections": [weak, card, *person_track]},
            {"file": "photo.jpg", "detections": [photo]},
            {"file": "broken.mp4", "detections": None},
        ],
    }
    snapshot = copy.deepcopy(res)

    assert vote_tracks(res) == 0
    assert weak["classifications"] == card["classifications"] == [[WOLF, 0.7], [FOX, 0.3]]
    snapshot["images"][0]["detections"][0]["classifications"] = [[WOLF, 0.7], [FOX, 0.3]]
    assert res == snapshot


def test_a_card_that_keeps_its_species_is_not_counted_as_changed():
    """The count is for the log: cards whose species changed, not cards
    whose scores moved in their decimals, which is nearly all of them."""
    card = _box(60, 0.9, 1, [[WOLF, 0.7], [FOX, 0.3]])
    other = _box(90, 0.8, 1, [[WOLF, 0.9], [FOX, 0.1]])
    res = _results(card, other)

    assert vote_tracks(res) == 0
    assert card["classifications"][0][0] == WOLF
    assert card["classifications"] != [[WOLF, 0.7], [FOX, 0.3]]


def test_a_single_voter_track_keeps_its_own_scores():
    card = _box(60, 0.9, 1, [[WOLF, 0.7], [FOX, 0.3]])
    res = _results(card, _box(30, 0.05, 1))

    assert vote_tracks(res) == 0
    assert card["classifications"] == [[WOLF, 0.7], [FOX, 0.3]]


def test_scores_come_back_sorted_at_full_precision():
    """Rollup reads the first five entries and smoothing the first one,
    so the list must lead with the strongest class. Not rounded: a score
    is rounded once, when it becomes a label confidence (SCORE_DIGITS)."""
    card = _box(60, 0.5, 1, [[FOX, 0.5], [BOAR, 0.3], [DEER, 0.2]])
    other = _box(90, 0.5, 1, [[DEER, 0.6], [WOLF, 0.4]])
    res = _results(card, other)

    vote_tracks(res)
    scores = card["classifications"]
    assert [c for c, _ in scores] == [DEER, FOX, WOLF, BOAR]
    assert scores == [
        [DEER, pytest.approx(0.4)], [FOX, pytest.approx(0.25)],
        [WOLF, pytest.approx(0.2)], [BOAR, pytest.approx(0.15)],
    ]


LYNX = "5"


def test_after_smoothing_a_split_track_takes_its_cards_species():
    """MegaDetector's smoother relabels only the clear boxes, so it can
    turn the card and one other clear box of a fox track to lynx and
    leave the faint ones fox (bug hunt, 2026-11-06). The card's verdict
    then holds for the whole animal."""
    card = _box(60, 0.9, 1, [[LYNX, 0.8]])
    clear = _box(90, 0.8, 1, [[LYNX, 0.8]])
    faint = [_box(f, 0.3, 1, [[FOX, 0.7], [WOLF, 0.3]]) for f in range(100, 240, 10)]
    other = _box(60, 0.9, 2, [[LYNX, 0.9]])
    res = _results(card, clear, *faint, other)

    assert tracks_follow_their_cards(res) == len(faint)
    assert all(b["classifications"] == [[LYNX, 0.8]] for b in [card, clear, *faint])
    assert other["classifications"] == [[LYNX, 0.9]]
    assert tracks_follow_their_cards(res) == 0  # nothing left to change


def test_a_card_without_scores_and_untracked_boxes_are_left_alone():
    """A person track has no scores to hand on; a photo has no track."""
    person = [_box(f, 0.8, 1) for f in (30, 60)]
    photo = {"category": "1", "conf": 0.9, "bbox": [0, 0, 0.5, 0.5],
             "classifications": [[FOX, 0.9]]}
    res = {"images": [{"file": "clip.mp4", "detections": person},
                      {"file": "photo.jpg", "detections": [photo]}]}
    snapshot = copy.deepcopy(res)

    assert tracks_follow_their_cards(res) == 0
    assert res == snapshot


def test_every_box_gets_its_own_copy_of_the_scores():
    """Rollup replaces a box's list today; if any step ever edits one in
    place, it must not reach the rest of the track."""
    card = _box(60, 0.9, 1, [[WOLF, 0.9]])
    other = _box(90, 0.8, 1, [[WOLF, 0.6], [FOX, 0.4]])
    res = _results(card, other)

    vote_tracks(res)
    other["classifications"][0][1] = 0.0
    assert card["classifications"][0][1] != 0.0

    other["classifications"] = [[FOX, 0.9]]
    tracks_follow_their_cards(res)
    other["classifications"][0][1] = 0.0
    assert card["classifications"][0][1] != 0.0
