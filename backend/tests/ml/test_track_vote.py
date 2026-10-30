"""`vote_track_cards`: a track's card carries the species its frames
agree on, weighted by how clearly each frame saw the animal.

Pure, like `test_track_summaries.py`: the input is a results dict as the
tracking script writes it, and only the card box may change.
"""

import copy

from app.ml.track_vote import vote_track_cards

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

    assert vote_track_cards(res) == 1
    assert _top1(card) == WOLF
    # 0.9 * 0.4 + 4 * 0.8 * 0.95 over 0.9 + 3.2
    assert card["classifications"][0][1] == round((0.36 + 3.04) / 4.1, 5)


def test_blurry_tail_frames_weigh_less_than_sharp_ones():
    """Three half-visible frames read boar, two clear ones read deer. A
    plain mean would say boar; weighted by detection confidence, the
    clear views win."""
    card = _box(60, 0.9, 1, [[DEER, 0.9], [BOAR, 0.1]])
    sharp = _box(90, 0.9, 1, [[DEER, 0.9], [BOAR, 0.1]])
    tail = [_box(f, 0.12, 1, [[BOAR, 0.9], [DEER, 0.1]]) for f in (120, 150, 180)]
    res = _results(card, sharp, *tail)

    vote_track_cards(res)
    assert _top1(card) == DEER
    # Plain mean of the top-1s would have been 3 boar to 2 deer.
    plain = sum(d["classifications"][0][1] for d in tail) / 5
    assert plain > 0.5


def test_only_the_card_changes():
    """Siblings keep their own guesses: opening the track is how a person
    sees that the tracker switched animals, and MaxN per species must see
    the switch too."""
    card = _box(60, 0.9, 1, [[FOX, 0.6], [WOLF, 0.4]])
    siblings = [_box(f, 0.8, 1, [[WOLF, 0.95], [FOX, 0.05]]) for f in (30, 90)]
    before = copy.deepcopy(siblings)
    res = _results(card, *siblings)

    vote_track_cards(res)
    assert siblings == before


def test_boxes_without_classifications_do_not_vote_and_untracked_boxes_are_untouched():
    """A box below the gate carries no scores and stays silent; a person
    track has no voters at all; a photo has no track. None of them count
    as a changed card."""
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

    assert vote_track_cards(res) == 0
    assert res == snapshot


def test_a_card_that_keeps_its_species_is_not_counted_as_changed():
    """The count is for the log: cards whose species changed, not cards
    whose scores moved in their decimals, which is nearly all of them."""
    card = _box(60, 0.9, 1, [[WOLF, 0.7], [FOX, 0.3]])
    other = _box(90, 0.8, 1, [[WOLF, 0.9], [FOX, 0.1]])
    res = _results(card, other)

    assert vote_track_cards(res) == 0
    assert card["classifications"][0][0] == WOLF
    assert card["classifications"] != [[WOLF, 0.7], [FOX, 0.3]]


def test_a_single_voter_track_keeps_its_own_scores():
    card = _box(60, 0.9, 1, [[WOLF, 0.7], [FOX, 0.3]])
    res = _results(card, _box(30, 0.05, 1))

    assert vote_track_cards(res) == 0
    assert card["classifications"] == [[WOLF, 0.7], [FOX, 0.3]]


def test_scores_come_back_sorted_and_rounded():
    """Rollup reads the first five entries and smoothing the first one,
    so the list must lead with the strongest class, in the five-decimal
    form the classification worker writes."""
    card = _box(60, 0.5, 1, [[FOX, 0.5], [BOAR, 0.3], [DEER, 0.2]])
    other = _box(90, 0.5, 1, [[DEER, 0.6], [WOLF, 0.4]])
    res = _results(card, other)

    vote_track_cards(res)
    scores = card["classifications"]
    assert [c for c, _ in scores] == [DEER, FOX, WOLF, BOAR]
    assert scores == [[DEER, 0.4], [FOX, 0.25], [WOLF, 0.2], [BOAR, 0.15]]
    assert all(round(p, 5) == p for _, p in scores)
