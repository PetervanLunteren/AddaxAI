"""A track carries one species: the one its frames agree on.

A track is one animal, but until 2026-10-30 its card's species was the
classifier's guess on one frame: the box where the detector was most
confident (`summarise_tracks`), which is the clearest view of the animal
but not a more reliable read of its species than any other sharp frame.
The other frames of the same animal were never asked, and a single
misread on that frame named the card, and through the card decided
which species reached the counts.

This asks them. Each classified frame carries its top class scores, and
the track gets their confidence-weighted mean: a sharp frame counts
fully, a blurry tail frame barely, so the half-visible frames where an
animal enters or leaves cannot outvote the clear views. Combining a
track's frames beats a single frame in every comparison we found (Mei
et al. on longline fishing video; two 2026 camera-trap papers, one of
which names the per-frame flicker it removes), and weighting is what
guards against edge frames that are all misread the same way.

Every frame of the track gets the voted scores, the frames below the
classification gate included, so a box has the same species on the
still, in the player, in the Frames view, in MaxN and in the exports.
Until 2026-11-05 only the card did and the other frames kept their own
guess, so a wolf read as a golden jackal on two frames showed a jackal
on those frames under a bar that said wolf, and on a busy frame a
misread lowered its species' count by one. CVAT, DIVE and SharkTrack
give a track one species the same way. A tracker that switched animals
halfway is put right by hand: frames a person relabels keep their label
(postprocessing never touches a person's decision). The vote runs in
memory inside postprocessing, after label exclusion and before taxonomic
rollup, so rollup rolls up the track's distribution rather than one
frame's; the JSON on disk keeps every frame's own scores. Pixel-free and
pure, like `summarise_tracks`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator

from app.ml.inference.scoring import summarise_tracks


def _tracks(md_results: dict) -> Iterator[tuple[dict, list[dict]]]:
    """Each track's card box and all its boxes, entry by entry.
    `summarise_tracks` names the card frame and ignores boxes without a
    track id, so photos and drawn boxes never come out of here."""
    for img in md_results.get("images") or []:
        dets = img.get("detections") or []
        summaries = summarise_tracks(dets)
        by_track: dict[int, list[dict]] = defaultdict(list)
        for det in dets:
            if det.get("track_id") is not None and det.get("frame_number") is not None:
                by_track[int(det["track_id"])].append(det)
        for track_id, boxes in by_track.items():
            card_frame = summaries[track_id].representative_frame_number
            card = next(b for b in boxes if int(b["frame_number"]) == card_frame)
            yield card, boxes


def _give(boxes: list[dict], classifications: list[list]) -> None:
    """Each box its own copy, so no later in-place edit of one box can
    reach the rest of its track."""
    for b in boxes:
        b["classifications"] = [list(pair) for pair in classifications]


def vote_tracks(md_results: dict) -> int:
    """Give every box of each track the confidence-weighted mean of its
    frames' class scores. In place; returns how many cards changed their
    top class, which is what a reader of the log wants to know (nearly
    every card's list changes in its decimals).

    The voters are the track's boxes that carry `classifications`, which
    is the frames at or above the classification gate; the weaker frames
    of the track were never classified and have nothing to say.
    ``score[class] = sum(p * conf) / sum(conf)`` over the voters, sorted
    descending and kept at full precision (a score is rounded once, when it
    becomes a label confidence: SCORE_DIGITS), written onto every box of
    the track, those that did not vote too. The whole list is kept:
    rollup reads its own top-5 and smoothing its top-1.

    A track with no voters (a person, a vehicle, a run without a
    classifier) is left as it is. A track whose only voter is the card
    comes out as its own list.
    """
    changed = 0
    for card, boxes in _tracks(md_results):
        voters = [b for b in boxes if b.get("classifications")]
        weight = sum(float(b["conf"]) for b in voters)
        if weight <= 0:
            continue
        score: dict[str, float] = defaultdict(float)
        for b in voters:
            conf = float(b["conf"])
            for class_id, p in b["classifications"]:
                score[str(class_id)] += float(p) * conf

        voted = [
            [class_id, total / weight]
            for class_id, total in sorted(score.items(), key=lambda item: (-item[1], item[0]))
        ]
        before = card.get("classifications") or []
        if not before or voted[0][0] != str(before[0][0]):
            changed += 1
        _give(boxes, voted)
    return changed


def tracks_follow_their_cards(md_results: dict) -> int:
    """Give every box of each track its card's scores again, after
    MegaDetector's smoothing. In place; returns how many boxes changed.

    The smoother judges box by box, and only the boxes at or above the
    counting threshold, so it can relabel the two clear boxes of a fox
    track (the card among them) to the lynx that fills the clip and leave
    its fourteen faint ones fox: one animal, two species, the split the
    vote exists to prevent (bug hunt, 2026-11-06). The smoother still
    decides; its verdict on the card, the track's clearest box, now holds
    for the whole animal. A track whose card carries no scores is left
    alone.
    """
    changed = 0
    for card, boxes in _tracks(md_results):
        scores = card.get("classifications")
        if not scores:
            continue
        stale = [b for b in boxes if b.get("classifications") != scores]
        changed += len(stale)
        _give(stale, scores)
    return changed
