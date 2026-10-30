"""A track's card carries the species its frames agree on.

A video card stands for one animal, but until 2026-10-30 its species was
the classifier's guess on one frame: the box where the detector was most
confident (`summarise_tracks`), which is the clearest view of the animal
but not a more reliable read of its species than any other sharp frame.
The other frames of the same animal were never asked, and a single
misread on that frame named the card, and through the card decided
which species reached the counts.

This asks them. Each classified frame carries its top class scores, and
the card gets their confidence-weighted mean: a sharp frame counts
fully, a blurry tail frame barely, so the half-visible frames where an
animal enters or leaves cannot outvote the clear views. Combining a
track's frames beats a single frame in every comparison we found (Mei
et al. on longline fishing video; two 2026 camera-trap papers), and
weighting is what guards against edge frames that are all misread the
same way.

Only the card changes. Every other frame keeps its own guess, because
opening a track is how a person sees that the tracker switched animals
halfway, and MaxN per species must see that switch too. The vote runs
in memory inside postprocessing, after label exclusion and before
taxonomic rollup, so rollup rolls up the track's distribution rather
than one frame's; the JSON on disk stays raw and `original_label` stays
the card frame's own top-1. Pixel-free and pure, like `summarise_tracks`.
"""

from __future__ import annotations

from collections import defaultdict

from app.ml.inference.scoring import summarise_tracks


def vote_track_cards(md_results: dict) -> int:
    """Give each track's card box the confidence-weighted mean of its
    frames' class scores. In place; returns how many cards changed their
    top class, which is what a reader of the log wants to know (nearly
    every card's list changes in its decimals).

    For every entry with detections: `summarise_tracks` names the card
    frame per track and ignores boxes without a track id, so photos and
    drawn boxes are never touched. The voters are the track's boxes that
    carry `classifications`, which is the frames at or above the
    classification gate, the same frames the opened track shows.
    ``score[class] = sum(p * conf) / sum(conf)`` over the voters, sorted
    descending and rounded to 5 decimals like the worker writes them,
    written onto the card box only. The whole list is kept: rollup reads
    its own top-5 and smoothing its top-1.

    A track with no voters (a person, a vehicle, a run without a
    classifier) is left as it is. A track whose only voter is the card
    comes out as its own list, up to rounding.
    """
    changed = 0
    for img in md_results.get("images") or []:
        dets = img.get("detections") or []
        summaries = summarise_tracks(dets)
        if not summaries:
            continue

        by_track: dict[int, list[dict]] = defaultdict(list)
        for det in dets:
            if det.get("track_id") is not None and det.get("frame_number") is not None:
                by_track[int(det["track_id"])].append(det)

        for track_id, boxes in by_track.items():
            voters = [b for b in boxes if b.get("classifications")]
            if not voters:
                continue
            card_frame = summaries[track_id].representative_frame_number
            card = next(b for b in boxes if int(b["frame_number"]) == card_frame)

            weight = sum(float(b["conf"]) for b in voters)
            if weight <= 0:
                continue
            score: dict[str, float] = defaultdict(float)
            for b in voters:
                conf = float(b["conf"])
                for class_id, p in b["classifications"]:
                    score[str(class_id)] += float(p) * conf

            voted = [
                [class_id, round(total / weight, 5)]
                for class_id, total in sorted(score.items(), key=lambda item: (-item[1], item[0]))
            ]
            before = card.get("classifications") or []
            if voted != before:
                card["classifications"] = voted
                if not before or voted[0][0] != str(before[0][0]):
                    changed += 1
    return changed
