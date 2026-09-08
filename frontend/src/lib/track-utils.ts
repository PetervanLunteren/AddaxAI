/**
 * What a track looks like on screen, derived from the file's boxes.
 *
 * A track row holds only its span and its representative frame; its
 * label and verdict live on its boxes (see the tracks section of
 * DEVELOPERS.md). The card modal, the video player and both viewers
 * read them through these three functions, so "which box stands for
 * the track", "what colour is it" and "is it verified" have one home.
 */

import type { DetectionResponse, FileWithDetections, TrackResponse } from "../api/types";
import {
  getDetectionColor,
  getDetectionDisplayName,
  isNonLabel,
  passesDrawFilter,
  shouldDrawBbox,
} from "./detection-utils";

export type TrackVerdict = "verified" | "rejected" | "unverified";

export interface TrackRow {
  track: TrackResponse;
  /** The species key the bar is grouped under: the representative box's
   *  label, else the detector's category. The Counts modal's MaxN
   *  markers carry the same key. */
  label: string;
  /** The species as the rest of the app prints it (common or scientific
   *  name per the name mode, the category otherwise). */
  displayLabel: string;
  color: string;
  verdict: TrackVerdict;
}

/** The box on the track's representative frame: the one its card shows. */
export function representativeBox(
  detections: DetectionResponse[],
  track: TrackResponse,
): DetectionResponse | undefined {
  return detections.find(
    (d) =>
      d.track_id === track.id &&
      d.frame_number === track.representative_frame_number,
  );
}

/**
 * One row per track, grouped by species and sorted by start frame
 * within a species. The label comes off the representative box (its
 * label, else the detector's category, and the category again once the
 * track is rejected); the verdict is "verified" when every box of the
 * track is, "rejected" when the representative box carries a non-label,
 * "unverified" otherwise.
 */
export function trackRows(
  tracks: TrackResponse[],
  detections: DetectionResponse[],
): TrackRow[] {
  const boxesByTrack = new Map<string, DetectionResponse[]>();
  for (const d of detections) {
    if (!d.track_id) continue;
    const list = boxesByTrack.get(d.track_id);
    if (list) list.push(d);
    else boxesByTrack.set(d.track_id, [d]);
  }
  const rows: TrackRow[] = [];
  for (const track of tracks) {
    const boxes = boxesByTrack.get(track.id) ?? [];
    const rep = representativeBox(boxes, track) ?? boxes[0];
    if (!rep) continue;
    const rejected = isNonLabel(rep.label);
    // A rejected track has no species any more, so its row is the
    // detector's category: the hollow bar sits beside the animals it was
    // taken for, instead of in a "False detection" row of its own.
    const named = rejected
      ? { ...rep, label: null, label_taxonomy_id: null, common_name: null, scientific_name: null }
      : rep;
    const verdict: TrackVerdict = rejected
      ? "rejected"
      : boxes.every((d) => d.verified)
        ? "verified"
        : "unverified";
    rows.push({
      track,
      label: named.label ?? named.category,
      displayLabel: getDetectionDisplayName(named),
      color: getDetectionColor(named),
      verdict,
    });
  }
  return rows.sort(
    (a, b) =>
      a.label.localeCompare(b.label) ||
      a.track.start_frame - b.track.start_frame,
  );
}

/**
 * Frames in the clip, for a bar to be a fraction of. The stored length
 * first; for a file analysed before the length was stored, the end of
 * its last track, so its detections still get bars (the lane then ends
 * at the last track rather than at the clip's real end).
 */
export function clipDurationFrames(file: {
  frame_rate: number | null;
  duration_seconds: number | null;
  tracks?: { end_frame: number }[];
}): number {
  if (file.duration_seconds && file.frame_rate) {
    return Math.round(file.duration_seconds * file.frame_rate);
  }
  const tracks = file.tracks ?? [];
  return tracks.length ? Math.max(...tracks.map((t) => t.end_frame)) + 1 : 0;
}

/**
 * One box per card a person can act on: a photo's boxes, and for a
 * video every track's representative box (a verdict on it reaches the
 * whole track through the cascade). The same gates the canvas draws
 * by. This is what "every detection in the file" means on the Files
 * tab: the tile chips, the viewer's actions with nothing selected, the
 * bulk bar and the Counts modal's landing file all read it.
 */
export function cardBoxes(
  file: FileWithDetections,
  threshold: number,
): DetectionResponse[] {
  if (file.file_type !== "video") {
    return file.detections.filter((d) => shouldDrawBbox(d, file, threshold));
  }
  const cards: DetectionResponse[] = [];
  for (const track of file.tracks ?? []) {
    const box = representativeBox(file.detections, track);
    if (box && passesDrawFilter(box, threshold) && box.bbox_x !== null) cards.push(box);
  }
  return cards;
}

/** Box centres of one track in frame order, in image fractions. */
export function trackPath(
  detections: DetectionResponse[],
  trackId: string,
): { frame: number; x: number; y: number }[] {
  return detections
    .filter(
      (d) =>
        d.track_id === trackId &&
        d.frame_number != null &&
        d.bbox_x != null &&
        d.bbox_y != null,
    )
    .map((d) => ({
      frame: d.frame_number as number,
      x: (d.bbox_x as number) + (d.bbox_width ?? 0) / 2,
      y: (d.bbox_y as number) + (d.bbox_height ?? 0) / 2,
    }))
    .sort((a, b) => a.frame - b.frame);
}
