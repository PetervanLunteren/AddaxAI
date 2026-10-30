/**
 * What a track looks like on screen, derived from the file's boxes.
 *
 * A track row holds only its span and its representative frame; its
 * label and verdict live on its boxes (see the tracks section of
 * DEVELOPERS.md). The card modal, the video player and both viewers
 * read them through these three functions, so "which box stands for
 * the track", "what colour is it" and "is it verified" have one home.
 */

import type {
  DetectionResponse,
  DetectionSummary,
  FileWithDetections,
  TrackDetectionRow,
  TrackResponse,
} from "../api/types";
import {
  type BboxedDetection,
  getDetectionColor,
  getDetectionDisplayName,
  inScope,
  isDrawableBox,
  isNonLabel,
  passesDrawFilter,
} from "./detection-utils";

export type TrackVerdict = "verified" | "rejected" | "unverified";

export interface TrackRow {
  track: TrackResponse;
  /** The key the bar is grouped under: the representative box's label
   *  (so a rejected track groups under "false detection", the same word
   *  its card shows), else the detector's category. The Counts modal's
   *  MaxN markers carry the same key. */
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
 * within a species, with the rejected tracks in a "False detection"
 * group after every species. The label comes off the representative
 * box (its label, else the detector's category). Until 2026-10-30 a
 * rejected track was filed under the detector's category instead, so
 * an animal pressed X on sat hollow in a row called "Animal", which
 * reads as the opposite of what X says; the species is gone the moment
 * a track is rejected, so there is no species row for it to keep. The
 * verdict is the representative box's too:
 * "rejected" when it carries a non-label, "verified" when it is
 * flagged, "unverified" otherwise. The card is where a verdict on an
 * animal lands (`verdict_targets` in the backend flags the card and
 * carries only the label to the other frames), so the card is what the
 * grid, the counts and the sign-off read, and the bar must read the
 * same. Until 2026-10-30 it demanded every frame be flagged, which a
 * card verdict never does, so a verified animal stayed hatched and the
 * "Mark correct" walk kept offering it.
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
    const verdict: TrackVerdict = rejected
      ? "rejected"
      : rep.verified
        ? "verified"
        : "unverified";
    rows.push({
      track,
      label: rep.label ?? rep.category,
      displayLabel: getDetectionDisplayName(rep),
      color: getDetectionColor(rep),
      verdict,
    });
  }
  // Rejected tracks last: their "False detection" row never pushes a
  // species down the lane.
  return rows.sort(
    (a, b) =>
      Number(a.verdict === "rejected") - Number(b.verdict === "rejected") ||
      a.label.localeCompare(b.label) ||
      a.track.start_frame - b.track.start_frame,
  );
}

/** One species on the timeline, its animals packed into as few rows as
 *  they need. */
export interface TimelineGroup {
  /** The species key, what the MaxN markers filter on. */
  label: string;
  /** The species as the row label prints it. */
  displayLabel: string;
  /** Each row holds tracks that never overlap in time. */
  rows: TrackRow[][];
}

/**
 * The timeline's rows: one group per species, and inside a group as
 * many rows as its animals need to sit beside each other.
 *
 * One rule, no tuning: a track goes in the first row whose last track
 * ends before this one starts, else in a new row. Walking tracks in
 * start order makes that greedy pass land on the fewest rows there can
 * be, which is the most animals of that species ever on screen at once.
 * A clip of 702 sharks needs seven.
 *
 * Until 2026-09-11 a species was one row and every one of its tracks
 * was drawn inside it, so on a shark drop the bars sat on top of each
 * other: one smear, and a click landed on whichever painted last.
 *
 * Ends increase along a row by construction (a track joins only when
 * the last one ended before it starts), so the last member is the one
 * to compare against and no scan is needed. `trackRows` already sorts
 * by label and then by start frame, so this is one linear walk.
 */
export function packTrackRows(rows: TrackRow[]): TimelineGroup[] {
  const groups = new Map<string, TimelineGroup>();
  for (const row of rows) {
    let group = groups.get(row.label);
    if (!group) {
      group = { label: row.label, displayLabel: row.displayLabel, rows: [] };
      groups.set(row.label, group);
    }
    const fits = group.rows.find(
      (packed) => packed[packed.length - 1].track.end_frame < row.track.start_frame,
    );
    if (fits) fits.push(row);
    else group.rows.push([row]);
  }
  return [...groups.values()];
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
 * Track id to its card, for the tracks a clip has on show: those whose
 * card is in scope (`inScope`: passes the counting threshold or was
 * verified as something real), the rule a photo's box is judged by. A
 * track is one thing on every surface, so this one decision drives the
 * bar on the timeline, the boxes in the player and the cards, and
 * inside a track on show nothing is hidden: the weak boxes the tracker
 * kept to follow the animal are drawn, shown and spanned like the rest.
 * Until 2026-10-30 the player cut per box at the counting threshold
 * while the bar spanned every box, so boxes blinked along a solid bar.
 *
 * A rejected card above the threshold is in the map: the timeline draws
 * its track hollow beside the animals it was taken for. One pass over
 * the boxes, so a clip of 702 sharks costs the same as one of two.
 */
export function trackCards(
  tracks: TrackResponse[],
  detections: DetectionResponse[],
  threshold: number,
): Map<string, DetectionResponse> {
  const cardFrame = new Map(tracks.map((t) => [t.id, t.representative_frame_number]));
  const cards = new Map<string, DetectionResponse>();
  for (const d of detections) {
    if (d.track_id == null || cardFrame.get(d.track_id) !== d.frame_number) continue;
    if (d.bbox_x !== null && inScope(d, threshold)) cards.set(d.track_id, d);
  }
  return cards;
}

/**
 * Whether one box of a clip is outlined. Inside a track on show every
 * frame draws, weak ones included, so boxes never blink along a solid
 * bar; only a rejected frame or a box without geometry stays off
 * (`isDrawableBox`). A box no track claims (a clip analysed before
 * tracking) is judged on its own, as a photo's box is. The one rule
 * the player and every rendered still share.
 */
export function drawsInClip(
  d: DetectionResponse,
  onShow: Map<string, DetectionResponse>,
  threshold: number,
): boolean {
  return d.track_id != null
    ? onShow.has(d.track_id) && isDrawableBox(d)
    : passesDrawFilter(d, threshold);
}

/**
 * The boxes a rendered frame of a file draws: every outlined box of a
 * photo, and for a video the boxes on the frame on screen (the cover
 * by default, or `frameNumber` when the surface shows another frame).
 * A clip's other boxes still exist in the data, but they must not
 * paint onto the canvas of an unrelated frame: that is the crop-service
 * bug in another costume, and it looks perfectly fine until the
 * subject moves. One function for the Files viewer still, the Tab
 * list, the filmstrip tiles and the card modal, so a still can never
 * show fewer boxes than the player does on the same frame, which it
 * did until 2026-10-30.
 */
export function frameBoxes(
  file: FileWithDetections,
  threshold: number,
  frameNumber?: number,
): BboxedDetection[] {
  if (file.file_type !== "video") {
    return file.detections.filter((d) => passesDrawFilter(d, threshold)) as BboxedDetection[];
  }
  const shown = frameNumber ?? file.best_frame_number;
  const onShow = trackCards(file.tracks ?? [], file.detections, threshold);
  return file.detections.filter(
    (d) => (shown == null || d.frame_number === shown) && drawsInClip(d, onShow, threshold),
  ) as BboxedDetection[];
}

/**
 * One box per card a person can act on: a photo's boxes, and for a
 * video the card of every track on show that was not rejected (a
 * verdict on it reaches the whole track through the cascade). This is
 * what "every detection in the file" means on the Files tab: the tile
 * chips, the viewer's actions with nothing selected, the bulk bar and
 * the Counts modal's landing file all read it.
 */
export function cardBoxes(
  file: FileWithDetections,
  threshold: number,
): DetectionResponse[] {
  if (file.file_type !== "video") {
    return frameBoxes(file, threshold);
  }
  const cards = trackCards(file.tracks ?? [], file.detections, threshold);
  return (file.tracks ?? []).flatMap((track) => {
    const card = cards.get(track.id);
    return card && !isNonLabel(card.label) ? [card] : [];
  });
}

/**
 * The cards still waiting for a verdict, in the order they are shown:
 * a clip's tracks in the timeline's own row order, a photo's boxes in
 * canvas order. "Mark correct" walks this, so agreeing with the AI on a
 * whole clip is one key per animal with no reaching for the mouse.
 *
 * Reads the verdict from `trackRows`, the same function that colours
 * the bars, so what the viewer calls unchecked and what the timeline
 * draws hatched can never drift apart.
 */
export function uncheckedCards(
  file: FileWithDetections,
  threshold: number,
): DetectionResponse[] {
  const cards = cardBoxes(file, threshold);
  if (file.file_type !== "video") return cards.filter((d) => !d.verified);
  const byTrack = new Map(cards.map((d) => [d.track_id, d]));
  const pending: DetectionResponse[] = [];
  for (const row of trackRows(file.tracks ?? [], file.detections)) {
    if (row.verdict !== "unverified") continue;
    const card = byTrack.get(row.track.id);
    if (card) pending.push(card);
  }
  return pending;
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


/**
 * One frame of an opened track, shaped as a grid card.
 *
 * The Detections grid renders `DetectionSummary`, which the labels
 * subprocess builds. A frame comes from the track endpoint instead, so
 * it carries its own label, verdict and crop box but knows nothing
 * about where it sits in the project. Those context fields are copied
 * from the card that was opened: every frame of a track is in the same
 * file, so the same deployment, site, event and capture time.
 *
 * `crop_url` is built the way the subprocess builds it, at the same
 * size, so both kinds of card hit one endpoint and share one browser
 * cache entry.
 */
export function trackFrameAsCard(
  frame: TrackDetectionRow,
  parent: DetectionSummary,
): DetectionSummary {
  return {
    detection_id: frame.id,
    // Every frame of a track is in the same file as the card that opened
    // it, and the wire type for a detection carries no file id.
    file_id: parent.file_id,
    label: frame.label,
    label_taxonomy_id: frame.label_taxonomy_id,
    label_confidence: frame.label_confidence,
    common_name: frame.common_name,
    scientific_name: frame.scientific_name,
    confidence: frame.confidence,
    category: frame.category,
    verified: frame.verified,
    classification_method: frame.classification_method,
    // Similarity and neighbour agreement are properties of a place in
    // the sorted grid. A frame has no place in it, so it has none.
    distance_to_centroid: null,
    similarity: null,
    neighbor_agreement: null,
    neighbor_top_label: null,
    neighbor_top_common_name: null,
    neighbor_top_scientific_name: null,
    site_name: parent.site_name,
    deployment_id: parent.deployment_id,
    captured_at_local: parent.captured_at_local,
    event_id: parent.event_id,
    event_start_local: parent.event_start_local,
    crop_url: `/api/detections/${frame.id}/crop?size=200`,
    crop_bbox: frame.crop_bbox,
    frame_number: frame.frame_number,
    track_id: frame.track_id,
    // A frame is not itself a track, so it carries no badge: opening one
    // would be opening what you are already looking at.
    track_frames: null,
    file_flagged: parent.file_flagged,
    file_favorited: parent.file_favorited,
  };
}
