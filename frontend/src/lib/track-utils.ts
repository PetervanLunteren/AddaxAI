/**
 * What a track looks like on screen, derived from the file's boxes.
 *
 * A track row holds only its span and its representative frame; its
 * label and verdict live on its boxes (see the tracks section of
 * DEVELOPERS.md). The card modal, the video player and both viewers
 * read them through these three functions, so "which box stands for
 * the track", "what colour is it" and "is it verified" have one home.
 */

import type { CSSProperties } from "react";

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
  /** The bar's extent in frames, half a sampling step either side of
   *  the first and last box (`trackGaps`). */
  startFrame: number;
  endFrame: number;
  /** The holes in the bar, in frames: where a sample was missed, the
   *  detector saw nothing of the animal (`trackGaps`). */
  gaps: [number, number][];
}

/**
 * Native frames between two consecutive samples of a clip, measured
 * from the frame numbers themselves: the smallest positive gap. 0 with
 * fewer than two distinct frames. Not `Project.video_fps`: a clip
 * analysed before that setting last changed keeps its own spacing, and
 * the file's numbers cannot disagree with the file. The player
 * (`buildOverlayIndex`) and the timeline's bars read the same number.
 */
export function samplingStep(frames: Iterable<number>): number {
  const sorted = [...new Set(frames)].sort((a, b) => a - b);
  let step = 0;
  for (let i = 1; i < sorted.length; i++) {
    const gap = sorted[i] - sorted[i - 1];
    if (step === 0 || gap < step) step = gap;
  }
  return step;
}

/**
 * The holes in a track's bar: each `[from, to)` in frames where a box
 * is followed by its next box more than one step later. On the bar a
 * box is drawn centred on its frame, half a step either side, so two
 * boxes one step apart touch, a missed sample is a hole one step wide,
 * and the playhead on a sampled frame sits in the middle of colour or
 * in the middle of a hole, never on an edge (the player's own rule,
 * one step from the box's frame on, put the first missed sample's edge
 * exactly under the playhead). The tracker keeps a track alive through
 * up to two seconds without a box, so on a dense drop this is a sixth
 * of all bar length (HGA_022: 14,178 missed samples in 5,154 tracks);
 * the bar used to run solid across them and claim an animal on frames
 * the still and the player showed nothing on (2026-11-03).
 */
export function trackGaps(frames: number[], step: number): [number, number][] {
  if (frames.length < 2 || step <= 0) return [];
  const sorted = [...frames].sort((a, b) => a - b);
  const gaps: [number, number][] = [];
  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i] - sorted[i - 1] > step) {
      gaps.push([sorted[i - 1] + step / 2, sorted[i] - step / 2]);
    }
  }
  return gaps;
}

/** A gap's connector: a line this many pixels tall through the middle
 *  of the bar, in the bar's own colour. Thin says "the same animal,
 *  not seen here", a shape nothing else on the lane has; a faint
 *  stretch said it with opacity, which is the channel that already
 *  means unverified, and the two were told apart only by guessing
 *  (2026-11-03). DIVE draws its timeline the same way. */
const GAP_LINE_PX = 2;

/**
 * A bar's background over the frames `from` to `to`: the colour where
 * the animal was seen, and a thin line of it through each gap. Two
 * layers of one element's background, hard-stop gradients both, rather
 * than a child per run: the lane re-renders on every playhead move
 * and a drop holds five thousand bars, so extra nodes would triple
 * that work. Gaps outside `[from, to)` are dropped, so a bar cut by
 * the zoom window gets its holes in the right place.
 */
export function barBackground(
  color: string,
  gaps: [number, number][],
  from: number,
  to: number,
): CSSProperties {
  const span = to - from;
  if (gaps.length === 0 || span <= 0) return { background: color };
  const pct = (frame: number) => `${(((frame - from) / span) * 100).toFixed(3)}%`;
  const runs: string[] = [];
  const holes: string[] = [];
  let at = from;
  for (const [a, b] of gaps) {
    const lo = Math.max(a, from);
    const hi = Math.min(b, to);
    if (hi <= lo) continue;
    runs.push(`${color} ${pct(at)} ${pct(lo)}`, `transparent ${pct(lo)} ${pct(hi)}`);
    holes.push(`transparent ${pct(at)} ${pct(lo)}`, `${color} ${pct(lo)} ${pct(hi)}`);
    at = hi;
  }
  if (runs.length === 0) return { background: color };
  runs.push(`${color} ${pct(at)} 100%`);
  holes.push(`transparent ${pct(at)} 100%`);
  return {
    backgroundImage: `linear-gradient(to right, ${runs.join(", ")}), linear-gradient(to right, ${holes.join(", ")})`,
    backgroundSize: `100% 100%, 100% ${GAP_LINE_PX}px`,
    backgroundPosition: "0 0, 0 50%",
    backgroundRepeat: "no-repeat",
  };
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
  const step = samplingStep(
    [...boxesByTrack.values()].flatMap((list) => list.map((d) => d.frame_number!)),
  );
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
      // The group key, shared with the MaxN markers of both surfaces.
      label: speciesKey(rep),
      displayLabel: getDetectionDisplayName(rep),
      color: getDetectionColor(rep),
      verdict,
      startFrame: track.start_frame - Math.max(step, 1) / 2,
      endFrame: track.end_frame + Math.max(step, 1) / 2,
      gaps: trackGaps(boxes.map((d) => d.frame_number!), step),
    });
  }
  // Rejected tracks last: their "False detection" row never pushes a
  // species down the lane.
  return rows.sort(
    (a, b) =>
      Number(a.verdict === "rejected") - Number(b.verdict === "rejected") ||
      a.displayLabel.localeCompare(b.displayLabel) ||
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
 * chips (their species; the number comes from `maxAtOnce`), the
 * viewer's actions with nothing selected, the bulk bar and the Counts
 * modal's landing file all read it.
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
 * The key a box's species is counted and coloured under: the taxonomy
 * id, else the label, else the detector's category, as the Counts page
 * groups them.
 */
export function speciesKey(d: {
  label_taxonomy_id: string | null;
  label: string | null;
  category?: string | null;
}): string {
  return (d.label_taxonomy_id || d.label || d.category || "").toLowerCase();
}

/**
 * The timeline's MaxN markers for a file on its own, one per species
 * with a frame, from the file's `max_at_once`: the backend's count per
 * frame and species at the counting threshold, the same rule the Counts
 * page's MaxN follows, so a tile chip, a marker and a count row say one
 * number. Keyed by `speciesKey`, as the timeline's groups are.
 */
export function fileMaxNMarkers(
  file: FileWithDetections,
): { frame: number; label: string; count: number }[] {
  const markers: { frame: number; label: string; count: number }[] = [];
  for (const entry of file.max_at_once) {
    if (entry.frame_number != null) {
      markers.push({ frame: entry.frame_number, label: speciesKey(entry), count: entry.count });
    }
  }
  return markers;
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

/** The clip's sampling step from its own stored boxes (`samplingStep`);
 *  0 for a clip with fewer than two sampled frames. */
/**
 * How many boxes each sampled frame of a clip draws, by the one rule the
 * player and every still share (`drawsInClip`): the timeline's activity
 * graph reads it, so a peak there is boxes a person would see on that
 * frame. With `species` (a `speciesKey`) only that species' boxes, the
 * per-frame count MaxN is the peak of. Frames with none are absent.
 */
export function frameBoxCounts(
  file: FileWithDetections,
  threshold: number,
  species?: string,
): Map<number, number> {
  const counts = new Map<number, number>();
  if (file.file_type !== "video") return counts;
  const onShow = trackCards(file.tracks ?? [], file.detections, threshold);
  for (const d of file.detections) {
    if (d.frame_number == null || !drawsInClip(d, onShow, threshold)) continue;
    if (species != null && speciesKey(d) !== species) continue;
    counts.set(d.frame_number, (counts.get(d.frame_number) ?? 0) + 1);
  }
  return counts;
}

/**
 * The busiest frame in each of `bins` equal slices of `[from, to)`: the
 * most boxes at once there, the measure MaxN is, so a peak in the graph
 * and a MaxN marker agree, and a one-frame spike is not averaged away.
 */
export function activityPeaks(
  counts: Map<number, number>,
  from: number,
  to: number,
  bins: number,
): number[] {
  const peaks = new Array<number>(bins).fill(0);
  const span = to - from;
  if (span <= 0) return peaks;
  for (const [frame, n] of counts) {
    if (frame < from || frame >= to) continue;
    const bin = Math.min(bins - 1, Math.floor(((frame - from) / span) * bins));
    if (n > peaks[bin]) peaks[bin] = n;
  }
  return peaks;
}

export function fileSamplingStep(file: FileWithDetections): number {
  return samplingStep(
    file.detections.flatMap((d) => (d.frame_number == null ? [] : [d.frame_number])),
  );
}

/**
 * The frame to show for a moment a person picked: the nearest frame the
 * detector sampled. The detector looks at every `step`th frame (every
 * 17th at 50 fps on a 3 fps run), so a moment picked on the ruler, in a
 * lane, by a pan or by pausing the player lands between samples 16
 * times in 17; the bars make claims only about samples, and the still
 * draws boxes by exact frame, so an unsnapped moment showed a bar under
 * the playhead and no box on screen (2026-11-03). Until the ruler this
 * snapped to the nearest box of an animal live at the moment instead,
 * which on an empty stretch jumped minutes to the nearest animal: the
 * frame a person got was not the one they pointed at. A sampled frame
 * with nothing on it is the honest answer to "show me that moment".
 */
export function snapToSample(frame: number, step: number): number {
  return step > 0 ? Math.max(0, Math.round(frame / step) * step) : Math.max(0, Math.round(frame));
}

export function nearestSampledFrame(file: FileWithDetections, frame: number): number {
  return snapToSample(frame, fileSamplingStep(file));
}

/**
 * The sampled frame one step before or after `from`, clamped to the
 * clip: what Shift+Arrow and Shift+wheel show in the Files viewer and
 * the Counts modal alike, so stepping a clip is one thing wherever a
 * clip is looked at.
 */
export function steppedFrame(file: FileWithDetections, from: number, dir: 1 | -1): number {
  const step = Math.max(fileSamplingStep(file), 1);
  const last = Math.max(0, clipDurationFrames(file) - 1);
  return Math.min(last, Math.max(0, snapToSample(from, step) + dir * step));
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
