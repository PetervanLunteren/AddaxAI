/**
 * What the video overlay shows at one moment in the clip.
 *
 * The detector samples a video a few frames per second, so a 30 fps clip
 * carries boxes every fifteenth frame and the overlay has real data for
 * one frame in fifteen. This module fills the gaps: it holds every
 * track's sampled boxes in frame order and answers, for any moment,
 * where each animal's box is and where it has just been.
 *
 * It is the single source of truth for that answer. `VideoPlayer` draws
 * it twice, once as SVG on screen and once on a canvas for the annotated
 * MP4 it can record, and those two used to work it out separately: the
 * export carried its own copy of the frame lookup and the fade, and had
 * already drifted (it never dimmed the unselected tracks). Two renderers
 * over one answer is the arrangement; a second place that decides *what*
 * to draw is the thing to keep out.
 *
 * Deliberately pure: no React, no DOM, no imports from components. What
 * a box looks like still belongs to `detection-overlay.ts`, and which
 * boxes exist at all still belongs to `passesDrawFilter`. This module
 * only answers "where, right now".
 */

import { passesDrawFilter } from "./detection-utils";
import type { DetectionResponse } from "../api/types";

/** How much of each track to trail behind the animal. */
export const TRAIL_SECONDS = 3;

/** Opacity factor for the boxes of every other track while one is selected. */
export const OTHER_TRACK_DIM = 0.25;

/**
 * Treat a moment this close to a sampled frame as being on it.
 *
 * Seeking to frame N sets `currentTime = N / frameRate`, and reading it
 * back and multiplying by the rate lands a hair either side of N. A hair
 * *below* used to lose the box: a track whose sample at N follows a gap
 * has nothing to draw at N minus an epsilon, so clicking its bar on the
 * timeline jumped to the animal and showed no box at all. A thousandth
 * of a frame is thirty microseconds, far below anything a viewer or a
 * browser clock can resolve.
 */
const FRAME_EPSILON = 1e-3;

/** A detection with its bbox fields narrowed: the filter guarantees them. */
export type BboxedDetection = DetectionResponse & {
  bbox_x: number;
  bbox_y: number;
  bbox_width: number;
  bbox_height: number;
};

/** One box to draw, in image fractions, at this moment. */
export interface OverlayBox {
  /** The sampled detection this box came from: its label, colour and
   *  pill text are still read from here, so nothing about styling moves
   *  into this module. */
  detection: BboxedDetection;
  x: number;
  y: number;
  width: number;
  height: number;
  /** 1, or `OTHER_TRACK_DIM` when another track is selected. */
  dim: number;
  /** Where this animal has just been, newest first. Every animal on
   *  screen has one: a trail only on the selected animal would say
   *  something about the animal ("this one is tracked, those are not")
   *  when it only means "this is the one you clicked". The selection is
   *  already carried by `dim`. */
  trail: TrailPoint[];
}

/** One point of the trail, in image fractions. `age` runs 0 at the
 *  animal to 1 at the tail, so a renderer can fade by `1 - age`. */
export interface TrailPoint {
  x: number;
  y: number;
  age: number;
}

export interface OverlayFrame {
  boxes: OverlayBox[];
}

export interface OverlayIndex {
  /** Sampled boxes per track, sorted by frame. Untracked boxes get one
   *  group each, keyed by detection id, so they never interpolate. */
  tracks: Map<string, BboxedDetection[]>;
  /** Native frames between two consecutive samples of the clip, measured
   *  from the data. 0 when the file has fewer than two distinct frames,
   *  which means nothing can be interpolated. */
  step: number;
}

const EMPTY: OverlayFrame = { boxes: [] };

/**
 * Group a file's drawable boxes by track and measure the sampling step.
 *
 * The step is measured from the file's own frame numbers rather than
 * taken from `Project.video_fps`, because a clip analysed before that
 * setting was last changed keeps its own older spacing and the file's
 * numbers cannot disagree with the file.
 */
export function buildOverlayIndex(
  detections: DetectionResponse[],
  detectionThreshold: number,
): OverlayIndex {
  const tracks = new Map<string, BboxedDetection[]>();
  const frames = new Set<number>();

  for (const d of detections) {
    if (d.frame_number == null) continue;
    if (!passesDrawFilter(d, detectionThreshold)) continue;
    const box = d as BboxedDetection;
    frames.add(d.frame_number);
    // An untracked box is its own group, so the bracketing below can
    // only ever find itself and it never glides anywhere.
    const key = d.track_id ?? `untracked:${d.id}`;
    const group = tracks.get(key);
    if (group) group.push(box);
    else tracks.set(key, [box]);
  }

  for (const group of tracks.values()) {
    group.sort((a, b) => a.frame_number! - b.frame_number!);
  }

  const sorted = [...frames].sort((a, b) => a - b);
  let step = 0;
  for (let i = 1; i < sorted.length; i++) {
    const gap = sorted[i] - sorted[i - 1];
    if (gap > 0 && (step === 0 || gap < step)) step = gap;
  }

  return { tracks, step };
}

/** The last sample at or before `frame`, by binary search. */
function sampleAtOrBefore(group: BboxedDetection[], frame: number): number {
  let lo = 0;
  let hi = group.length - 1;
  let found = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (group[mid].frame_number! <= frame) {
      found = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return found;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/**
 * Where every animal's box is at `frame`, and where it has just been.
 *
 * `frame` is fractional: the overlay runs off `video.currentTime`, not
 * off a rounded frame index, or the box would step rather than glide.
 *
 * The rule, one sentence for every case: **a sampled box is drawn for
 * one sampling step starting at its own frame, gliding to the next
 * sample when that sample is one step away and holding its position
 * when it is not.**
 *
 * So a box is never drawn before its track's first sample or more than
 * one step past its last, an animal never glides across a gap the
 * detector could not see through, and the box at the end of a track
 * stops rather than drifting or fading. There is no hold and no fade:
 * both existed to cover for boxes that vanished between samples, which
 * is the thing this replaces.
 */
export function overlayAt(
  index: OverlayIndex,
  frame: number,
  frameRate: number,
  selectedTrackId?: string | null,
): OverlayFrame {
  if (index.tracks.size === 0) return EMPTY;
  // With one sampled frame in the whole file there is no spacing to
  // interpolate over; show each box on its own frame and nothing more.
  const step = index.step > 0 ? index.step : 1;

  const boxes: OverlayBox[] = [];
  const trailFrames = TRAIL_SECONDS * (frameRate > 0 ? frameRate : 30);

  for (const group of index.tracks.values()) {
    const i = sampleAtOrBefore(group, frame + FRAME_EPSILON);
    if (i < 0) continue; // before this track's first sample
    const from = group[i];
    const elapsed = Math.max(0, frame - from.frame_number!);
    if (elapsed >= step) continue; // more than one step past a sample

    const next = group[i + 1];
    const adjacent = next != null && next.frame_number! - from.frame_number! <= step;
    const t = adjacent ? elapsed / (next.frame_number! - from.frame_number!) : 0;

    const box: OverlayBox = {
      detection: from,
      x: adjacent ? lerp(from.bbox_x, next.bbox_x, t) : from.bbox_x,
      y: adjacent ? lerp(from.bbox_y, next.bbox_y, t) : from.bbox_y,
      width: adjacent ? lerp(from.bbox_width, next.bbox_width, t) : from.bbox_width,
      height: adjacent ? lerp(from.bbox_height, next.bbox_height, t) : from.bbox_height,
      dim: selectedTrackId != null && from.track_id !== selectedTrackId ? OTHER_TRACK_DIM : 1,
      trail: [],
    };
    box.trail = trailFor(group, frame, box, trailFrames);
    boxes.push(box);
  }

  return { boxes };
}

/**
 * One animal's path over the last `TRAIL_SECONDS`, newest first,
 * starting at the box on screen so the line meets the animal.
 *
 * The window arrives in frames rather than seconds because the caller
 * already holds the clip's frame rate and this module stays free of
 * state of its own.
 */
function trailFor(
  group: BboxedDetection[],
  frame: number,
  head: OverlayBox,
  windowFrames: number,
): TrailPoint[] {
  const span = windowFrames;
  const oldest = frame - span;
  const points: TrailPoint[] = [
    { x: head.x + head.width / 2, y: head.y + head.height / 2, age: 0 },
  ];
  for (let i = group.length - 1; i >= 0; i--) {
    const f = group[i].frame_number!;
    // Strictly before: sitting exactly on a sample, the head above
    // already is that sample and a second point would stack on it.
    if (f >= frame) continue;
    if (f < oldest) break;
    points.push({
      x: group[i].bbox_x + group[i].bbox_width / 2,
      y: group[i].bbox_y + group[i].bbox_height / 2,
      age: Math.min(1, (frame - f) / span),
    });
  }
  return points.length > 1 ? points : [];
}
