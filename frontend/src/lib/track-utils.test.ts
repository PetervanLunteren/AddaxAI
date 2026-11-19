/**
 * "Hold to show all stored boxes": `hiddenFrameBoxes` returns exactly
 * the stored boxes of the frame on screen that `frameBoxes` leaves out,
 * so the dashed boxes and the real ones never overlap and never drift.
 */

import { describe, expect, it } from "vitest";

import type {
  DetectionResponse,
  FileWithDetections,
  TrackResponse,
} from "../api/types";
import { frameBoxes, hiddenFrameBoxes } from "./track-utils";

const THRESHOLD = 0.5;

function box(id: string, over: Partial<DetectionResponse> = {}): DetectionResponse {
  return {
    id,
    file_id: "f",
    category: "animal",
    confidence: 0.9,
    job_id: "job",
    bbox_x: 0.1,
    bbox_y: 0.1,
    bbox_width: 0.2,
    bbox_height: 0.2,
    label: null,
    label_confidence: null,
    common_name: null,
    scientific_name: null,
    label_taxonomy_id: null,
    classification_method: null,
    frame_number: null,
    verified: false,
    verified_at_utc: null,
    track_id: null,
    ...over,
  };
}

function track(id: string, cardFrame: number): TrackResponse {
  return {
    id,
    track_key: 1,
    start_frame: 0,
    end_frame: 30,
    frame_count: 3,
    max_confidence: 0.9,
    representative_frame_number: cardFrame,
  };
}

function file(
  detections: DetectionResponse[],
  over: Partial<FileWithDetections> = {},
): FileWithDetections {
  return {
    file_type: "image",
    best_frame_number: null,
    detections,
    tracks: [],
    ...over,
  } as FileWithDetections;
}

const ids = (boxes: { id: string }[]) => boxes.map((b) => b.id).sort();

describe("hiddenFrameBoxes on a photo", () => {
  const photo = file([
    box("strong"),
    box("weak", { confidence: 0.12 }),
    box("rejected", { label: "false detection", verified: true }),
    box("verified-weak", { confidence: 0.1, verified: true, label: "fox" }),
    box("no-geometry", {
      confidence: 0.1,
      bbox_x: null,
      bbox_y: null,
      bbox_width: null,
      bbox_height: null,
    }),
  ]);

  it("shows what the threshold and a false verdict hide", () => {
    expect(ids(hiddenFrameBoxes(photo, THRESHOLD))).toEqual(["rejected", "weak"]);
  });

  it("leaves out what is already drawn and what has no box", () => {
    const hidden = ids(hiddenFrameBoxes(photo, THRESHOLD));
    expect(hidden).not.toContain("strong");
    expect(hidden).not.toContain("verified-weak");
    expect(hidden).not.toContain("no-geometry");
  });

  it("follows the threshold", () => {
    expect(ids(hiddenFrameBoxes(photo, 0.1))).toEqual(["rejected"]);
  });
});

describe("hiddenFrameBoxes on a clip", () => {
  // Track "on" has its card at 0.9 on frame 10, so it is on show and
  // its weak frame draws too. Track "off" has a 0.2 card on frame 10,
  // so none of its boxes draw. An untracked box is judged on its own.
  const clip = file(
    [
      box("on-10", { frame_number: 10, track_id: "on" }),
      box("on-20-weak", { frame_number: 20, track_id: "on", confidence: 0.1 }),
      box("off-10", { frame_number: 10, track_id: "off", confidence: 0.2 }),
      box("off-20", { frame_number: 20, track_id: "off", confidence: 0.15 }),
      box("loose-10", { frame_number: 10, confidence: 0.05 }),
    ],
    {
      file_type: "video",
      best_frame_number: 10,
      tracks: [track("on", 10), track("off", 10)],
    },
  );

  it("shows only the frame on screen, the cover by default", () => {
    expect(ids(hiddenFrameBoxes(clip, THRESHOLD))).toEqual(["loose-10", "off-10"]);
    expect(ids(hiddenFrameBoxes(clip, THRESHOLD, 20))).toEqual(["off-20"]);
  });

  it("never shows a weak frame of a track that is on show, which already draws", () => {
    expect(ids(frameBoxes(clip, THRESHOLD, 20))).toEqual(["on-20-weak"]);
    expect(ids(hiddenFrameBoxes(clip, THRESHOLD, 20))).not.toContain("on-20-weak");
  });
});

describe("frameBoxes and hiddenFrameBoxes", () => {
  it("split the boxes of a frame with no overlap", () => {
    const photo = file([
      box("a"),
      box("b", { confidence: 0.2 }),
      box("c", { confidence: 0.7, label: "false detection", verified: true }),
    ]);
    const drawn = ids(frameBoxes(photo, THRESHOLD));
    const hidden = ids(hiddenFrameBoxes(photo, THRESHOLD));
    expect(drawn.filter((id) => hidden.includes(id))).toEqual([]);
    expect([...drawn, ...hidden].sort()).toEqual(["a", "b", "c"]);
  });
});
