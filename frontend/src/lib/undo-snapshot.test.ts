import { describe, expect, it } from "vitest";
import type { DetectionState, FileState } from "../api/types";
import { emptySnapshot, isEmptySnapshot, mergeSnapshots } from "./undo-snapshot";

function det(id: string, verified = false): DetectionState {
  return {
    detection_id: id,
    category: "animal",
    label: null,
    label_confidence: null,
    scientific_name: null,
    common_name: null,
    classification_method: null,
    verified,
    verified_at_utc: null,
  };
}

function file(id: string, verified: boolean): FileState {
  return { file_id: id, verified, verified_at_utc: null };
}

describe("mergeSnapshots", () => {
  it("concatenates disjoint detections in chunk order", () => {
    const merged = mergeSnapshots([
      { detections: [det("a"), det("b")], files: [] },
      { detections: [det("c")], files: [] },
    ]);
    expect(merged.detections.map((d) => d.detection_id)).toEqual(["a", "b", "c"]);
  });

  it("keeps the first state of a file two chunks both touched", () => {
    // Chunk 1 saw the file unverified; by the time chunk 2 ran, chunk 1
    // had already flipped it. Undo must go back to the first.
    const merged = mergeSnapshots([
      { detections: [det("a")], files: [file("f", false)] },
      { detections: [det("b")], files: [file("f", true)] },
    ]);
    expect(merged.files).toEqual([file("f", false)]);
  });

  it("drops a repeated detection the same way", () => {
    const merged = mergeSnapshots([
      { detections: [det("a", false)], files: [] },
      { detections: [det("a", true)], files: [] },
    ]);
    expect(merged.detections).toEqual([det("a", false)]);
  });

  it("merges nothing into an empty snapshot", () => {
    expect(mergeSnapshots([])).toEqual(emptySnapshot());
    expect(isEmptySnapshot(mergeSnapshots([emptySnapshot()]))).toBe(true);
  });
});
