import { describe, expect, it } from "vitest";
import type { LabelsProgressEvent, LabelsProgressPhase } from "../api/labels";
import { overallProgressPct } from "./labels-progress";

const at = (
  phase: LabelsProgressPhase,
  done: number,
  total = 100,
): LabelsProgressEvent => ({ type: "progress", phase, done, total });

describe("overallProgressPct", () => {
  it("gives the similarity sort three equal slices", () => {
    expect(overallProgressPct(at("load", 100), "similarity")).toBe(33);
    expect(overallProgressPct(at("sort", 50), "similarity")).toBe(50);
    expect(overallProgressPct(at("neighbors", 100), "similarity")).toBe(100);
  });

  it("does not jump across the walk suggestions never run", () => {
    // Loading done and neighbours just started are the same moment, so
    // they must be the same percentage (this was 33% then 67%).
    expect(overallProgressPct(at("load", 100), "suggestions")).toBe(50);
    expect(overallProgressPct(at("neighbors", 0), "suggestions")).toBe(50);
    expect(overallProgressPct(at("neighbors", 100), "suggestions")).toBe(100);
  });

  it("lets the event sort's one phase span the whole bar", () => {
    expect(overallProgressPct(at("detections", 25), "events")).toBe(25);
  });

  it("stays at 0 for an empty counter or a phase the sort does not run", () => {
    expect(overallProgressPct(at("load", 0, 0), "suggestions")).toBe(0);
    expect(overallProgressPct(at("sort", 50), "suggestions")).toBe(0);
  });

  it("never passes 100 when a counter overshoots its total", () => {
    expect(overallProgressPct(at("neighbors", 120), "suggestions")).toBe(100);
  });
});
