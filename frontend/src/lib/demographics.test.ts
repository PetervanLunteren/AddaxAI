import { describe, expect, it } from "vitest";
import { SEGMENT_COLORS, toSegments } from "./demographics";
import { BEHAVIOR_OPTIONS, SEX_OPTIONS } from "./observation-attributes";

describe("toSegments", () => {
  it("names known values, puts unknown last in neutral", () => {
    const segments = toSegments(
      [
        { value: null, count: 6 },
        { value: "male", count: 1 },
        { value: "female", count: 3 },
      ],
      SEX_OPTIONS,
    );

    expect(segments.map((s) => [s.label, s.count, s.color])).toEqual([
      ["Female", 3, SEGMENT_COLORS[0]],
      ["Male", 1, SEGMENT_COLORS[1]],
      ["Unknown", 6, null],
    ]);
    expect(segments.reduce((sum, s) => sum + s.share, 0)).toBeCloseTo(1);
  });

  it("folds everything past the third value into Other", () => {
    const segments = toSegments(
      [
        { value: "resting", count: 5 },
        { value: "foraging", count: 4 },
        { value: "traveling", count: 3 },
        { value: "drinking", count: 2 },
        { value: "grooming", count: 1 },
      ],
      BEHAVIOR_OPTIONS,
    );

    expect(segments.map((s) => [s.label, s.count])).toEqual([
      ["Resting", 5],
      ["Foraging", 4],
      ["Travelling", 3],
      ["Other", 3],
    ]);
  });

  it("names a single leftover value instead of calling it Other", () => {
    const segments = toSegments(
      [
        { value: "resting", count: 4 },
        { value: "foraging", count: 3 },
        { value: "traveling", count: 2 },
        { value: "drinking", count: 1 },
      ],
      BEHAVIOR_OPTIONS,
    );

    expect(segments.map((s) => s.label)).toEqual([
      "Resting", "Foraging", "Travelling", "Drinking",
    ]);
  });

  it("returns no segments when nothing was observed", () => {
    expect(toSegments([], SEX_OPTIONS)).toEqual([]);
  });

  it("keeps everything unknown as one neutral segment", () => {
    expect(toSegments([{ value: null, count: 9 }], SEX_OPTIONS)).toEqual([
      { label: "Unknown", count: 9, share: 1, color: null },
    ]);
  });
});
