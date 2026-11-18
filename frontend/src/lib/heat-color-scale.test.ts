import chroma from "chroma-js";
import { describe, expect, it } from "vitest";

import {
  calculateRateDomain,
  getRateColor,
  isRateRamp,
  rampGradient,
  RATE_RAMPS,
} from "./heat-color-scale";

describe("calculateRateDomain", () => {
  it("ends the ramp at the 95th percentile, below an outlier", () => {
    const rates = Array.from({ length: 99 }, (_, i) => i + 1).concat(5000);
    expect(calculateRateDomain(rates)).toEqual({ max: 5000, cap: 96 });
  });

  it("ends the ramp earlier when a view asks for it", () => {
    const rates = Array.from({ length: 100 }, (_, i) => i + 1);
    expect(calculateRateDomain(rates, 0.66)).toEqual({ max: 100, cap: 67 });
  });

  it("uses the maximum for a small set, where 95% is the last value", () => {
    expect(calculateRateDomain([3, 1, 40, 2, 5])).toEqual({ max: 40, cap: 40 });
  });

  it("ignores zeros and gives zeros for nothing to scale", () => {
    expect(calculateRateDomain([0, 0, 7])).toEqual({ max: 7, cap: 7 });
    expect(calculateRateDomain([0, 0])).toEqual({ max: 0, cap: 0 });
    expect(calculateRateDomain([])).toEqual({ max: 0, cap: 0 });
  });
});

describe("getRateColor", () => {
  it("keeps the teal ramp as the default", () => {
    expect(getRateColor(0, 10)).toBe("#f9f871");
    expect(getRateColor(10, 10)).toBe("#0f6064");
  });

  it.each(RATE_RAMPS)("runs the $id ramp from zero to the cap", ({ id, scale }) => {
    const first = scale(0).hex();
    const last = scale(1).hex();
    expect(getRateColor(0, 10, id)).toBe(first);
    expect(getRateColor(10, 10, id)).toBe(last);
    // Above the cap is the last colour too: the legend's "+".
    expect(getRateColor(500, 10, id)).toBe(last);
    const half = getRateColor(5, 10, id);
    expect(half).toBe(scale(0.5).hex());
    expect([first, last]).not.toContain(half);
  });

  it("ends Heat on bright red, where teal ends dark", () => {
    // Why Heat exists: on satellite imagery the highest rates must stand
    // out. ColorBrewer's own last colour is a dark burgundy (#800026).
    expect(getRateColor(10, 10, "heat")).toBe("#e31a1c");
    expect(chroma(getRateColor(10, 10, "heat")).luminance()).toBeGreaterThan(
      chroma(getRateColor(10, 10, "teal")).luminance(),
    );
  });
});

describe("isRateRamp", () => {
  it("accepts the known ramps and nothing else", () => {
    expect(["teal", "heat", "viridis", "rainbow", "thermal"].every(isRateRamp)).toBe(true);
    expect(isRateRamp("sepia")).toBe(false);
    expect(isRateRamp(null)).toBe(false);
  });
});

describe("the classic ramps", () => {
  it("run blue to red, and black to white", () => {
    expect([getRateColor(0, 10, "rainbow"), getRateColor(10, 10, "rainbow")]).toEqual(["#0000ff", "#ff0000"]);
    expect([getRateColor(0, 10, "thermal"), getRateColor(10, 10, "thermal")]).toEqual(["#000000", "#ffffff"]);
  });
});

describe("rampGradient", () => {
  it("runs from the ramp's first to its last colour", () => {
    expect(rampGradient("teal", "to top")).toMatch(
      /^linear-gradient\(to top, #f9f871, .*#0f6064\)$/,
    );
  });
});
