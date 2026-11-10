/**
 * Issue #126: Oslo on June 21 has sunset 22:43 and civil dusk 00:30,
 * sent as dusk = 24.5. A plain `start <= h < end` missed the part of
 * the band after midnight.
 */

import { describe, expect, it } from "vitest";
import { inSunBand, overlapsSunBand } from "./sun-bands";

const SUNSET = 22.72;
const DUSK = 24.5;

describe("inSunBand", () => {
  it("includes the part of a band after midnight", () => {
    expect(inSunBand(23, SUNSET, DUSK)).toBe(true);
    expect(inSunBand(0.25, SUNSET, DUSK)).toBe(true);
    expect(inSunBand(0.5, SUNSET, DUSK)).toBe(false);
    expect(inSunBand(12, SUNSET, DUSK)).toBe(false);
  });
});

describe("overlapsSunBand", () => {
  it("colours the clock bars either side of midnight as twilight", () => {
    const bar = (hour: number) => overlapsSunBand(hour, hour + 1, SUNSET, DUSK);
    expect([21, 22, 23, 0, 1].map(bar)).toEqual([false, true, true, true, false]);
  });
});
