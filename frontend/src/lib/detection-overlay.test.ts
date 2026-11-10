/**
 * Box labels must never ask for a named font.
 *
 * Every label pill used to name Arial (and Konva's own default is Arial).
 * On a Mac with a broken or duplicate Arial the browser still finds the
 * font, so it never falls back, and every label was drawn as boxed "?"
 * while the rest of the app looked fine (reported 2026-11-04). A render
 * test cannot see this: jsdom has no fonts, and the bug only shows on a
 * machine with a broken Arial. So the guard is on the source: the
 * overlay font lives in OVERLAY_FONT_FAMILY, and nothing names Arial.
 */

import { describe, expect, it } from "vitest";
import { OVERLAY_FONT_FAMILY } from "./detection-overlay";

const sources = import.meta.glob<string>("/src/**/*.{ts,tsx}", {
  query: "?raw",
  import: "default",
  eager: true,
});

describe("overlay font", () => {
  it("is the system UI font, not a named one", () => {
    expect(OVERLAY_FONT_FAMILY.split(",")[0].trim()).toBe("system-ui");
    expect(OVERLAY_FONT_FAMILY).not.toMatch(/arial/i);
  });

  it("is named nowhere else in the source", () => {
    const self = "/src/lib/detection-overlay.test.ts";
    const offenders = Object.entries(sources)
      .filter(([path, text]) => path !== self && /\barial\b/i.test(text))
      .map(([path]) => path);
    expect(Object.keys(sources).length).toBeGreaterThan(50);
    expect(offenders).toEqual([]);
  });
});
