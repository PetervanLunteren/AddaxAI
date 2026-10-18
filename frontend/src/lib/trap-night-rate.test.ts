/**
 * A tester saw "Trap nights: 1" on a 923K-file project and every bar
 * read as count x 100 with no way to see the count itself. The tooltip
 * now carries the plain count, and a project with no effort at all says
 * so instead of pretending a rate.
 */

import { describe, expect, it } from "vitest";
import {
  wildlifeAxisTitle,
  wildlifeBarValue,
  wildlifeTooltipLabel,
} from "./trap-night-rate";

describe("with trap nights", () => {
  it("draws the rate and names the count it came from", () => {
    expect(wildlifeBarValue(45, 365)).toBe(12.33);
    expect(wildlifeTooltipLabel("events", 45, 365)).toBe(
      "12.33 per 100 trap nights (45 events over 365 trap nights)",
    );
    expect(wildlifeTooltipLabel("max_n", 1212, 365)).toBe(
      "332.05 per 100 trap nights (1,212 individuals over 365 trap nights)",
    );
    expect(wildlifeTooltipLabel("events", 45, 1)).toBe(
      "4,500 per 100 trap nights (45 events over 1 trap night)",
    );
    expect(wildlifeTooltipLabel("max_n", 1, 84)).toBe(
      "1.19 per 100 trap nights (1 individual over 84 trap nights)",
    );
    expect(wildlifeTooltipLabel("events", 1, 84)).toBe(
      "1.19 per 100 trap nights (1 event over 84 trap nights)",
    );
    expect(wildlifeAxisTitle("events", 365)).toBe(
      "Independent events per 100 trap nights",
    );
    expect(wildlifeAxisTitle("max_n", 365)).toBe(
      "Observations per 100 trap nights",
    );
  });
});

describe("without trap nights", () => {
  it("shows the plain count and does not call it a rate", () => {
    expect(wildlifeBarValue(45, 0)).toBe(45);
    expect(wildlifeTooltipLabel("events", 45, 0)).toBe("45 events");
    expect(wildlifeAxisTitle("events", 0)).toBe("Independent events");
    expect(wildlifeAxisTitle("max_n", 0)).toBe("Observations");
  });
});
