/**
 * A tester saw "Trap nights: 1" on a 923K-file project and every bar
 * read as count x 100 with no way to see the count itself. The tooltip
 * now carries the plain count, and a project with no effort at all says
 * so instead of pretending a rate.
 */

import { describe, expect, it } from "vitest";
import {
  describeRatePer100,
  formatRate,
  getRateUnit,
  isRateUnit,
  parseRateUnit,
  rateFromPer100,
  rateLabel,
  rateValue,
  wildlifeAxisTitle,
  wildlifeBarValue,
  wildlifeTooltipLabel,
  type RateUnit,
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

  it("does so in every rate unit", () => {
    for (const unit of UNITS) {
      expect(wildlifeBarValue(45, 0, unit)).toBe(45);
      expect(wildlifeTooltipLabel("max_n", 1, 0, unit)).toBe("1 individual");
      expect(wildlifeAxisTitle("events", 0, unit)).toBe("Independent events");
    }
  });
});

/*
 * DOC New Zealand reports deer per week, others per day or per month.
 * They are the same ratio with another multiplier, so the unit is a
 * display choice and these pin that every unit scales and reads right.
 */
const UNITS: RateUnit[] = ["night", "week", "month", "hundred"];

describe("rate units", () => {
  it("scales one count and effort into each unit", () => {
    expect(UNITS.map((u) => rateValue(14, 200, u))).toEqual([0.07, 0.49, 2.1, 7]);
  });

  it("scales a rate the backend already gave per 100 the same way", () => {
    expect(UNITS.map((u) => rateFromPer100(7, u))).toEqual([0.07, 0.49, 2.1, 7]);
  });

  it("keeps a rare species above zero per trap night", () => {
    expect(rateValue(1, 2500, "night")).toBe(0.0004);
    expect(rateFromPer100(0.04, "night")).toBe(0.0004);
  });

  it("writes a rate without trailing zeros and with separators", () => {
    expect(formatRate(7, "hundred")).toBe("7");
    expect(formatRate(0.07, "night")).toBe("0.07");
    expect(formatRate(0.0004, "night")).toBe("0.0004");
    expect(formatRate(4500, "hundred")).toBe("4,500");
    expect(formatRate(12.345678, "week")).toBe("12.346");
  });

  it("names each unit", () => {
    expect(UNITS.map((u) => rateLabel(u))).toEqual([
      "per trap night",
      "per 7 trap nights",
      "per 30 trap nights",
      "per 100 trap nights",
    ]);
  });

  it("writes a backend rate out with its unit", () => {
    expect(describeRatePer100(7, "week")).toBe("0.49 per 7 trap nights");
    expect(describeRatePer100(12.5, "hundred")).toBe("12.5 per 100 trap nights");
    expect(describeRatePer100(0, "night")).toBe("0 per trap night");
  });

  it("carries the unit into the wildlife chart's axis and tooltip", () => {
    expect(wildlifeBarValue(14, 200, "week")).toBe(0.49);
    expect(wildlifeAxisTitle("max_n", 200, "week")).toBe(
      "Observations per 7 trap nights",
    );
    expect(wildlifeTooltipLabel("events", 14, 200, "night")).toBe(
      "0.07 per trap night (14 events over 200 trap nights)",
    );
  });

  it("reads anything that is not a unit as per 100 trap nights", () => {
    for (const stored of [null, undefined, "", "weekly", "100", 7]) {
      expect(parseRateUnit(stored)).toBe("hundred");
    }
    expect(parseRateUnit("week")).toBe("week");
    // A key every object has is not a unit.
    expect(isRateUnit("toString")).toBe(false);
  });

  it("starts on per 100 trap nights when nothing is stored", () => {
    expect(getRateUnit()).toBe("hundred");
  });
});

