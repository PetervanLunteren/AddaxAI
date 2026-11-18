import { describe, expect, it } from "vitest";
import type { CalibrationLabel } from "../api/calibration";
import { curveSummary, neededForCurve, statusLine } from "./calibration-summary";

function row(scores: (number | null)[], right = 80, wrong = 21): CalibrationLabel {
  return {
    label: "ursus_americanus",
    common_name: "American black bear",
    scientific_name: "Ursus americanus",
    detections: 3000,
    right,
    wrong,
    unsure: 0,
    right_needed: 0,
    wrong_needed: 0,
    thresholds: [0.9, 0.95, 0.99].map((target, i) => ({ target, score: scores[i] })),
    curve: [],
    lower_bound: null,
  };
}

/** A label without a curve: the answers so far and the bound the backend sent. */
function started(
  right: number,
  wrong: number,
  lower_bound: number | null,
  unsure = 0,
): CalibrationLabel {
  return {
    ...row([]),
    right,
    wrong,
    unsure,
    right_needed: Math.max(0, 10 - right),
    wrong_needed: Math.max(0, 10 - wrong),
    thresholds: null,
    curve: null,
    lower_bound,
  };
}

describe("statusLine", () => {
  it("gives the share right, its lower bound and what a curve still needs", () => {
    expect(statusLine(started(155, 5, 0.9354))).toBe(
      "155 of 160 right: at least 93% right. 5 more wrong needed for a curve",
    );
  });

  it("reads the same without a wrong answer", () => {
    expect(statusLine(started(60, 0, 0.9513))).toBe(
      "60 of 60 right: at least 95% right. 10 more wrong needed for a curve",
    );
  });

  it("gives a mostly wrong label its share without a bound", () => {
    expect(statusLine(started(0, 132, null))).toBe(
      "0 of 132 right. 10 more right needed for a curve",
    );
  });

  it("names both when right and wrong are short", () => {
    expect(statusLine(started(0, 0, null, 4))).toBe(
      "10 more right and 10 more wrong needed for a curve",
    );
  });

  it("is absent with a curve and before the first answer", () => {
    expect(statusLine(row([0.6, 0.7, 0.8]))).toBeNull();
    expect(statusLine(started(0, 0, null))).toBeNull();
  });
});

describe("neededForCurve", () => {
  it("is null once the minimum is reached", () => {
    expect(neededForCurve(started(12, 10, null))).toBeNull();
  });
});

describe("curveSummary", () => {
  it("names where the curve passes each target and which it misses", () => {
    expect(curveSummary(row([0.97, 0.99, null]), "Black bear")).toBe(
      "Reading from left to right, the share of right Black bear labels rises with the score. " +
        "It passes 90% at 0.97 and 95% at 0.99, and does not reach 99% within the scores " +
        "you answered. On the Labels page, the classification confidence slider under More " +
        "filters selects the Black bear labels below or above a score. Based on 101 answers. " +
        "The shaded band is the 95% uncertainty.",
    );
  });

  it("lists all three when every target is reached", () => {
    expect(curveSummary(row([0.6, 0.7, 0.8]), "Wolf")).toContain(
      "It passes 90% at 0.60, 95% at 0.70 and 99% at 0.80.",
    );
  });

  it("says plainly when no target is reached", () => {
    expect(curveSummary(row([null, null, null]), "Wolf")).toContain(
      "Within the scores you answered, fewer than 90% of the Wolf labels are right at " +
        "every score, so no score reaches 90%, 95% and 99%.",
    );
  });

  it("is empty without a curve", () => {
    expect(curveSummary({ ...row([]), thresholds: null }, "Wolf")).toBe("");
  });
});
