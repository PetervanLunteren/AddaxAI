import { describe, expect, it } from "vitest";
import type { CalibrationLabel } from "../api/calibration";
import { curveSummary } from "./calibration-summary";

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
    no_error_bound: null,
  };
}

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
