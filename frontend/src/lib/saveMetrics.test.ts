import { describe, expect, it } from "vitest";

import type { SaveResults, StatSnapshot } from "../components/projects/SaveResultsModal";
import { hasChanges, metricsPhrase } from "./saveMetrics";

const snap = (total: number, labels: [string, number][]): StatSnapshot => ({
  total,
  labels: labels.map(([label, count]) => ({ label, count })),
});

const results = (
  labels: [StatSnapshot, StatSnapshot],
  counts: [StatSnapshot, StatSnapshot],
): SaveResults => ({
  observations: { before: labels[0], after: labels[1] },
  independent_observations: { before: counts[0], after: counts[1] },
});

const same = snap(3, [["fox", 2], ["deer", 1]]);

describe("hasChanges", () => {
  it("is false when every number stayed", () => {
    expect(hasChanges(results([same, same], [same, same]), ["labels", "counts"])).toBe(false);
  });

  it("sees a label that moved, even with the same total", () => {
    const moved = snap(3, [["fox", 1], ["deer", 2]]);
    expect(hasChanges(results([same, moved], [same, same]), ["labels"])).toBe(true);
  });

  it("sees a label that appeared or disappeared", () => {
    const gone = snap(2, [["fox", 2]]);
    expect(hasChanges(results([same, gone], [same, same]), ["labels"])).toBe(true);
  });

  it("only looks at the cards that are shown", () => {
    const moved = snap(4, [["fox", 3], ["deer", 1]]);
    const r = results([same, same], [same, moved]);
    expect(hasChanges(r, ["labels"])).toBe(false);
    expect(hasChanges(r, ["labels", "counts"])).toBe(true);
  });
});

describe("metricsPhrase", () => {
  it("names the shown cards", () => {
    expect(metricsPhrase(["labels", "counts"])).toBe("labels and counts");
    expect(metricsPhrase(["labels"])).toBe("labels");
  });
});
