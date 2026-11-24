/**
 * The Labels progress pill while its count is out. On a big project
 * `labels/progress` takes 20 s or more, and the pill used to read
 * "0% verified" all that time, for a project that may be half done.
 */

import { describe, expect, it } from "vitest";

import type { LabelsProgress } from "../api/types";
import { labelsProgressValue } from "./verified-progress";

const counts: LabelsProgress = {
  total_labels: 200,
  verified_labels: 50,
  crop_labels: 150,
  crop_labels_verified: 40,
  empty_labels: 50,
  empty_labels_verified: 10,
  files: 120,
  files_verified: 30,
};

describe("labelsProgressValue", () => {
  it("has no percentage until the count is back", () => {
    const value = labelsProgressValue(undefined);
    expect(value.pct).toBeNull();
    expect(value.title).toBe("Counting the labels to verify");
    // A count of 0 hides the tab chips, so they need no special case.
    expect(value.cropsLeft).toBe(0);
    expect(value.filesLeft).toBe(0);
  });

  it("gives the share verified once the count is back", () => {
    const value = labelsProgressValue(counts);
    expect(value.pct).toBe(25);
    expect(value.cropsLeft).toBe(110);
    expect(value.filesLeft).toBe(90);
    expect(value.title).toMatch(/^50 of 200 labels verified/);
  });

  it("reads 0%, not counting, for a project with nothing to verify", () => {
    const value = labelsProgressValue({ ...counts, total_labels: 0, verified_labels: 0 });
    expect(value.pct).toBe(0);
  });
});
