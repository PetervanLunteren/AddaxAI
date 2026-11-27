import { describe, expect, it, vi } from "vitest";

vi.mock("../api/projects", () => ({ projectsApi: {} }));

import { hasReprocessChanges, hasSettingsChanges } from "./reprocessSettings";

const list = (name: string) => ({
  file_name: "mara.csv",
  text: "",
  rows: [{ latin: "aepyceros", label: name }],
});

describe("hasReprocessChanges", () => {
  it("reads a reloaded copy of the same taxa list as no change", () => {
    expect(hasReprocessChanges({ taxa_mapping: list("impala") }, { taxa_mapping: list("impala") })).toBe(false);
  });

  it("reads a different, a new or a removed taxa list as a change", () => {
    expect(hasReprocessChanges({ taxa_mapping: list("impala") }, { taxa_mapping: list("antelope") })).toBe(true);
    expect(hasReprocessChanges({ taxa_mapping: null }, { taxa_mapping: list("impala") })).toBe(true);
    expect(hasReprocessChanges({ taxa_mapping: list("impala") }, { taxa_mapping: null })).toBe(true);
  });
});

describe("hasSettingsChanges", () => {
  const saved = { sequence_smoothing: true, counting_threshold: 0.5, independence_interval: 1800 };

  it("reads a new independence interval as a change to save, without a reprocess", () => {
    const edited = { ...saved, independence_interval: 300 };
    expect(hasSettingsChanges(saved, edited)).toBe(true);
    expect(hasReprocessChanges(saved, edited)).toBe(false);
  });

  it("reads a new threshold as a change to save, without a reprocess", () => {
    const edited = { ...saved, counting_threshold: 0.6 };
    expect(hasSettingsChanges(saved, edited)).toBe(true);
    expect(hasReprocessChanges(saved, edited)).toBe(false);
  });

  it("reads a reprocess-triggering change as a change, and no change as none", () => {
    expect(hasSettingsChanges(saved, { ...saved, sequence_smoothing: false })).toBe(true);
    expect(hasSettingsChanges(saved, { ...saved })).toBe(false);
  });
});
