import { describe, expect, it, vi } from "vitest";

vi.mock("../api/projects", () => ({ projectsApi: {} }));

import { hasReprocessChanges } from "./reprocessSettings";

const list = (name: string) => ({
  file_name: "mara.csv",
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
