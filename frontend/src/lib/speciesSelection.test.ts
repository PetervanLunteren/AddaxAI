import { describe, expect, it } from "vitest";
import { clearSpeciesSelection } from "./speciesSelection";

describe("clearSpeciesSelection", () => {
  it("clears country, state and exclusions together, as dirty edits", () => {
    const calls: unknown[][] = [];
    const setValue = (...args: unknown[]) => {
      calls.push(args);
    };

    clearSpeciesSelection(setValue as never);

    expect(calls).toEqual([
      ["country_code", null, { shouldDirty: true }],
      ["state_code", null, { shouldDirty: true }],
      ["excluded_classes", [], { shouldDirty: true }],
    ]);
  });
});
