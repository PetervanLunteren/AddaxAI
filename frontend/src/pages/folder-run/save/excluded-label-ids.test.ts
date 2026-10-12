import { describe, expect, it } from "vitest";

import { excludedLabelIds, type SeparateState } from "./media-options";

const ALL = ["a", "b", "c"];

function withIncluded(includedLabelIds: string[]): SeparateState {
  return {
    enabled: true,
    groupBy: "flat",
    groupEvents: true,
    speciesLast: false,
    copyEmpties: false,
    mediaThreshold: 0.2,
    includedLabelIds,
  };
}

describe("excludedLabelIds", () => {
  it("sends the complement of a real subset", () => {
    expect(excludedLabelIds(withIncluded(["b"]), ALL)).toEqual(["a", "c"]);
  });

  it("sends no filter when everything is picked", () => {
    expect(excludedLabelIds(withIncluded(ALL), ALL)).toEqual([]);
    expect(excludedLabelIds(withIncluded(ALL), ALL, "none")).toEqual([]);
  });

  it("reads an empty selection as all on the folder-run step", () => {
    expect(excludedLabelIds(withIncluded([]), ALL)).toEqual([]);
  });

  it("reads an empty selection as nothing picked in the image copies dialog", () => {
    expect(excludedLabelIds(withIncluded([]), ALL, "none")).toEqual(ALL);
  });
});
