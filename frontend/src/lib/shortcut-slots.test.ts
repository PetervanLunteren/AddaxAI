import { describe, expect, it } from "vitest";

import { withoutSlot } from "./shortcut-slots";

describe("withoutSlot", () => {
  it("removes one key and leaves the others on their numbers", () => {
    expect(withoutSlot({ 1: "moose", 2: "elk", 3: "giraffe" }, 2)).toEqual({
      1: "moose",
      3: "giraffe",
    });
  });

  it("does not change the slots it was given", () => {
    const labels = { 1: "moose" };
    expect(withoutSlot(labels, 1)).toEqual({});
    expect(labels).toEqual({ 1: "moose" });
  });
});
