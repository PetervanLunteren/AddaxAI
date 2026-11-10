import { describe, expect, it } from "vitest";
import { formatShare, shareNote } from "./share";

describe("formatShare", () => {
  it("never rounds a real share down to 0%", () => {
    expect(formatShare(0.001)).toBe("<1%");
    expect(formatShare(0)).toBe("0%");
    expect(formatShare(0.126)).toBe("13%");
  });
});

describe("shareNote", () => {
  it("lists the parts that exist, largest first", () => {
    expect(
      shareNote([
        { label: "animals", count: 6338 },
        { label: "people", count: 6 },
        { label: "vehicles", count: 121 },
        { label: "empty", count: 0 },
      ]),
    ).toBe("98% animals · 2% vehicles · <1% people");
  });

  it("says nothing about an empty total", () => {
    expect(shareNote([{ label: "animals", count: 0 }])).toBe("");
  });
});
