import { describe, expect, it } from "vitest";
import { keepOnScreen, selectOnClick } from "./grid-selection";

const plain = { shiftKey: false, metaKey: false, ctrlKey: false };
const shift = { ...plain, shiftKey: true };

describe("selectOnClick", () => {
  it("adds the range from the anchor on shift-click and keeps the anchor", () => {
    const r = selectOnClick(["a", "b", "c", "d"], "b", "d", shift, new Set(["b"]));
    expect([...r.ids].sort()).toEqual(["b", "c", "d"]);
    expect(r.anchor).toBe("b");
  });

  it("treats a shift-click whose anchor left the list as a plain click", () => {
    // Selected on page 1, shift-clicked on page 2.
    const r = selectOnClick(["b1", "b2"], "a2", "b2", shift, new Set(["a1", "a2"]));
    expect([...r.ids]).toEqual(["b2"]);
    expect(r.anchor).toBe("b2");
  });
});

describe("keepOnScreen", () => {
  it("drops the ids that left the grid", () => {
    const kept = keepOnScreen(["b", "c"], new Set(["a", "b"]));
    expect([...kept]).toEqual(["b"]);
  });

  it("returns the same set when nothing left, so state does not change", () => {
    const current = new Set(["a", "b"]);
    expect(keepOnScreen(["a", "b", "c"], current)).toBe(current);
  });
});
