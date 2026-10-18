/**
 * A leaf the page's filters leave nothing of stays in the label tree,
 * muted, but cannot be ticked. These are the leaves bulk actions and
 * "everything ticked means no filter" must leave out.
 */

import { describe, expect, it } from "vitest";
import type { TaxonomyNode } from "../api/types";
import { collectLiveLeafIds, isDeadLeaf } from "./taxonomy-utils";

const leaf = (id: string, count?: number): TaxonomyNode => ({
  id, name: id, level: 1, children: [], selected: true, count,
});
const tree: TaxonomyNode[] = [
  {
    id: "genus", name: "Genus", level: 0, selected: true,
    children: [leaf("fox", 3), leaf("roo", 0), leaf("nocount")],
  },
];

describe("dead leaves", () => {
  it("are the leaves with a count of zero, never a parent or an uncounted leaf", () => {
    expect(isDeadLeaf(leaf("roo", 0))).toBe(true);
    expect(isDeadLeaf(leaf("fox", 3))).toBe(false);
    expect(isDeadLeaf(leaf("nocount"))).toBe(false);
    expect(isDeadLeaf(tree[0])).toBe(false);
  });

  it("are left out of the live leaf set", () => {
    expect([...collectLiveLeafIds(tree)].sort()).toEqual(["fox", "nocount"]);
  });
});
