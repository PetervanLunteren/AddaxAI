/**
 * A leaf the page's filters leave nothing of stays in the label tree,
 * muted, but cannot be ticked. These are the leaves bulk actions and
 * "everything ticked means no filter" must leave out.
 */

import { describe, expect, it } from "vitest";
import type { TaxonomyNode } from "../api/types";
import { collectLiveLeafIds, isDeadLeaf, withCounts } from "./taxonomy-utils";

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

describe("withCounts", () => {
  const tree: TaxonomyNode[] = [
    leaf("fox"),
    {
      id: "family:canidae", name: "Canidae", level: 1, selected: true,
      children: [leaf("wolf"), leaf("jackal")],
    },
  ];

  it("puts each leaf's count on it and sums the parents", () => {
    const counted = withCounts(tree, { fox: 2, wolf: 3, jackal: 1 });
    expect(counted[0].count).toBe(2);
    expect(counted[1].count).toBe(4);
    expect(counted[1].children.map((c) => c.count)).toEqual([3, 1]);
  });

  it("reads a label the counts leave out as 0, so it is dead", () => {
    const counted = withCounts(tree, { wolf: 3 });
    expect(counted[0].count).toBe(0);
    expect(isDeadLeaf(counted[0])).toBe(true);
    expect(collectLiveLeafIds(counted)).toEqual(new Set(["wolf"]));
  });

  it("leaves the tree it was given as it was", () => {
    withCounts(tree, { fox: 2, wolf: 3, jackal: 1 });
    expect(tree[0].count).toBeUndefined();
    expect(tree[1].count).toBeUndefined();
  });
});
