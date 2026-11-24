/**
 * Pure utility functions for taxonomy tree operations.
 *
 * Used by both the settings label selector (full tree, exclusion mode)
 * and the verify filter modal (pruned tree, inclusion mode).
 */

import type { TaxonomyNode } from "../api/types";

/**
 * Prune a taxonomy tree to only contain branches leading to leaves in `keepSet`.
 *
 * - Leaf nodes are kept only if their `id` is in `keepSet`
 * - Parent nodes are kept only if they have surviving children
 * - If `eventCounts` is provided, sets `count` on leaf nodes
 * - Returns a new tree (no mutation)
 */
export function pruneTaxonomyTree(
  tree: TaxonomyNode[],
  keepSet: Set<string>,
  eventCounts?: Map<string, number>
): TaxonomyNode[] {
  const prune = (nodes: TaxonomyNode[]): TaxonomyNode[] => {
    const result: TaxonomyNode[] = [];
    for (const node of nodes) {
      if (!node.children || node.children.length === 0) {
        // Leaf: keep only if in the set
        if (keepSet.has(node.id)) {
          const count = eventCounts?.get(node.id);
          result.push(count != null ? { ...node, count } : { ...node });
        }
      } else {
        // Parent: recurse, keep if any children survive
        const prunedChildren = prune(node.children);
        if (prunedChildren.length > 0) {
          result.push({ ...node, children: prunedChildren });
        }
      }
    }
    return result;
  };
  return prune(tree);
}

/**
 * Put a page's label counts on a label tree that has none.
 *
 * Each leaf gets its count (0 when the counts do not mention it: the
 * page's filters leave nothing of that label), each parent the sum of its
 * leaves. The tree and its counts arrive separately because they change
 * at different times: the tree when labels change, the counts with every
 * filter. Returns a new tree; the input is not changed.
 */
export function withCounts(
  tree: TaxonomyNode[],
  counts: Record<string, number>,
): TaxonomyNode[] {
  const place = (node: TaxonomyNode): TaxonomyNode => {
    if (!node.children || node.children.length === 0) {
      return { ...node, count: counts[node.id] ?? 0 };
    }
    const children = node.children.map(place);
    const count = children.reduce((sum, child) => sum + (child.count ?? 0), 0);
    return { ...node, children, count };
  };
  return tree.map(place);
}

/**
 * Collect all leaf node IDs from a tree.
 * Used for "select all / clear all" and counter display.
 */
export function collectLeafIds(nodes: TaxonomyNode[]): Set<string> {
  const ids = new Set<string>();
  const walk = (nodeList: TaxonomyNode[]) => {
    for (const node of nodeList) {
      if (!node.children || node.children.length === 0) {
        ids.add(node.id);
      } else {
        walk(node.children);
      }
    }
  };
  walk(nodes);
  return ids;
}

/**
 * A leaf the page's filters leave nothing of. It stays in the tree so
 * the reader sees it is there, but cannot be ticked: ticking it would
 * add nothing. A leaf without a count (a tree built without counts)
 * is never dead.
 */
export function isDeadLeaf(node: TaxonomyNode): boolean {
  return (!node.children || node.children.length === 0) && node.count === 0;
}

/** The leaf ids that can be ticked: every leaf that is not dead. */
export function collectLiveLeafIds(nodes: TaxonomyNode[]): Set<string> {
  const ids = new Set<string>();
  const walk = (nodeList: TaxonomyNode[]) => {
    for (const node of nodeList) {
      if (!node.children || node.children.length === 0) {
        if (!isDeadLeaf(node)) ids.add(node.id);
      } else {
        walk(node.children);
      }
    }
  };
  walk(nodes);
  return ids;
}
