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
