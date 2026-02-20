/**
 * Pure utility functions for taxonomy tree operations.
 *
 * Used by both the settings species selector (full tree, exclusion mode)
 * and the verify filter modal (pruned tree, inclusion mode).
 */

import type { TaxonomyNode } from "../api/types";

/**
 * Prune a taxonomy tree to only contain branches leading to leaves in `keepSet`.
 *
 * - Leaf nodes are kept only if their `id` is in `keepSet`
 * - Parent nodes are kept only if they have surviving children
 * - Parent name annotations like ` (N)` are updated to reflect the pruned count
 * - Returns a new tree (no mutation)
 */
export function pruneTaxonomyTree(
  tree: TaxonomyNode[],
  keepSet: Set<string>
): TaxonomyNode[] {
  const prune = (nodes: TaxonomyNode[]): TaxonomyNode[] => {
    const result: TaxonomyNode[] = [];
    for (const node of nodes) {
      if (!node.children || node.children.length === 0) {
        // Leaf: keep only if in the set
        if (keepSet.has(node.id)) {
          result.push({ ...node });
        }
      } else {
        // Parent: recurse, keep if any children survive
        const prunedChildren = prune(node.children);
        if (prunedChildren.length > 0) {
          // Update the descendant count annotation in the name
          const leafCount = countLeaves(prunedChildren);
          const updatedName = node.name.replace(/\s*`\(\d+\)`\s*$/, "")
            + ` \`(${leafCount})\``;
          result.push({
            ...node,
            name: updatedName,
            children: prunedChildren,
          });
        }
      }
    }
    return result;
  };
  return prune(tree);
}

/** Count leaf nodes in a subtree. */
function countLeaves(nodes: TaxonomyNode[]): number {
  let count = 0;
  for (const node of nodes) {
    if (!node.children || node.children.length === 0) {
      count++;
    } else {
      count += countLeaves(node.children);
    }
  }
  return count;
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
