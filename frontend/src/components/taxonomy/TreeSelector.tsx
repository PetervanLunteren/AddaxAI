/**
 * Shared tree selection component.
 *
 * Extracted from SpeciesSelector so the same tree logic serves both:
 * - Settings page (exclusion mode: selected = excluded species)
 * - Verify filter modal (inclusion mode: selected = included species)
 *
 * Manages expand/collapse, search, bulk actions, and cascading toggle.
 * Does NOT manage data fetching, tree pruning, working-copy pattern,
 * dialog wrapping, or counter text — those are left to wrappers.
 */

import { useState, useEffect, useMemo, useCallback } from "react";
import { CheckSquare, Square, ChevronDown, ChevronRight, Search } from "lucide-react";
import type { TaxonomyNode } from "../../api/types";
import { collectLeafIds } from "../../lib/taxonomy-utils";
import { TreeNode } from "./TreeNode";
import { Button } from "../ui/button";
import { ScrollArea } from "../ui/scroll-area";

interface TreeSelectorProps {
  /** The tree to render (full or pruned). */
  tree: TaxonomyNode[];
  /** Currently selected/excluded leaf IDs. */
  selectedIds: Set<string>;
  /** "inclusion" = checked means selected; "exclusion" = checked means NOT in set (inverted). */
  mode: "inclusion" | "exclusion";
  /** Callback when selection changes. */
  onSelectionChange: (ids: Set<string>) => void;
  /** ScrollArea height (default "300px"). Ignored if fillHeight is set. */
  height?: string;
  /** If true, the component stretches to fill its parent instead of using a fixed height. */
  fillHeight?: boolean;
  /** Shown when tree is empty. */
  emptyMessage?: string;
  /** Optional counter text displayed above the search bar (e.g. "3 of 6 species selected"). */
  counterText?: string;
}

/** Recursively filter tree nodes by search query (case-insensitive match on name). */
function filterTree(nodes: TaxonomyNode[], query: string): TaxonomyNode[] {
  if (!query) return nodes;
  const lower = query.toLowerCase();
  const result: TaxonomyNode[] = [];
  for (const node of nodes) {
    if (node.name.toLowerCase().includes(lower)) {
      // Match: include the node with all its children
      result.push(node);
    } else if (node.children && node.children.length > 0) {
      // No direct match: check children
      const filteredChildren = filterTree(node.children, query);
      if (filteredChildren.length > 0) {
        result.push({ ...node, children: filteredChildren });
      }
    }
  }
  return result;
}

/** Collect all node IDs (parents + leaves) for expand/collapse. */
function collectAllNodeIds(nodes: TaxonomyNode[]): Set<string> {
  const ids = new Set<string>();
  const walk = (nodeList: TaxonomyNode[]) => {
    for (const node of nodeList) {
      ids.add(node.id);
      if (node.children) walk(node.children);
    }
  };
  walk(nodes);
  return ids;
}

export function TreeSelector({
  tree,
  selectedIds,
  mode,
  onSelectionChange,
  height = "300px",
  fillHeight = false,
  emptyMessage = "No species available",
  counterText,
}: TreeSelectorProps) {
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());
  const [searchQuery, setSearchQuery] = useState("");

  // Expand all nodes when tree changes (e.g. on load)
  useEffect(() => {
    setExpandedNodes(collectAllNodeIds(tree));
  }, [tree]);

  // Filtered tree from search
  const filteredTree = useMemo(
    () => filterTree(tree, searchQuery),
    [tree, searchQuery]
  );

  // All leaf IDs in the full tree (for bulk actions)
  const allLeafIds = useMemo(() => collectLeafIds(tree), [tree]);

  // Toggle a node and all its descendant leaves
  const toggleNodeAndDescendants = useCallback(
    (node: TaxonomyNode, add: boolean, set: Set<string>) => {
      if (!node.children || node.children.length === 0) {
        if (add) set.add(node.id);
        else set.delete(node.id);
      } else {
        for (const child of node.children) {
          toggleNodeAndDescendants(child, add, set);
        }
      }
    },
    []
  );

  const handleToggle = useCallback(
    (nodeId: string, checked: boolean) => {
      const newSet = new Set(selectedIds);

      const findAndToggle = (nodes: TaxonomyNode[]): boolean => {
        for (const node of nodes) {
          if (node.id === nodeId) {
            if (mode === "exclusion") {
              // Exclusion mode: checked means INCLUDED → remove from excluded set
              toggleNodeAndDescendants(node, !checked, newSet);
            } else {
              // Inclusion mode: checked means IN set
              toggleNodeAndDescendants(node, checked, newSet);
            }
            return true;
          }
          if (node.children && findAndToggle(node.children)) return true;
        }
        return false;
      };

      findAndToggle(tree);
      onSelectionChange(newSet);
    },
    [tree, selectedIds, mode, onSelectionChange, toggleNodeAndDescendants]
  );

  const handleExpand = useCallback((nodeId: string, expanded: boolean) => {
    setExpandedNodes((prev) => {
      const next = new Set(prev);
      if (expanded) next.add(nodeId);
      else next.delete(nodeId);
      return next;
    });
  }, []);

  // Bulk actions
  const handleSelectAll = useCallback(() => {
    if (mode === "exclusion") {
      onSelectionChange(new Set()); // Empty = nothing excluded
    } else {
      onSelectionChange(new Set(allLeafIds)); // All selected
    }
  }, [mode, allLeafIds, onSelectionChange]);

  const handleClearAll = useCallback(() => {
    if (mode === "exclusion") {
      onSelectionChange(new Set(allLeafIds)); // All excluded
    } else {
      onSelectionChange(new Set()); // None selected
    }
  }, [mode, allLeafIds, onSelectionChange]);

  const handleExpandAll = useCallback(() => {
    setExpandedNodes(collectAllNodeIds(tree));
  }, [tree]);

  const handleCollapseAll = useCallback(() => {
    setExpandedNodes(new Set());
  }, []);

  const selectLabel = mode === "exclusion" ? "Include all" : "Select all";
  const clearLabel = mode === "exclusion" ? "Exclude all" : "Clear all";

  return (
    <div className={`space-y-2 border rounded-md p-3${fillHeight ? " h-full flex flex-col overflow-hidden" : ""}`}>
      {counterText && (
        <p className="text-sm text-muted-foreground">{counterText}</p>
      )}
      {/* Search */}
      <div className="relative">
        <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
        <input
          type="text"
          placeholder="Search species..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="flex h-9 w-full rounded-md border border-input bg-background pl-9 pr-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
        />
      </div>

      {/* Bulk action buttons */}
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={handleSelectAll}
          className="flex-1 min-w-[100px]"
        >
          <CheckSquare className="h-4 w-4 mr-1.5" />
          {selectLabel}
        </Button>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={handleClearAll}
          className="flex-1 min-w-[100px]"
        >
          <Square className="h-4 w-4 mr-1.5" />
          {clearLabel}
        </Button>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={handleExpandAll}
          className="flex-1 min-w-[100px]"
        >
          <ChevronDown className="h-4 w-4 mr-1.5" />
          Expand all
        </Button>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={handleCollapseAll}
          className="flex-1 min-w-[100px]"
        >
          <ChevronRight className="h-4 w-4 mr-1.5" />
          Collapse all
        </Button>
      </div>

      {/* Tree */}
      {filteredTree.length > 0 ? (
        <ScrollArea className={`border rounded-md p-4 bg-background${fillHeight ? " h-0 flex-grow" : ""}`} style={fillHeight ? undefined : { height }}>
          <div className="space-y-1">
            {filteredTree.map((node) => (
              <TreeNode
                key={node.id}
                node={node}
                selectedClasses={selectedIds}
                excludedMode={mode === "exclusion"}
                expandedNodes={expandedNodes}
                onToggle={handleToggle}
                onExpand={handleExpand}
                level={0}
              />
            ))}
          </div>
        </ScrollArea>
      ) : (
        <div className="text-sm text-muted-foreground py-4">
          {searchQuery ? "No matching species" : emptyMessage}
        </div>
      )}
    </div>
  );
}
