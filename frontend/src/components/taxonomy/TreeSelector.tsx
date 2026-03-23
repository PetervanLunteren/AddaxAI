/**
 * Shared tree selection component with virtualized rendering.
 *
 * Extracted from SpeciesSelector so the same tree logic serves both:
 * - Settings page (exclusion mode: selected = excluded labels)
 * - Verify filter modal (inclusion mode: selected = included labels)
 *
 * Uses @tanstack/react-virtual to render only visible rows, enabling
 * smooth scrolling even with 2,000+ nodes.
 *
 * Manages expand/collapse, search, bulk actions, and cascading toggle.
 * Does NOT manage data fetching, tree pruning, working-copy pattern,
 * dialog wrapping, or counter text — those are left to wrappers.
 */

import { useState, useEffect, useMemo, useCallback, useRef } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { CheckSquare, Square, ChevronDown, ChevronRight, Search } from "lucide-react";
import type { TaxonomyNode } from "../../api/types";
import { collectLeafIds } from "../../lib/taxonomy-utils";
import { Checkbox } from "../ui/checkbox";
import { Button } from "../ui/button";

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
  /** Count unit label, e.g. "event". Omit for settings tree (no item counts). */
  countUnit?: string;
  /** Optional content rendered in the same row as the search bar (right side). */
  searchRowExtra?: React.ReactNode;
}

/** A flattened row representing one visible node in the tree. */
interface FlatRow {
  node: TaxonomyNode;
  depth: number;
  isLastChild: boolean;
  /** Which ancestor levels need a vertical connector line. */
  ancestorLines: boolean[];
}

/** Recursively filter tree nodes by search query (case-insensitive match on name or annotation). */
function filterTree(nodes: TaxonomyNode[], query: string): TaxonomyNode[] {
  if (!query) return nodes;
  const lower = query.toLowerCase();
  const result: TaxonomyNode[] = [];
  for (const node of nodes) {
    const nameMatch = node.name.toLowerCase().includes(lower);
    const annotationMatch = node.annotation?.toLowerCase().includes(lower);
    if (nameMatch || annotationMatch) {
      result.push(node);
    } else if (node.children && node.children.length > 0) {
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

/** Flatten tree into visible rows respecting expanded state. */
function flattenTree(
  nodes: TaxonomyNode[],
  expandedNodes: Set<string>,
  depth: number = 0,
  ancestorLines: boolean[] = [],
): FlatRow[] {
  const rows: FlatRow[] = [];
  for (let i = 0; i < nodes.length; i++) {
    const node = nodes[i];
    const isLast = i === nodes.length - 1;
    rows.push({ node, depth, isLastChild: isLast, ancestorLines: [...ancestorLines] });

    const hasChildren = node.children && node.children.length > 0;
    if (hasChildren && expandedNodes.has(node.id)) {
      const childAncestorLines = depth > 0
        ? [...ancestorLines, !isLast]
        : [];
      rows.push(...flattenTree(node.children, expandedNodes, depth + 1, childAncestorLines));
    }
  }
  return rows;
}

/** Sum node.count for all leaves in the tree that are in the given ID set. */
function sumLeafCounts(nodes: TaxonomyNode[], ids: Set<string>): number {
  let total = 0;
  for (const node of nodes) {
    if (!node.children || node.children.length === 0) {
      if (ids.has(node.id) && node.count != null) {
        total += node.count;
      }
    } else {
      total += sumLeafCounts(node.children, ids);
    }
  }
  return total;
}

/** Sum node.count for all leaves in the tree. */
function sumAllLeafCounts(nodes: TaxonomyNode[]): number {
  let total = 0;
  for (const node of nodes) {
    if (!node.children || node.children.length === 0) {
      if (node.count != null) total += node.count;
    } else {
      total += sumAllLeafCounts(node.children);
    }
  }
  return total;
}

function pluralize(unit: string, n: number): string {
  if (n === 1) return unit;
  if (unit.endsWith("y")) return unit.slice(0, -1) + "ies";
  return unit + "s";
}

/** Compute checkbox state for a node (recursive for parents). */
function getCheckState(
  node: TaxonomyNode,
  selectedIds: Set<string>,
  excludedMode: boolean,
): { checked: boolean; indeterminate: boolean } {
  const isLeaf = !node.children || node.children.length === 0;
  if (isLeaf) {
    const isInSet = selectedIds.has(node.id);
    return { checked: excludedMode ? !isInSet : isInSet, indeterminate: false };
  }

  let allChecked = true;
  let anyChecked = false;

  for (const child of node.children) {
    const state = getCheckState(child, selectedIds, excludedMode);
    if (state.checked && !state.indeterminate) {
      anyChecked = true;
    } else {
      allChecked = false;
    }
    if (state.indeterminate || state.checked) {
      anyChecked = true;
    }
  }

  if (allChecked) return { checked: true, indeterminate: false };
  if (anyChecked) return { checked: false, indeterminate: true };
  return { checked: false, indeterminate: false };
}

const ROW_HEIGHT = 32;

export function TreeSelector({
  tree,
  selectedIds,
  mode,
  onSelectionChange,
  height = "300px",
  fillHeight = false,
  emptyMessage = "No labels available",
  countUnit,
  searchRowExtra,
}: TreeSelectorProps) {
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());
  const [searchQuery, setSearchQuery] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  // Expand all nodes when tree loads
  useEffect(() => {
    setExpandedNodes(collectAllNodeIds(tree));
  }, [tree]);

  // Filtered tree from search
  const filteredTree = useMemo(
    () => filterTree(tree, searchQuery),
    [tree, searchQuery]
  );

  // Flatten tree into visible rows
  const flatRows = useMemo(
    () => flattenTree(filteredTree, expandedNodes),
    [filteredTree, expandedNodes]
  );

  // Virtualizer
  const virtualizer = useVirtualizer({
    count: flatRows.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 20,
  });

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
              toggleNodeAndDescendants(node, !checked, newSet);
            } else {
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
      onSelectionChange(new Set());
    } else {
      onSelectionChange(new Set(allLeafIds));
    }
  }, [mode, allLeafIds, onSelectionChange]);

  const handleClearAll = useCallback(() => {
    if (mode === "exclusion") {
      onSelectionChange(new Set(allLeafIds));
    } else {
      onSelectionChange(new Set());
    }
  }, [mode, allLeafIds, onSelectionChange]);

  const handleExpandAll = useCallback(() => {
    setExpandedNodes(collectAllNodeIds(tree));
  }, [tree]);

  const handleCollapseAll = useCallback(() => {
    setExpandedNodes(new Set());
  }, []);

  // Compute counter text
  const counterText = useMemo(() => {
    const totalCategories = allLeafIds.size;
    if (mode === "exclusion") {
      const includedCount = totalCategories - selectedIds.size;
      const suffix = selectedIds.size > 0 ? ` (${selectedIds.size} excluded)` : "";
      return `Currently included ${includedCount} of ${totalCategories}${suffix}`;
    }
    const selectedCount = selectedIds.size;
    let text = `${selectedCount} of ${totalCategories} ${pluralize("category", totalCategories)} selected`;
    if (countUnit) {
      const selectedItemCount = sumLeafCounts(tree, selectedIds);
      const totalItemCount = sumAllLeafCounts(tree);
      text += ` (${selectedItemCount} of ${totalItemCount} ${pluralize(countUnit, totalItemCount)})`;
    }
    return text;
  }, [allLeafIds, selectedIds, mode, countUnit, tree]);

  const selectLabel = mode === "exclusion" ? "Include all" : "Select all";
  const clearLabel = mode === "exclusion" ? "Exclude all" : "Clear all";
  const excludedMode = mode === "exclusion";

  return (
    <div className={`space-y-2 border rounded-md p-3${fillHeight ? " h-full flex flex-col overflow-hidden" : ""}`}>
      <p className="text-sm text-muted-foreground">{counterText}</p>
      {/* Search (+ optional extra content like geography filter) */}
      <div className={searchRowExtra ? "grid grid-cols-2 gap-2" : "flex"}>
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search labels..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="flex h-9 w-full rounded-md border border-input bg-background pl-9 pr-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          />
        </div>
        {searchRowExtra}
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

      {/* Virtualized tree */}
      {flatRows.length > 0 ? (
        <div
          ref={scrollRef}
          className={`border rounded-md bg-background overflow-auto${fillHeight ? " h-0 flex-grow" : ""}`}
          style={fillHeight ? undefined : { height }}
        >
          <div
            style={{
              height: `${virtualizer.getTotalSize()}px`,
              width: "100%",
              position: "relative",
            }}
          >
            {virtualizer.getVirtualItems().map((virtualRow) => {
              const { node, depth, isLastChild, ancestorLines } = flatRows[virtualRow.index];
              const hasChildren = node.children && node.children.length > 0;
              const isLeaf = !hasChildren;
              const expanded = expandedNodes.has(node.id);
              const { checked, indeterminate } = getCheckState(node, selectedIds, excludedMode);
              const indent = depth * 20;

              return (
                <div
                  key={node.id}
                  style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    width: "100%",
                    height: `${ROW_HEIGHT}px`,
                    transform: `translateY(${virtualRow.start}px)`,
                  }}
                >
                  <div
                    className="flex items-center h-full px-3 hover:bg-accent rounded cursor-pointer relative"
                    style={{ paddingLeft: `${indent + 12}px` }}
                    onClick={() => handleToggle(node.id, !checked)}
                  >
                    {/* Tree connector lines */}
                    {depth > 0 && (
                      <>
                        {ancestorLines.map((needsLine, idx) =>
                          needsLine ? (
                            <div
                              key={`a-${idx}`}
                              className="absolute border-l border-gray-300"
                              style={{
                                left: `${idx * 20 + 22}px`,
                                top: 0,
                                bottom: 0,
                              }}
                            />
                          ) : null
                        )}
                        <div
                          className="absolute border-l border-gray-300"
                          style={{
                            left: `${(depth - 1) * 20 + 22}px`,
                            top: 0,
                            bottom: isLastChild ? "50%" : 0,
                          }}
                        />
                        <div
                          className="absolute border-b border-gray-300"
                          style={{
                            left: `${(depth - 1) * 20 + 22}px`,
                            top: "50%",
                            width: "10px",
                          }}
                        />
                      </>
                    )}

                    {/* Expand/collapse icon */}
                    {hasChildren ? (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleExpand(node.id, !expanded);
                        }}
                        className="mr-1 w-4 h-4 flex items-center justify-center text-xs relative z-10"
                      >
                        {expanded ? "▼" : "▶"}
                      </button>
                    ) : (
                      <span className="mr-1 w-4 h-4" />
                    )}

                    {/* Checkbox */}
                    <Checkbox
                      checked={checked}
                      indeterminate={indeterminate}
                      onCheckedChange={(newChecked) => handleToggle(node.id, !!newChecked)}
                    />

                    {/* Node label */}
                    <span className="ml-2 text-sm truncate">
                      {node.name}
                      {node.annotation && (
                        <>
                          {" "}
                          (<em>{node.annotation}</em>)
                        </>
                      )}
                      {isLeaf && node.count != null && countUnit && (
                        <span className="ml-1 text-muted-foreground text-xs">
                          ({node.count} {pluralize(countUnit, node.count)})
                        </span>
                      )}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      ) : (
        <div className="text-sm text-muted-foreground py-4">
          {searchQuery ? "No matching labels" : emptyMessage}
        </div>
      )}
    </div>
  );
}
