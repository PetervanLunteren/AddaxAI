/**
 * Species Selector Component
 *
 * Hierarchical tree for excluding species from a classification model's taxonomy.
 * Extracted from CreateProjectDialog for reuse in project settings.
 *
 * Features:
 * - Tri-state checkboxes (included, excluded, indeterminate)
 * - Expand/collapse functionality
 * - Include all / Exclude all bulk actions
 * - Visual tree connectors
 * - Exclusion counter
 *
 * Usage example:
 * ```tsx
 * const [excludedClasses, setExcludedClasses] = useState<string[]>([]);
 *
 * <SpeciesSelector
 *   modelId="EUR-DF-v1-3"
 *   excludedClasses={excludedClasses}
 *   onExclusionChange={(classes) => setExcludedClasses(classes)}
 * />
 * ```
 *
 * Following DEVELOPERS.md principles:
 * - Type hints everywhere
 * - Simple, clear structure
 * - Explicit error handling
 */

import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { CheckSquare, Square, ChevronDown, ChevronRight } from "lucide-react";
import { modelsApi } from "../../api/models";
import type { TaxonomyNode } from "../../api/types";
import { TreeNode } from "./TreeNode";
import { Button } from "../ui/button";
import { ScrollArea } from "../ui/scroll-area";

interface SpeciesSelectorProps {
  /**
   * ID of the classification model to load taxonomy for
   */
  modelId: string;

  /**
   * Array of currently excluded species class IDs
   */
  excludedClasses: string[];

  /**
   * Callback when exclusion changes
   */
  onExclusionChange: (classes: string[]) => void;

  /**
   * Optional height for the scrollable tree area (default: 300px)
   */
  treeHeight?: string;
}

export function SpeciesSelector({
  modelId,
  excludedClasses,
  onExclusionChange,
  treeHeight = "300px",
}: SpeciesSelectorProps) {
  const [excludedSet, setExcludedSet] = useState<Set<string>>(new Set(excludedClasses));
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());

  // Sync internal state with prop changes
  useEffect(() => {
    setExcludedSet(new Set(excludedClasses));
  }, [excludedClasses]);

  // Fetch taxonomy from backend
  const { data: taxonomy, isLoading } = useQuery({
    queryKey: ["taxonomy", modelId],
    queryFn: () => modelsApi.getTaxonomy(modelId),
    enabled: !!modelId,
  });

  // Expand all nodes by default when taxonomy loads
  useEffect(() => {
    if (taxonomy?.tree) {
      const allNodeIds = new Set<string>();
      const collectAllNodeIds = (nodes: TaxonomyNode[]) => {
        for (const node of nodes) {
          allNodeIds.add(node.id);
          if (node.children) {
            collectAllNodeIds(node.children);
          }
        }
      };
      collectAllNodeIds(taxonomy.tree);
      setExpandedNodes(allNodeIds);
    }
  }, [taxonomy]);

  const allClasses = taxonomy?.all_classes || [];
  const tree = taxonomy?.tree || [];

  // Update parent when internal exclusion changes
  const updateExclusion = (newSet: Set<string>) => {
    setExcludedSet(newSet);
    onExclusionChange(Array.from(newSet));
  };

  // Toggle a node (and all descendants)
  const handleToggle = (nodeId: string, checked: boolean) => {
    const newExcluded = new Set(excludedSet);

    const findAndToggleNode = (nodes: TaxonomyNode[]): boolean => {
      for (const node of nodes) {
        if (node.id === nodeId) {
          // Inverted logic: checked means INCLUDED (remove from excluded)
          toggleNodeAndDescendants(node, !checked, newExcluded);
          return true;
        }
        if (node.children && findAndToggleNode(node.children)) {
          return true;
        }
      }
      return false;
    };

    findAndToggleNode(tree);
    updateExclusion(newExcluded);
  };

  const toggleNodeAndDescendants = (
    node: TaxonomyNode,
    excluded: boolean,
    excludedSet: Set<string>
  ) => {
    if (!node.children || node.children.length === 0) {
      // Leaf node
      if (excluded) {
        excludedSet.add(node.id);
      } else {
        excludedSet.delete(node.id);
      }
    } else {
      // Parent node - toggle all descendants
      for (const child of node.children) {
        toggleNodeAndDescendants(child, excluded, excludedSet);
      }
    }
  };

  // Bulk actions
  const handleIncludeAll = () => {
    updateExclusion(new Set()); // Empty = nothing excluded = all included
  };

  const handleExcludeAll = () => {
    updateExclusion(new Set(allClasses)); // All excluded
  };

  const handleExpandAll = () => {
    const allNodeIds = new Set<string>();
    const collectAllNodeIds = (nodes: TaxonomyNode[]) => {
      for (const node of nodes) {
        allNodeIds.add(node.id);
        if (node.children) {
          collectAllNodeIds(node.children);
        }
      }
    };
    collectAllNodeIds(tree);
    setExpandedNodes(allNodeIds);
  };

  const handleCollapseAll = () => {
    setExpandedNodes(new Set());
  };

  const handleExpand = (nodeId: string, expanded: boolean) => {
    const newExpanded = new Set(expandedNodes);
    if (expanded) {
      newExpanded.add(nodeId);
    } else {
      newExpanded.delete(nodeId);
    }
    setExpandedNodes(newExpanded);
  };

  return (
    <div className="space-y-2 border rounded-md p-3">
      {/* Bulk action buttons */}
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={handleIncludeAll}
          className="flex-1 min-w-[100px]"
        >
          <CheckSquare className="h-4 w-4 mr-1.5" />
          Include all
        </Button>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={handleExcludeAll}
          className="flex-1 min-w-[100px]"
        >
          <Square className="h-4 w-4 mr-1.5" />
          Exclude all
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

      {/* Inclusion counter (showing included = total - excluded) */}
      <p className="text-sm text-muted-foreground">
        Currently included {allClasses.length - excludedSet.size} of {allClasses.length}
        {excludedSet.size > 0 && <span className="ml-1">({excludedSet.size} excluded)</span>}
      </p>

      {/* Taxonomy tree */}
      {isLoading ? (
        <div className="text-sm text-muted-foreground py-4">
          Loading taxonomy...
        </div>
      ) : tree.length > 0 ? (
        <ScrollArea className="border rounded-md p-4 bg-background" style={{ height: treeHeight }}>
          <div className="space-y-1">
            {tree.map((node) => (
              <TreeNode
                key={node.id}
                node={node}
                selectedClasses={excludedSet}
                excludedMode={true}
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
          No taxonomy available for this model
        </div>
      )}
    </div>
  );
}
