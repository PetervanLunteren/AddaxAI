/**
 * Taxonomy Editor Component.
 *
 * Following DEVELOPERS.md principles:
 * - Type hints everywhere
 * - Simple, clear implementation
 *
 * Hierarchical tree with tri-state checkboxes for selecting species.
 */

import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { modelsApi } from "../../api/models";
import { TreeNode } from "./TreeNode";
import type { TaxonomyNode } from "../../api/types";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";
import { Button } from "../ui/button";
import { ScrollArea } from "../ui/scroll-area";

interface TaxonomyEditorProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  modelId: string | null;
  selectedClasses: string[];
  onSave: (selectedClasses: string[]) => void;
}

export function TaxonomyEditor({
  open,
  onOpenChange,
  modelId,
  selectedClasses,
  onSave,
}: TaxonomyEditorProps) {
  const [localSelected, setLocalSelected] = useState<Set<string>>(
    new Set(selectedClasses)
  );
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());

  // Fetch taxonomy when model changes
  const { data: taxonomy, isLoading } = useQuery({
    queryKey: ["taxonomy", modelId],
    queryFn: () => modelsApi.getTaxonomy(modelId!),
    enabled: open && !!modelId,
  });

  // Update local state when selectedClasses prop changes or when taxonomy loads
  useEffect(() => {
    if (selectedClasses.length === 0 && taxonomy?.all_classes) {
      // If no classes selected, default to all classes
      setLocalSelected(new Set(taxonomy.all_classes));
    } else {
      setLocalSelected(new Set(selectedClasses));
    }
  }, [selectedClasses, taxonomy?.all_classes]);

  const allClasses = taxonomy?.all_classes || [];
  const tree = taxonomy?.tree || [];

  // Handle node toggle with cascading to children
  const handleToggle = (nodeId: string, checked: boolean) => {
    const newSelected = new Set(localSelected);

    // Find the node in the tree
    const findAndToggleNode = (nodes: TaxonomyNode[]): boolean => {
      for (const node of nodes) {
        if (node.id === nodeId) {
          // Toggle this node and all descendants
          toggleNodeAndDescendants(node, checked, newSelected);
          return true;
        }
        if (node.children && findAndToggleNode(node.children)) {
          return true;
        }
      }
      return false;
    };

    findAndToggleNode(tree);
    setLocalSelected(newSelected);
  };

  // Recursively toggle node and all descendants
  const toggleNodeAndDescendants = (
    node: TaxonomyNode,
    checked: boolean,
    selectedSet: Set<string>
  ) => {
    // If it's a leaf node (model_class), toggle it
    if (!node.children || node.children.length === 0) {
      if (checked) {
        selectedSet.add(node.id);
      } else {
        selectedSet.delete(node.id);
      }
    } else {
      // If it's a parent, toggle all descendants
      for (const child of node.children) {
        toggleNodeAndDescendants(child, checked, selectedSet);
      }
    }
  };

  const handleSelectAll = () => {
    setLocalSelected(new Set(allClasses));
  };

  const handleDeselectAll = () => {
    setLocalSelected(new Set());
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

  const handleSave = () => {
    onSave(Array.from(localSelected));
    onOpenChange(false);
  };

  const handleCancel = () => {
    // Reset to original selection
    setLocalSelected(new Set(selectedClasses));
    onOpenChange(false);
  };

  if (!modelId) {
    return (
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Configure Taxonomy</DialogTitle>
            <DialogDescription>
              Please select a classification model first.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button onClick={() => onOpenChange(false)}>Close</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    );
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl h-[80vh] flex flex-col">
        <DialogHeader>
          <DialogTitle>Configure species taxonomy</DialogTitle>
          <DialogDescription>
            Select which species classes you want to monitor for this project.
            {localSelected.size > 0 && (
              <span className="ml-2 font-medium">
                ({localSelected.size} selected)
              </span>
            )}
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 flex flex-col gap-4 min-h-0">
          {/* Bulk actions */}
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleSelectAll}
              disabled={isLoading}
            >
              Select all ({allClasses.length})
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={handleDeselectAll}
              disabled={isLoading}
            >
              Deselect all
            </Button>
            <div className="flex-1" />
            <Button
              variant="outline"
              size="sm"
              onClick={handleExpandAll}
              disabled={isLoading}
            >
              Expand all
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={handleCollapseAll}
              disabled={isLoading}
            >
              Collapse all
            </Button>
          </div>

          {/* Taxonomy tree */}
          <ScrollArea className="flex-1 border rounded-md">
            {isLoading ? (
              <div className="p-4 text-center text-muted-foreground">
                Loading taxonomy...
              </div>
            ) : tree.length === 0 ? (
              <div className="p-4 text-center text-muted-foreground">
                No taxonomy available
              </div>
            ) : (
              <div className="p-4">
                {tree.map((node, index) => (
                  <TreeNode
                    key={node.id}
                    node={node}
                    level={0}
                    selectedClasses={localSelected}
                    onToggle={handleToggle}
                    expandedNodes={expandedNodes}
                    onExpand={handleExpand}
                    isLastChild={index === tree.length - 1}
                  />
                ))}
              </div>
            )}
          </ScrollArea>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={handleCancel}>
            Cancel
          </Button>
          <Button onClick={handleSave} disabled={isLoading}>
            Save ({localSelected.size} species)
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
