/**
 * Recursive Tree Node Component.
 *
 * Displays a single node in the taxonomy tree with:
 * - Expand/collapse functionality
 * - Tri-state checkbox (checked, unchecked, indeterminate)
 * - Hierarchical indentation
 * - Scientific names with bold model_class at leaf nodes
 * - Visual tree connector lines
 */

import { Checkbox } from "../ui/checkbox";
import type { TaxonomyNode as TaxonomyNodeType } from "../../api/types";

interface TreeNodeProps {
  node: TaxonomyNodeType;
  level: number;
  selectedClasses: Set<string>;
  onToggle: (nodeId: string, checked: boolean) => void;
  expandedNodes: Set<string>;
  onExpand: (nodeId: string, expanded: boolean) => void;
  isLastChild?: boolean;
  ancestorLines?: boolean[]; // Tracks which ancestor levels need vertical lines
}

export function TreeNode({
  node,
  level,
  selectedClasses,
  onToggle,
  expandedNodes,
  onExpand,
  isLastChild = false,
  ancestorLines = [],
}: TreeNodeProps) {
  const expanded = expandedNodes.has(node.id);

  const hasChildren = node.children && node.children.length > 0;
  const isLeaf = !hasChildren;

  // Calculate checkbox state
  const getCheckboxState = (): { checked: boolean; indeterminate: boolean } => {
    if (isLeaf) {
      // Leaf nodes: checked if in selectedClasses
      return { checked: selectedClasses.has(node.id), indeterminate: false };
    }

    // Parent nodes: check children state
    const checkedChildren = node.children.filter((child) => {
      const childState = getChildCheckState(child);
      return childState.checked && !childState.indeterminate;
    });

    const indeterminateChildren = node.children.filter((child) => {
      const childState = getChildCheckState(child);
      return childState.indeterminate || childState.checked;
    });

    if (checkedChildren.length === node.children.length) {
      return { checked: true, indeterminate: false };
    } else if (indeterminateChildren.length > 0) {
      return { checked: false, indeterminate: true };
    } else {
      return { checked: false, indeterminate: false };
    }
  };

  const getChildCheckState = (
    child: TaxonomyNodeType
  ): { checked: boolean; indeterminate: boolean } => {
    if (!child.children || child.children.length === 0) {
      return { checked: selectedClasses.has(child.id), indeterminate: false };
    }

    const checkedDescendants = child.children.filter((descendant) => {
      const state = getChildCheckState(descendant);
      return state.checked && !state.indeterminate;
    });

    const indeterminateDescendants = child.children.filter((descendant) => {
      const state = getChildCheckState(descendant);
      return state.indeterminate || state.checked;
    });

    if (checkedDescendants.length === child.children.length) {
      return { checked: true, indeterminate: false };
    } else if (indeterminateDescendants.length > 0) {
      return { checked: false, indeterminate: true };
    } else {
      return { checked: false, indeterminate: false };
    }
  };

  const { checked, indeterminate } = getCheckboxState();

  const handleToggle = () => {
    onToggle(node.id, !checked);
  };

  const handleExpand = (e: React.MouseEvent) => {
    e.stopPropagation();
    onExpand(node.id, !expanded);
  };

  const indentationWidth = level * 20; // 20px per level

  return (
    <div className="relative">
      <div
        className="flex items-center py-1 hover:bg-accent rounded cursor-pointer relative"
        style={{ paddingLeft: `${indentationWidth}px` }}
        onClick={handleToggle}
      >
        {/* Tree connector lines */}
        {level > 0 && (
          <>
            {/* Draw vertical lines for all ancestor levels that have more siblings */}
            {ancestorLines.map((needsLine, idx) =>
              needsLine ? (
                <div
                  key={`ancestor-${idx}`}
                  className="absolute border-l border-gray-300"
                  style={{
                    left: `${idx * 20 + 10}px`,
                    top: 0,
                    bottom: 0,
                  }}
                />
              ) : null
            )}

            {/* Vertical line for current level */}
            <div
              className="absolute border-l border-gray-300"
              style={{
                left: `${(level - 1) * 20 + 10}px`,
                top: 0,
                bottom: isLastChild ? "50%" : 0,
              }}
            />
            {/* Horizontal line */}
            <div
              className="absolute border-b border-gray-300"
              style={{
                left: `${(level - 1) * 20 + 10}px`,
                top: "50%",
                width: "10px",
              }}
            />
          </>
        )}

        {/* Expand/collapse icon */}
        {hasChildren && (
          <button
            onClick={handleExpand}
            className="mr-1 w-4 h-4 flex items-center justify-center text-xs relative z-10"
          >
            {expanded ? "▼" : "▶"}
          </button>
        )}
        {!hasChildren && <span className="mr-1 w-4 h-4" />}

        {/* Checkbox */}
        <Checkbox
          checked={checked}
          indeterminate={indeterminate}
          onCheckedChange={(newChecked) => onToggle(node.id, !!newChecked)}
        />

        {/* Node label - backend provides formatted names with counts */}
        <span
          className="ml-2 text-sm"
          dangerouslySetInnerHTML={{
            __html: node.name
              // Convert **bold** to <strong>
              .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
              // Convert _italic_ to <em>
              .replace(/_([^_]+)_/g, '<em>$1</em>')
              // Convert `code` to <code> with teal color (#0f6064 primary)
              .replace(/`([^`]+)`/g, '<code style="color: #0f6064; font-size: 0.85em;">$1</code>')
          }}
        />
      </div>

      {/* Render children if expanded */}
      {hasChildren && expanded && (
        <div>
          {node.children.map((child, index) => {
            // Calculate ancestorLines for child
            // Add a line at current level only if this node is NOT the last child
            // But only if we're not at the root level (level 0 doesn't contribute to ancestor lines)
            const childAncestorLines = level > 0
              ? [...ancestorLines, !isLastChild]
              : [];

            return (
              <TreeNode
                key={child.id}
                node={child}
                level={level + 1}
                selectedClasses={selectedClasses}
                onToggle={onToggle}
                expandedNodes={expandedNodes}
                onExpand={onExpand}
                isLastChild={index === node.children.length - 1}
                ancestorLines={childAncestorLines}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}
