/**
 * Species Selector Component
 *
 * Hierarchical tree for excluding species from a classification model's taxonomy.
 * Thin wrapper around TreeSelector (exclusion mode) that handles data fetching
 * and the exclusion counter.
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
 */

import { useState, useEffect, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { modelsApi } from "../../api/models";
import { TreeSelector } from "./TreeSelector";

interface SpeciesSelectorProps {
  /** ID of the classification model to load taxonomy for. */
  modelId: string;
  /** Array of currently excluded species class IDs. */
  excludedClasses: string[];
  /** Callback when exclusion changes. */
  onExclusionChange: (classes: string[]) => void;
  /** Optional height for the scrollable tree area (default: 300px). Ignored if fillHeight is set. */
  treeHeight?: string;
  /** If true, stretch to fill parent instead of using a fixed height. */
  fillHeight?: boolean;
}

export function SpeciesSelector({
  modelId,
  excludedClasses,
  onExclusionChange,
  treeHeight = "300px",
  fillHeight = false,
}: SpeciesSelectorProps) {
  const [excludedSet, setExcludedSet] = useState<Set<string>>(new Set(excludedClasses));

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

  const tree = taxonomy?.tree || [];

  const handleSelectionChange = useCallback(
    (newSet: Set<string>) => {
      setExcludedSet(newSet);
      onExclusionChange(Array.from(newSet));
    },
    [onExclusionChange]
  );

  return (
    <div className={fillHeight ? "h-full" : ""}>
      {isLoading ? (
        <div className="text-sm text-muted-foreground py-4">
          Loading taxonomy...
        </div>
      ) : (
        <TreeSelector
          tree={tree}
          selectedIds={excludedSet}
          mode="exclusion"
          onSelectionChange={handleSelectionChange}
          height={treeHeight}
          fillHeight={fillHeight}
          emptyMessage="No taxonomy available for this model"
        />
      )}
    </div>
  );
}
