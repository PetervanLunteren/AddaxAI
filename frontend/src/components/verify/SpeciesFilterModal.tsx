/**
 * Species Filter Modal for the Verify page.
 *
 * Thin wrapper around TreeSelector (inclusion mode) that prunes the full
 * taxonomy tree to only show species with actual detections.
 * Uses a working-copy pattern: changes are only applied on "Apply".
 */

import { useState, useMemo, useCallback, useEffect } from "react";
import type { TaxonomyNode } from "../../api/types";
import { pruneTaxonomyTree, collectLeafIds } from "../../lib/taxonomy-utils";
import { TreeSelector } from "../taxonomy/TreeSelector";
import { Button } from "../ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";

interface SpeciesFilterModalProps {
  /** Full taxonomy tree from the API. */
  fullTree: TaxonomyNode[];
  /** Species IDs that have detections (from filter-options endpoint). */
  detectedSpecies: string[];
  /** Currently active species filter values. */
  selectedSpecies: string[];
  /** Called with the new species list when the user clicks Apply. */
  onApply: (species: string[]) => void;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function SpeciesFilterModal({
  fullTree,
  detectedSpecies,
  selectedSpecies,
  onApply,
  open,
  onOpenChange,
}: SpeciesFilterModalProps) {
  // Working copy — initialized from props each time the modal opens
  const [workingSet, setWorkingSet] = useState<Set<string>>(new Set());

  // Prune tree to only branches leading to detected species
  const detectedSet = useMemo(() => new Set(detectedSpecies), [detectedSpecies]);
  const prunedTree = useMemo(
    () => pruneTaxonomyTree(fullTree, detectedSet),
    [fullTree, detectedSet]
  );
  const allPrunedLeafIds = useMemo(() => collectLeafIds(prunedTree), [prunedTree]);

  // Re-initialize working set each time the modal opens
  useEffect(() => {
    if (open) {
      // No filter active → start with all species selected
      if (selectedSpecies.length === 0) {
        setWorkingSet(new Set(allPrunedLeafIds));
      } else {
        setWorkingSet(new Set(selectedSpecies));
      }
    }
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleApply = useCallback(() => {
    onApply(Array.from(workingSet));
    onOpenChange(false);
  }, [workingSet, onApply, onOpenChange]);

  const handleCancel = useCallback(() => {
    onOpenChange(false);
  }, [onOpenChange]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[85vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle>Filter by species</DialogTitle>
          <DialogDescription>
            Select which species to include in results
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 overflow-auto">
          <TreeSelector
            tree={prunedTree}
            selectedIds={workingSet}
            mode="inclusion"
            onSelectionChange={setWorkingSet}
            height="500px"
            emptyMessage="No species with detections"
            counterText={`${workingSet.size} of ${allPrunedLeafIds.size} species selected`}
          />
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={handleCancel}>
            Cancel
          </Button>
          <Button onClick={handleApply}>Apply</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
