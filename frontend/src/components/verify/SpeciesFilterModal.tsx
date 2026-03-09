/**
 * Species Filter Modal for the Verify page.
 *
 * Accepts a pre-built taxonomy tree (already pruned server-side to only
 * detected species with event counts). Uses a working-copy pattern:
 * changes are only applied on "Apply".
 */

import { useState, useMemo, useCallback, useEffect } from "react";
import type { TaxonomyNode } from "../../api/types";
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
  /** Pre-built taxonomy tree from the species-tree endpoint. */
  preBuiltTree: TaxonomyNode[];
  /** All leaf IDs from the species-tree endpoint. */
  allLeafIds: string[];
  /** Currently active species filter values. */
  selectedSpecies: string[];
  /** Called with the new species list when the user clicks Apply. */
  onApply: (species: string[]) => void;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Count unit label, e.g. "event" or "detection". */
  countUnit?: string;
}

export function SpeciesFilterModal({
  preBuiltTree,
  allLeafIds,
  selectedSpecies,
  onApply,
  open,
  onOpenChange,
  countUnit,
}: SpeciesFilterModalProps) {
  // Working copy — initialized from props each time the modal opens
  const [workingSet, setWorkingSet] = useState<Set<string>>(new Set());

  const allLeafIdSet = useMemo(() => new Set(allLeafIds), [allLeafIds]);

  // Re-initialize working set each time the modal opens
  useEffect(() => {
    if (open) {
      // No filter active → start with all species selected
      if (selectedSpecies.length === 0) {
        setWorkingSet(new Set(allLeafIdSet));
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
      <DialogContent className="max-w-4xl h-[85vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle>Filter by species</DialogTitle>
          <DialogDescription>
            Select which species to include in results
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 min-h-0">
          <TreeSelector
            tree={preBuiltTree}
            selectedIds={workingSet}
            mode="inclusion"
            onSelectionChange={setWorkingSet}
            fillHeight
            emptyMessage="No species with detections"
            countUnit={countUnit ?? "event"}
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
