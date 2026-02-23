/**
 * Species Selection Modal
 *
 * Modal dialog containing the SpeciesSelector tree for excluding species.
 * Keeps the settings page clean by hiding the complex tree UI until needed.
 * Uses a working-copy pattern: changes are only applied on "Apply".
 */

import { useState, useCallback, useEffect } from "react";
import { SpeciesSelector } from "./SpeciesSelector";
import { Button } from "../ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";

interface SpeciesSelectionModalProps {
  modelId: string;
  excludedClasses: string[];
  onExclusionChange: (classes: string[]) => void;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  totalSpeciesCount: number;
}

export function SpeciesSelectionModal({
  modelId,
  excludedClasses,
  onExclusionChange,
  open,
  onOpenChange,
  totalSpeciesCount,
}: SpeciesSelectionModalProps) {
  const [workingExcluded, setWorkingExcluded] = useState<string[]>([]);

  // Re-initialize working copy each time the modal opens
  useEffect(() => {
    if (open) {
      setWorkingExcluded(excludedClasses);
    }
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleApply = useCallback(() => {
    onExclusionChange(workingExcluded);
    onOpenChange(false);
  }, [workingExcluded, onExclusionChange, onOpenChange]);

  const handleCancel = useCallback(() => {
    onOpenChange(false);
  }, [onOpenChange]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl h-[85vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle>Configure species selection</DialogTitle>
          <DialogDescription>
            Select which species to include in classifications
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 min-h-0">
          <SpeciesSelector
            modelId={modelId}
            excludedClasses={workingExcluded}
            onExclusionChange={setWorkingExcluded}
            fillHeight
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
