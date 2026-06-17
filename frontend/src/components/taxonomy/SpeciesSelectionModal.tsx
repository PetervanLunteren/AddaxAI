/**
 * Species Selection Modal
 *
 * Modal dialog containing the SpeciesSelector tree for excluding species.
 * Opened from LabelSelectionField when the user wants to refine the included
 * labels by hand. Uses a working-copy pattern: changes are only applied on
 * "Apply". The country/geofence filter lives in LabelSelectionField, not here.
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
}

export function SpeciesSelectionModal({
  modelId,
  excludedClasses,
  onExclusionChange,
  open,
  onOpenChange,
}: SpeciesSelectionModalProps) {
  const [workingExcluded, setWorkingExcluded] = useState<string[]>([]);

  // Re-initialize the working copy each time the modal opens.
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
          <DialogTitle>Configure label selection</DialogTitle>
          <DialogDescription>
            Select which labels to include in classifications
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
