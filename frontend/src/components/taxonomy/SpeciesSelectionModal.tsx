/**
 * Species Selection Modal
 *
 * Modal dialog containing the SpeciesSelector tree for excluding species.
 * Keeps the settings page clean by hiding the complex tree UI until needed.
 */

import { SpeciesSelector } from "./SpeciesSelector";
import {
  Dialog,
  DialogContent,
  DialogDescription,
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
  const includedCount = totalSpeciesCount - excludedClasses.length;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[85vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle>Configure species selection</DialogTitle>
          <DialogDescription>
            Select which species to include in classifications.
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 overflow-auto">
          <SpeciesSelector
            modelId={modelId}
            excludedClasses={excludedClasses}
            onExclusionChange={onExclusionChange}
            treeHeight="500px"
          />
        </div>
      </DialogContent>
    </Dialog>
  );
}
