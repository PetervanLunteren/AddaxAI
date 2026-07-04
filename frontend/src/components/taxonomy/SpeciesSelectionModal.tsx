/**
 * Species Selection Modal
 *
 * Modal dialog containing the SpeciesSelector tree for excluding species.
 * Opened from LabelSelectionField when the user wants to refine the included
 * labels by hand. Uses a working-copy pattern: changes are only applied on
 * "Apply". The country/geofence filter lives in LabelSelectionField, not here.
 *
 * Save / Load: the current selection can be written to a small JSON file the
 * user names and manages themselves (so they can keep one per region and share
 * it with colleagues), and loaded back later. The file stores the *included*
 * species (human-readable) plus the model id; loading converts back to the
 * internal exclusion set against the current model, so a species the file
 * doesn't mention becomes excluded and any label the current model lacks is
 * ignored. No preset store, no database — the files are the user's to own.
 */

import { useState, useCallback, useEffect, useRef } from "react";
import { toast } from "sonner";
import { SpeciesSelector } from "./SpeciesSelector";
import { Button } from "../ui/button";
import { downloadTextFile } from "../../lib/download";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";

/** Marker + version so a loaded file is recognisably ours. */
const FILE_MARKER = "addaxai_species_selection";

interface SpeciesSelectionFile {
  [FILE_MARKER]: number;
  model_id: string;
  included: string[];
}

interface SpeciesSelectionModalProps {
  modelId: string;
  excludedClasses: string[];
  /** Every label name in the current model taxonomy. Needed to convert
   *  between the internal exclusion set and the included list stored in the
   *  file. */
  allClasses: string[];
  onExclusionChange: (classes: string[]) => void;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function SpeciesSelectionModal({
  modelId,
  excludedClasses,
  allClasses,
  onExclusionChange,
  open,
  onOpenChange,
}: SpeciesSelectionModalProps) {
  const [workingExcluded, setWorkingExcluded] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

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

  // Save the on-screen selection as included species (minus stale exclusions
  // not in the current model).
  const handleSave = useCallback(() => {
    const excludedSet = new Set(workingExcluded);
    const included = allClasses.filter((c) => !excludedSet.has(c));
    const payload: SpeciesSelectionFile = {
      [FILE_MARKER]: 1,
      model_id: modelId,
      included,
    };
    const safeModel = modelId.replace(/[^A-Za-z0-9._-]+/g, "_") || "model";
    downloadTextFile(
      `addaxai-species-${safeModel}.json`,
      JSON.stringify(payload, null, 2),
    );
  }, [workingExcluded, allClasses, modelId]);

  const applyLoadedFile = useCallback(
    (text: string) => {
      let parsed: unknown;
      try {
        parsed = JSON.parse(text);
      } catch {
        toast.error("That file is not a valid species selection");
        return;
      }
      const included =
        parsed &&
        typeof parsed === "object" &&
        Array.isArray((parsed as SpeciesSelectionFile).included)
          ? (parsed as SpeciesSelectionFile).included
          : null;
      if (!included) {
        toast.error("That file is not a valid species selection");
        return;
      }
      const includedSet = new Set(included);
      const overlap = allClasses.filter((c) => includedSet.has(c));
      if (overlap.length === 0) {
        const savedFor = (parsed as SpeciesSelectionFile).model_id;
        toast.error("No matching species for this model", {
          description: savedFor
            ? `The file was saved for ${savedFor}.`
            : undefined,
        });
        return;
      }
      // Anything the file doesn't include becomes excluded for this model.
      setWorkingExcluded(allClasses.filter((c) => !includedSet.has(c)));

      const savedFor = (parsed as SpeciesSelectionFile).model_id;
      if (savedFor && savedFor !== modelId) {
        toast.warning("Selection was saved for a different model", {
          description: `Kept the ${overlap.length} species this model has; the rest are excluded.`,
        });
      } else {
        toast.success(`Loaded ${overlap.length} species`);
      }
    },
    [allClasses, modelId],
  );

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      // Reset immediately so re-selecting the same file fires change again.
      e.target.value = "";
      if (!file) return;
      file
        .text()
        .then(applyLoadedFile)
        .catch(() => toast.error("Could not read that file"));
    },
    [applyLoadedFile],
  );

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
          <input
            ref={fileInputRef}
            type="file"
            accept="application/json,.json"
            className="hidden"
            onChange={handleFileChange}
          />
          <Button
            variant="outline"
            className="mr-auto"
            onClick={handleSave}
          >
            Save selection
          </Button>
          <Button
            variant="outline"
            onClick={() => fileInputRef.current?.click()}
          >
            Load selection
          </Button>
          <Button variant="outline" onClick={handleCancel}>
            Cancel
          </Button>
          <Button onClick={handleApply}>Apply</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
