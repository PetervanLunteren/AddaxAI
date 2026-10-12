/**
 * Label Filter Modal for the Verify page.
 *
 * Accepts a pre-built taxonomy tree (already pruned server-side to only
 * detected labels with event counts). Uses a working-copy pattern:
 * changes are only applied on "Apply".
 */

import { useState, useMemo, useCallback, useEffect } from "react";
import type { TaxonomyNode } from "../../api/types";
import { collectLiveLeafIds } from "../../lib/taxonomy-utils";
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

interface LabelFilterModalProps {
  /** Pre-built taxonomy tree from the label-tree endpoint. */
  preBuiltTree: TaxonomyNode[];
  /** All leaf IDs from the label-tree endpoint. */
  allLeafIds: string[];
  /** Currently active label filter values. */
  selectedLabels: string[];
  /** Called with the new label list when the user clicks Apply. */
  onApply: (labels: string[]) => void;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Count unit label, e.g. "event" or "detection". */
  countUnit?: string;
  /** An empty `selectedLabels` normally means "no filter" and opens with
   * every live leaf ticked. Set this where an empty selection means
   * "nothing picked yet" (the project image copies dialog), so the
   * modal opens with nothing ticked and "Select all" is the deliberate
   * step. */
  startEmpty?: boolean;
}

export function LabelFilterModal({
  preBuiltTree,
  allLeafIds,
  selectedLabels,
  onApply,
  open,
  onOpenChange,
  countUnit,
  startEmpty = false,
}: LabelFilterModalProps) {
  // Working copy — initialized from props each time the modal opens
  const [workingSet, setWorkingSet] = useState<Set<string>>(new Set());

  // "No filter" means every label the page can show: the live leaves.
  // A dead leaf (nothing under the page's filters) is never ticked here;
  // `allLeafIds` is what the tree would be with no counts at all.
  const liveLeafIds = useMemo(
    () => (preBuiltTree.length ? collectLiveLeafIds(preBuiltTree) : new Set(allLeafIds)),
    [preBuiltTree, allLeafIds],
  );

  // Re-initialize working set each time the modal opens
  useEffect(() => {
    if (open) {
      // No filter active -> start with all labels selected, unless the
      // caller says an empty selection is "nothing picked yet".
      if (selectedLabels.length === 0 && !startEmpty) {
        setWorkingSet(new Set(liveLeafIds));
      } else {
        setWorkingSet(new Set(selectedLabels));
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
          <DialogTitle>Filter by label</DialogTitle>
          <DialogDescription>
            Select which labels to include in results
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 min-h-0">
          <TreeSelector
            tree={preBuiltTree}
            selectedIds={workingSet}
            mode="inclusion"
            onSelectionChange={setWorkingSet}
            fillHeight
            emptyMessage="No labels with observations"
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
