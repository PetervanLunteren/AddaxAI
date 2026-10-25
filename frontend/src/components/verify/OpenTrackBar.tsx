/**
 * The row above the grid while one animal's frames are open: the way
 * back, and which animal.
 *
 * It says nothing about how many frames are shown, because the grid
 * below shows every box of the track, the number on the card's chip.
 * It used to be gated at the confidence setting and a sentence here
 * reconciled the two numbers; neither is needed now.
 *
 * It is drawn like the grid's own event rows, a label and a rule, not
 * as a third bordered card under the filter card and the toolbar card.
 * The way back is an outline button, the same as the step's Back button
 * below the grid.
 */

import { ArrowLeft, Film } from "lucide-react";

import { getDetectionColor, getDetectionDisplayName } from "../../lib/detection-utils";
import { Button } from "../ui/button";
import type { DetectionSummary } from "../../api/types";

interface OpenTrackBarProps {
  /** The card that was opened. Its label and colour name the animal. */
  card: DetectionSummary;
  fileName: string;
  onClose: () => void;
}

export function OpenTrackBar({ card, fileName, onClose }: OpenTrackBarProps) {
  return (
    <div
      data-testid="open-track-bar"
      className="flex flex-wrap items-center gap-x-3 gap-y-2 px-1 pt-1 pb-4"
    >
      <Button variant="outline" size="sm" onClick={onClose}>
        <ArrowLeft className="h-4 w-4" />
        Back to detections
      </Button>

      <span
        className="inline-flex items-center gap-1.5 text-xs font-medium whitespace-nowrap"
        style={{ color: getDetectionColor(card) }}
      >
        <Film className="h-3.5 w-3.5" />
        {getDetectionDisplayName(card)}
      </span>

      {fileName && (
        <span className="truncate text-xs text-muted-foreground">{fileName}</span>
      )}

      <div className="h-px min-w-8 flex-1 bg-border" />
    </div>
  );
}
