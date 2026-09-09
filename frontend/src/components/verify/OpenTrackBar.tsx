/**
 * The bar above the grid while one animal's frames are open.
 *
 * It has one job beyond the way back: reconciling two numbers that do
 * not match and should not. The badge on the card says how many frames
 * the tracker followed the animal for, which is a fact about the
 * animal. The grid shows the ones at or above the confidence you are
 * working at, which is usually fewer, because the tracker keeps boxes
 * far below the counting threshold. Saying both, in words, is cheaper
 * and more honest than making them agree.
 */

import { ArrowLeft, Film } from "lucide-react";

import { getDetectionColor, getDetectionDisplayName } from "../../lib/detection-utils";
import { Button } from "../ui/button";
import type { DetectionSummary } from "../../api/types";

interface OpenTrackBarProps {
  /** The card that was opened. Its label and colour name the animal. */
  card: DetectionSummary;
  /** Frames on screen: what passed the confidence you are working at. */
  shown: number;
  /** Frames the tracker followed the animal through. */
  tracked: number;
  fileName: string;
  loading: boolean;
  onClose: () => void;
}

export function OpenTrackBar({
  card,
  shown,
  tracked,
  fileName,
  loading,
  onClose,
}: OpenTrackBarProps) {
  return (
    <div
      data-testid="open-track-bar"
      className="mb-3 flex flex-wrap items-center gap-x-3 gap-y-1 rounded-lg border bg-card px-3 py-2"
    >
      <Button variant="ghost" size="sm" onClick={onClose} className="-ml-2">
        <ArrowLeft className="mr-1 h-4 w-4" />
        Back to detections
      </Button>

      <span
        className="inline-flex items-center gap-1.5 text-sm font-medium"
        style={{ color: getDetectionColor(card) }}
      >
        <Film className="h-3.5 w-3.5" />
        {getDetectionDisplayName(card)}
      </span>

      {fileName && (
        <span className="truncate text-xs text-muted-foreground">{fileName}</span>
      )}

      <span className="ml-auto text-xs text-muted-foreground">
        {loading ? (
          "Loading the frames..."
        ) : shown === tracked ? (
          <>Every one of its {tracked} frames.</>
        ) : (
          <>
            {shown} of its {tracked} frames, the ones at your confidence
            setting. A verdict here applies to the frames you pick, not to
            the whole animal.
          </>
        )}
      </span>
    </div>
  );
}
