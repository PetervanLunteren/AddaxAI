/**
 * The row above the grid in a focused view: the way back, and one fact
 * the view shows nowhere else.
 *
 * Two views use it, an animal's frames and the suggestions review. Both
 * are a piece of work you step into from the grid, and in both the
 * filter card and the toolbar are hidden, because their controls do not
 * apply there. This bar is what is left, so the two feel the same.
 *
 * The context line names the view, so it still reads after a break:
 * "Frames the AI followed" plus the clip name (nothing else in that view
 * shows it), or "Suggestions left: 8". It says "followed", not "one
 * animal", because whether it was one animal is what the view checks.
 * It names no species: every card below carries its own label chip.
 *
 * It is drawn like the grid's own event rows, a label and a rule, not
 * as one more bordered card. The way back is an outline button, the same
 * as the step's Back button below the grid.
 */

import { ArrowLeft } from "lucide-react";

import { Button } from "../ui/button";

interface FocusedViewBarProps {
  /** Where Back goes, said on the button: "Back to detections", or
   *  "Back to suggestions" for frames opened from a suggestion. */
  backLabel: string;
  /** What the view is: "Frames the AI followed, <clip>" or
   *  "Suggestions left: <n>". */
  context: string;
  onBack: () => void;
}

export function FocusedViewBar({ backLabel, context, onBack }: FocusedViewBarProps) {
  return (
    <div
      data-testid="focused-view-bar"
      className="flex flex-wrap items-center gap-x-3 gap-y-2 px-1 pt-1 pb-4"
    >
      <Button variant="outline" size="sm" onClick={onBack}>
        <ArrowLeft className="h-4 w-4" />
        {backLabel}
      </Button>

      {context && (
        <span className="truncate text-sm text-muted-foreground">{context}</span>
      )}

      <div className="h-px min-w-8 flex-1 bg-border" />
    </div>
  );
}
