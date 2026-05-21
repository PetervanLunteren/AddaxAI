/**
 * Step progress indicator for the folder-run stepper.
 *
 * Renders the four steps as numbered chips. The current step is
 * highlighted, completed steps render with a check mark, upcoming
 * steps are muted.
 *
 * Step ordering is fixed: model → edit → overview → save.
 * "Completed" means a step preceding the current one.
 *
 * Direct navigation: chips up to and including the backend's
 * persisted step are clickable so users can hop straight to any step
 * they've already reached. Upcoming steps stay disabled — jumping
 * forward past unfilled requirements (no models picked, no analysis
 * run, etc.) lands the user on a page that can't function. When
 * ``furthest`` isn't known yet (e.g. the brand-new-run path before
 * a run id exists) only the current step is interactive.
 */

import { Check } from "lucide-react";
import { cn } from "../../lib/utils";
import type { FolderRunStep } from "../../api/folder-runs";

interface Step {
  id: FolderRunStep;
  label: string;
}

const STEPS: Step[] = [
  { id: "model", label: "Setup" },
  { id: "edit", label: "Edit" },
  { id: "overview", label: "Summary" },
  { id: "save", label: "Output" },
];

interface StepProgressProps {
  current: FolderRunStep;
  /** Backend's persisted ``folder_run_state.step`` — the furthest
   * step the user has reached. Chips at or before this index are
   * clickable. Omit (or pass undefined) when no run is loaded yet;
   * in that case nothing is clickable. */
  furthest?: FolderRunStep;
  /** Called when the user clicks a clickable chip. */
  onStepClick?: (step: FolderRunStep) => void;
}

export function StepProgress({
  current,
  furthest,
  onStepClick,
}: StepProgressProps) {
  const currentIndex = STEPS.findIndex((s) => s.id === current);
  const furthestIndex =
    furthest !== undefined
      ? STEPS.findIndex((s) => s.id === furthest)
      : -1;
  // Allow nav to anything up to max(currentIndex, furthestIndex) — the
  // URL is the source of truth for what's on screen, the backend step
  // is the persisted "I made it this far" marker, and the user should
  // be free to revisit either.
  const reachableIndex = Math.max(currentIndex, furthestIndex);

  return (
    <ol className="flex w-full items-start">
      {STEPS.map((step, index) => {
        const isCurrent = index === currentIndex;
        const isDone = index < currentIndex;
        const isLast = index === STEPS.length - 1;
        const isClickable =
          !!onStepClick && index <= reachableIndex && !isCurrent;
        // Connector halves: the segment between step i-1 and i is
        // filled once step i-1 is done. The left half of a step shows
        // that incoming segment, the right half shows the outgoing one.
        const leftDone = index <= currentIndex;
        const rightDone = index < currentIndex;

        const chipInner = (
          <>
            <div
              className={cn(
                "flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-xs font-semibold transition-colors",
                isCurrent && "bg-primary text-primary-foreground",
                isDone && "bg-primary/20 text-primary",
                !isCurrent &&
                  !isDone &&
                  "bg-muted text-muted-foreground",
                isClickable && "group-hover:ring-2 group-hover:ring-primary/40",
              )}
              aria-current={isCurrent ? "step" : undefined}
            >
              {isDone ? (
                <Check className="h-4 w-4" />
              ) : (
                <span>{index + 1}</span>
              )}
            </div>
            <span
              className={cn(
                "hidden text-center text-xs font-medium sm:block",
                isCurrent && "text-foreground",
                isDone && "text-muted-foreground",
                !isCurrent && !isDone && "text-muted-foreground/70",
                isClickable && "group-hover:text-foreground",
              )}
            >
              {step.label}
            </span>
          </>
        );

        return (
          <li
            key={step.id}
            className="relative flex flex-1 flex-col items-center gap-2"
          >
            {/* Connector halves sit at the circle's vertical centre
                (h-7 = 28px, so 14px) and stop short of the circle
                (radius 14px + 10px gap = 24px from centre), meeting the
                neighbour's half at the column boundary. */}
            {index > 0 && (
              <div
                className={cn(
                  "absolute left-0 right-[calc(50%+24px)] top-[13px] h-px",
                  leftDone ? "bg-primary/30" : "bg-border",
                )}
                aria-hidden="true"
              />
            )}
            {!isLast && (
              <div
                className={cn(
                  "absolute left-[calc(50%+24px)] right-0 top-[13px] h-px",
                  rightDone ? "bg-primary/30" : "bg-border",
                )}
                aria-hidden="true"
              />
            )}
            {isClickable ? (
              <button
                type="button"
                onClick={() => onStepClick?.(step.id)}
                className="group relative z-10 flex flex-col items-center gap-2 rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
              >
                {chipInner}
              </button>
            ) : (
              <div className="relative z-10 flex flex-col items-center gap-2">
                {chipInner}
              </div>
            )}
          </li>
        );
      })}
    </ol>
  );
}
