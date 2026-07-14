/**
 * Step progress indicator for the folder-run stepper.
 *
 * Renders the steps as numbered chips. The current step is
 * highlighted, completed steps render with a check mark, upcoming
 * steps are muted.
 *
 * Step ordering is fixed: setup → labels → save.
 * "Completed" means a step preceding the current one.
 *
 * Direct navigation: before the analysis has run, only the current
 * step and steps already reached are clickable — jumping forward past
 * unfilled requirements (no models picked, no analysis run) lands the
 * user on a page that can't function. Once analysis has completed
 * (``furthest`` past "setup"), every step functions, so the whole row
 * unlocks and the user can hop freely between Setup, Labels, and Save.
 * When ``furthest`` isn't known yet (e.g. the brand-new-run path before
 * a run id exists) only the current step is interactive.
 */

import { Fragment } from "react";
import { Check } from "lucide-react";
import { cn } from "../../lib/utils";
import type { FolderRunStep } from "../../api/folder-runs";

interface Step {
  id: FolderRunStep;
  label: string;
}

const STEPS: Step[] = [
  { id: "setup", label: "Setup" },
  { id: "labels", label: "Labels" },
  { id: "save", label: "Save" },
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
  // `step` only ever advances past "setup" via the post-processing
  // modal (or a resumed run, which also already processed), so
  // furthest > setup is a reliable "analysis has run" signal. Once
  // that's true every step functions — labels has detections to edit,
  // save has results to write — so unlock the whole row. Before that
  // the gate still guards against clicking into a page that can't work
  // yet (no analysis run). The URL is the source of truth for what's
  // on screen, so the current step is always reachable too.
  const setupIndex = STEPS.findIndex((s) => s.id === "setup");
  const processed = furthestIndex > setupIndex;
  const reachableIndex = processed
    ? STEPS.length - 1
    : Math.max(currentIndex, furthestIndex);

  return (
    <ol className="flex w-full items-start">
      {STEPS.map((step, index) => {
        const isCurrent = index === currentIndex;
        const isDone = index < currentIndex;
        const isClickable =
          !!onStepClick && index <= reachableIndex && !isCurrent;
        // The connector entering this step from the left is filled once
        // the previous step is complete.
        const connectorDone = index <= currentIndex;

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

        // Steps take their natural width and the connectors flex to
        // fill the gaps, so the first chip sits at the left edge and the
        // last at the right edge — the row spans the full content width.
        return (
          <Fragment key={step.id}>
            {index > 0 && (
              <li
                aria-hidden="true"
                className={cn(
                  "mx-2 mt-[13px] h-px flex-1",
                  connectorDone ? "bg-primary/30" : "bg-border",
                )}
              />
            )}
            <li className="shrink-0">
              {isClickable ? (
                <button
                  type="button"
                  onClick={() => onStepClick?.(step.id)}
                  className="group flex flex-col items-center gap-2 rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                >
                  {chipInner}
                </button>
              ) : (
                <div className="flex flex-col items-center gap-2">
                  {chipInner}
                </div>
              )}
            </li>
          </Fragment>
        );
      })}
    </ol>
  );
}
