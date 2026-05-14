/**
 * Step progress indicator for the folder-run stepper.
 *
 * Renders the five steps as numbered chips. The current step is
 * highlighted, completed steps render with a check mark, upcoming
 * steps are muted. Pure presentational: takes the current step name
 * and computes the visual state.
 *
 * Step ordering is fixed: folder → model → run → review → save.
 * "Completed" means a step preceding the current one. The flow does
 * not enforce forward-only navigation here; that lives in the
 * FolderRunLayout (and on the backend via folder_run_state.step).
 */

import { Check } from "lucide-react";
import { cn } from "../../lib/utils";
import type { FolderRunStep } from "../../api/folder-runs";

interface Step {
  id: FolderRunStep;
  label: string;
}

const STEPS: Step[] = [
  { id: "folder", label: "Choose folder" },
  { id: "model", label: "Choose AI" },
  { id: "run", label: "Run analysis" },
  { id: "review", label: "Review results" },
  { id: "save", label: "Save outputs" },
];

interface StepProgressProps {
  current: FolderRunStep;
}

export function StepProgress({ current }: StepProgressProps) {
  const currentIndex = STEPS.findIndex((s) => s.id === current);

  return (
    <ol className="mx-auto flex w-full max-w-3xl items-center justify-between gap-2">
      {STEPS.map((step, index) => {
        const isCurrent = index === currentIndex;
        const isDone = index < currentIndex;
        const isLast = index === STEPS.length - 1;

        return (
          <li
            key={step.id}
            className="flex flex-1 items-center gap-2 last:flex-none"
          >
            <div className="flex items-center gap-2">
              <div
                className={cn(
                  "flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-xs font-semibold",
                  isCurrent && "bg-primary text-primary-foreground",
                  isDone && "bg-primary/20 text-primary",
                  !isCurrent &&
                    !isDone &&
                    "bg-muted text-muted-foreground",
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
                  "hidden text-xs font-medium sm:inline",
                  isCurrent && "text-foreground",
                  isDone && "text-muted-foreground",
                  !isCurrent && !isDone && "text-muted-foreground/70",
                )}
              >
                {step.label}
              </span>
            </div>
            {!isLast && (
              <div
                className={cn(
                  "h-px flex-1",
                  isDone ? "bg-primary/30" : "bg-border",
                )}
                aria-hidden="true"
              />
            )}
          </li>
        );
      })}
    </ol>
  );
}
