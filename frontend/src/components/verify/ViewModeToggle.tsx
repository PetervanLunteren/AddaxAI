/**
 * Detections / Empties switch for the Labels page.
 *
 * The two halves of one job. Detections shows every detection above the
 * detection threshold, one card per box. Empties shows every photo with
 * none, one card per photo. Every photo in the project is in exactly
 * one of them, and the threshold decides which.
 *
 * Same segmented look as `TileSizeToggle`; not shared with it because
 * that one is typed to tile sizes and a generic control for two call
 * sites would be more indirection than it saves.
 */

import { compactNumber } from "../../lib/compact-number";
import { cn } from "../../lib/utils";

export type LabelsViewMode = "crops" | "empties";

interface ViewModeToggleProps {
  value: LabelsViewMode;
  onChange: (v: LabelsViewMode) => void;
  /** Labels not yet checked in each tab. The chip is what stops the
   *  other half of the work being invisible; it is dropped at zero so a
   *  finished tab reads clean rather than showing a nought. */
  cropsLeft?: number;
  emptiesLeft?: number;
}

export function ViewModeToggle({
  value,
  onChange,
  cropsLeft = 0,
  emptiesLeft = 0,
}: ViewModeToggleProps) {
  const modes: {
    value: LabelsViewMode;
    label: string;
    left: number;
  }[] = [
    { value: "crops", label: "Detections", left: cropsLeft },
    { value: "empties", label: "Empties", left: emptiesLeft },
  ];

  return (
    <div className="flex rounded-lg bg-muted p-0.5" role="tablist">
      {modes.map((mode) => {
        const active = value === mode.value;
        return (
          <button
            key={mode.value}
            type="button"
            role="tab"
            aria-selected={active}
            className={cn(
              "flex items-center gap-1.5 px-3 py-1 text-xs font-medium",
              "rounded-md transition-colors",
              active
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground",
            )}
            onClick={() => onChange(mode.value)}
          >
            {mode.label}
            {mode.left > 0 && (
              <span
                title={`${mode.left.toLocaleString()} not verified yet`}
                className={cn(
                  "rounded px-1 py-px text-[10px] font-normal tabular-nums",
                  active
                    ? "bg-muted text-muted-foreground"
                    : "bg-background/60 text-muted-foreground",
                )}
              >
                {compactNumber(mode.left)}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
