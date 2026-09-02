/**
 * Detections / Files switch for the Labels page.
 *
 * Two views of one job. Detections shows every detection above the
 * detection threshold, one card per box. Files shows every file, one
 * card per file, with its boxes on it. They overlap on purpose: a box
 * verified in Detections is one step towards its file, and a file
 * signed off in Files takes its boxes out of Detections. The chips
 * count different units, boxes and files.
 *
 * Same segmented look as `TileSizeToggle`; not shared with it because
 * that one is typed to tile sizes and a generic control for two call
 * sites would be more indirection than it saves.
 */

import { compactNumber } from "../../lib/compact-number";
import { cn } from "../../lib/utils";

export type LabelsViewMode = "crops" | "files";

interface ViewModeToggleProps {
  value: LabelsViewMode;
  onChange: (v: LabelsViewMode) => void;
  /** Work left in each tab. The chip is what stops the other view of
   *  the work being invisible; it is dropped at zero so a finished tab
   *  reads clean rather than showing a nought. */
  cropsLeft?: number;
  filesLeft?: number;
}

export function ViewModeToggle({
  value,
  onChange,
  cropsLeft = 0,
  filesLeft = 0,
}: ViewModeToggleProps) {
  const modes: {
    value: LabelsViewMode;
    label: string;
    left: number;
    unit: string;
  }[] = [
    { value: "crops", label: "Detections", left: cropsLeft, unit: "boxes" },
    { value: "files", label: "Files", left: filesLeft, unit: "files" },
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
                title={`${mode.left.toLocaleString()} ${mode.unit} not verified yet`}
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
