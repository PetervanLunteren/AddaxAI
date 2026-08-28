/**
 * One settings row: label + caption on the left, control on the right.
 *
 * The single source of this layout. It was previously hand-copied in four
 * places (the folder-run setup step, the project Settings page, BatchSizeRow
 * and ImageSizeRow), which is exactly the drift CONVENTIONS.md warns about:
 * a tweak had to be made four times or the pages diverged.
 *
 * `isCustom` marks the row as holding a non-default value. Analysis settings
 * persist across runs by design (lib/folderRunSettings: run 2 starts identical
 * to run 1), so a value changed once for a test quietly governs every later
 * run. The chip names which row that happened to; the collapsed section header
 * carries the count. Callers derive it from `advancedNonDefaultKeys`
 * (lib/advancedSettingsDefaults) so "what counts as default" has one
 * definition.
 */

import type { ReactNode } from "react";

import { FormDescription, FormLabel } from "../ui/form";
import { Badge } from "../ui/badge";
import { cn } from "@/lib/utils";

export function SettingRow({
  label,
  description,
  isCustom = false,
  disabled = false,
  children,
}: {
  label: string;
  /** ReactNode, not string: some captions append a clause inline. */
  description: ReactNode;
  /** True when this setting differs from its factory default. */
  isCustom?: boolean;
  /** Greys the row and disables its controls. The caption should say why. */
  disabled?: boolean;
  children: ReactNode;
}) {
  return (
    <div
      className={cn(
        "grid grid-cols-2 items-center gap-8 py-6",
        // pointer-events-none as well as the fieldset: Radix triggers and
        // the "Model details" link do not obey a disabled fieldset.
        disabled && "opacity-50 pointer-events-none",
      )}
      aria-disabled={disabled || undefined}
    >
      <div className="space-y-1">
        <div className="flex items-center gap-2">
          <FormLabel>{label}</FormLabel>
          {isCustom && (
            <Badge variant="secondary" className="font-normal">
              Custom
            </Badge>
          )}
        </div>
        <FormDescription className="text-sm">{description}</FormDescription>
      </div>
      {/* A disabled fieldset disables every native control inside it,
          including the Radix select trigger (a button), so callers do
          not have to thread `disabled` into each control. */}
      <fieldset disabled={disabled} className="min-w-0 space-y-2">
        {children}
      </fieldset>
    </div>
  );
}
