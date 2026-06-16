/**
 * Keyboard shortcuts popover for the Labels tab.
 *
 * Renders its own toolbar icon trigger (Keyboard) so it sits inline
 * with the other utility icons in the verify toolbar. Lists the grid
 * shortcuts on the left and the user-configurable label slots (1-5)
 * on the right.
 */

import { Keyboard } from "lucide-react";

import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover";
import { LabelPicker } from "./LabelPicker";
import type { LabelOption } from "../../hooks/useLabelOptions";

interface LabelsKeyboardPopoverProps {
  shortcutLabels: Record<number, LabelOption>;
  onShortcutLabelsChange: (
    updater: (prev: Record<number, LabelOption>) => Record<number, LabelOption>,
  ) => void;
  labelOptions: LabelOption[];
  labelOptionsLoading: boolean;
  projectId: string;
}

const GRID_SHORTCUTS: ReadonlyArray<readonly [string, string]> = [
  ["Click", "Select"],
  [navigator.platform.includes("Mac") ? "Cmd + Click" : "Ctrl + Click", "Toggle select"],
  ["Shift + Click", "Extend range"],
  ["Double-click", "Open detail"],
  ["Click outside", "Deselect all"],
  ["Enter", "Verify selected"],
  ["X", "Mark false detection"],
  ["R", "Relabel selected"],
  ["M", "Relabel to most common in selection"],
  [navigator.platform.includes("Mac") ? "Cmd + A" : "Ctrl + A", "Select all"],
  ["Esc", "Deselect / close"],
];

function ShortcutKey({ keys }: { keys: string }) {
  const parts = keys.split("+").map((p) => p.trim());
  return (
    <code className="bg-zinc-100 text-zinc-500 px-1.5 py-0.5 rounded text-[11px] w-24 shrink-0 text-center whitespace-nowrap">
      {parts.map((part, i) => (
        <span key={i}>
          {part}
          {i < parts.length - 1 && <span className="text-[#bbbbc1]"> + </span>}
        </span>
      ))}
    </code>
  );
}

export function LabelsKeyboardPopover({
  shortcutLabels,
  onShortcutLabelsChange,
  labelOptions,
  labelOptionsLoading,
  projectId,
}: LabelsKeyboardPopoverProps) {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          title="Keyboard shortcuts"
          aria-label="Keyboard shortcuts"
          className="text-muted-foreground hover:text-foreground transition-colors"
        >
          <Keyboard className="h-4 w-4" />
        </button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-auto px-4 py-3">
        <div className="flex gap-8">
          <div>
            {GRID_SHORTCUTS.map(([key, action]) => (
              <div key={key} className="flex items-center text-xs gap-3 h-7">
                <ShortcutKey keys={key} />
                <span>{action}</span>
              </div>
            ))}
          </div>
          <div>
            {[1, 2, 3, 4, 5].map((n) => (
              <div key={n} className="flex items-center text-xs gap-3 h-7">
                <ShortcutKey keys={String(n)} />
                <span>Change selected to</span>
                <LabelPicker
                  value={shortcutLabels[n]?.value ?? null}
                  onSelect={(option) =>
                    onShortcutLabelsChange((prev) => ({ ...prev, [n]: option }))
                  }
                  options={labelOptions}
                  isLoading={labelOptionsLoading}
                  projectId={projectId}
                />
              </div>
            ))}
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}
