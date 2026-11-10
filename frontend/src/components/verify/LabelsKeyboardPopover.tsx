/**
 * Keyboard shortcuts popover for the Labels page, both tabs.
 *
 * Renders its own toolbar icon trigger (Keyboard) so it sits inline
 * with the other utility icons in the verify toolbar. Takes the list of
 * shortcuts to show, because Detections and Files have different ones,
 * and renders the user-configurable number-key label slots alongside.
 * Both tabs pass them: the Files grid answers the same keys, and a
 * keypad user who works there could not find where to set them.
 *
 * Each list holds the grid's keys and then its viewer's, so every key a
 * person can press on the tab is in one place; the buttons that have a
 * key also show it. One row per key: rows are keyed by it.
 */

import { Keyboard } from "lucide-react";

import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover";
import { VERIFY_TOOLBAR_ICON_CLASS } from "./VerifyToolbar";
import { LabelPicker } from "./LabelPicker";
import type { LabelOption } from "../../hooks/useLabelOptions";
import { SHORTCUT_SLOTS } from "../../hooks/useShortcutLabels";
import type { Shortcut } from "./shortcuts";

interface LabelSlots {
  shortcutLabels: Record<number, LabelOption>;
  onShortcutLabelsChange: (
    updater: (prev: Record<number, LabelOption>) => Record<number, LabelOption>,
  ) => void;
  labelOptions: LabelOption[];
  labelOptionsLoading: boolean;
  projectId: string;
}

interface LabelsKeyboardPopoverProps {
  shortcuts: readonly Shortcut[];
  /** Closing line, e.g. what happens to the selection after an action. */
  footer: string;
  /** The configurable number-key label slots. Omitted by tabs without labels. */
  labelSlots?: LabelSlots;
}

/** One key of a shortcut list, the same chip in every list (the Labels
 *  popovers and the Counts modal's). Wide enough for the longest key,
 *  "Tab / Shift + Tab". */
export function ShortcutKey({ keys }: { keys: string }) {
  const parts = keys.split("+").map((p) => p.trim());
  return (
    <code className="bg-muted text-muted-foreground px-1.5 py-0.5 rounded text-[11px] w-32 shrink-0 text-center whitespace-nowrap">
      {parts.map((part, i) => (
        <span key={i}>
          {part}
          {i < parts.length - 1 && <span className="text-muted-foreground/60"> + </span>}
        </span>
      ))}
    </code>
  );
}

export function LabelsKeyboardPopover({
  shortcuts,
  footer,
  labelSlots,
}: LabelsKeyboardPopoverProps) {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          title="Keyboard shortcuts"
          aria-label="Keyboard shortcuts"
          className={VERIFY_TOOLBAR_ICON_CLASS}
        >
          <Keyboard className="h-4 w-4" />
        </button>
      </PopoverTrigger>
      {/* Capped at the room Radix measures below the trigger, and
          scrolling past it: the Files list is taller than a laptop
          screen leaves, and its last rows used to sit off the bottom. */}
      <PopoverContent
        align="end"
        className="scroll-shade w-auto max-h-[var(--radix-popover-content-available-height)] overflow-y-auto px-4 py-3"
      >
        <div className="flex gap-8">
          <div>
            {shortcuts.map(([key, action]) => (
              <div key={key} className="flex items-center text-xs gap-3 h-7">
                <ShortcutKey keys={key} />
                <span>{action}</span>
              </div>
            ))}
          </div>
          {labelSlots && (
            <div>
              {SHORTCUT_SLOTS.map((n) => (
                <div key={n} className="flex items-center text-xs gap-3 h-7">
                  <ShortcutKey keys={String(n)} />
                  <span>Change selected to</span>
                  <LabelPicker
                    value={labelSlots.shortcutLabels[n]?.value ?? null}
                    onSelect={(option) =>
                      labelSlots.onShortcutLabelsChange((prev) => ({
                        ...prev,
                        [n]: option,
                      }))
                    }
                    options={labelSlots.labelOptions}
                    isLoading={labelSlots.labelOptionsLoading}
                    projectId={labelSlots.projectId}
                  />
                </div>
              ))}
            </div>
          )}
        </div>
        <p className="mt-3 text-[11px] text-muted-foreground">{footer}</p>
      </PopoverContent>
    </Popover>
  );
}
