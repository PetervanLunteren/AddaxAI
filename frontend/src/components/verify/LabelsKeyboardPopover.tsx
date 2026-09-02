/**
 * Keyboard shortcuts popover for the Labels page, both tabs.
 *
 * Renders its own toolbar icon trigger (Keyboard) so it sits inline
 * with the other utility icons in the verify toolbar. Takes the list of
 * shortcuts to show, because Detections and Files have different ones, and
 * renders the user-configurable label slots (1-5) alongside only when a
 * tab has them. Files does not: labels there are set on the box, in the viewer.
 *
 * Only grid shortcuts are listed, in both tabs. The detail views label
 * their own keys on the buttons that use them, so repeating them here
 * would be a second place to keep in step for no gain.
 */

import { Keyboard } from "lucide-react";

import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover";
import { VERIFY_TOOLBAR_ICON_CLASS } from "./VerifyToolbar";
import { LabelPicker } from "./LabelPicker";
import type { LabelOption } from "../../hooks/useLabelOptions";
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
  /** The configurable 1-5 label slots. Omitted by tabs without labels. */
  labelSlots?: LabelSlots;
}

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
      <PopoverContent align="end" className="w-auto px-4 py-3">
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
              {[1, 2, 3, 4, 5].map((n) => (
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
