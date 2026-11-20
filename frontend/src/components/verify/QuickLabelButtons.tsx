/**
 * The quick labels in a viewer's side panel: the project's number-key
 * slots (`useShortcutLabels`) as split buttons. The body applies the
 * slot's label (the number key does the same), the chevron segment
 * opens the label search that changes or removes it. Only the set slots show,
 * plus one row to fill the next free slot. Changes are saved on the
 * project, so every surface sees them at once.
 *
 * One component for the Files viewer and the Detection viewer, so the
 * two look and work the same. What applying a label means stays with
 * the viewer (`onApply`): the selected boxes of a file, or the one
 * detection on screen.
 */

import { useState } from "react";
import { Bookmark, ChevronDown, Plus } from "lucide-react";

import { Button } from "../ui/button";
import { Kbd } from "../ui/kbd";
import { LabelPicker } from "./LabelPicker";
import type { LabelOption } from "../../hooks/useLabelOptions";
import { SHORTCUT_SLOTS, useShortcutLabels } from "../../hooks/useShortcutLabels";
import { withoutSlot } from "../../lib/shortcut-slots";

interface QuickLabelButtonsProps {
  projectId: string;
  labelOptions: LabelOption[];
  labelOptionsLoading: boolean;
  /** What a slot's label would land on, for the tooltip: "this detection". */
  scope: string;
  /** Greys out the apply buttons; changing a slot stays possible. */
  disabled: boolean;
  onApply: (option: LabelOption) => void;
}

/** The remove row's text, the key drawn as on the buttons. One wording
 *  for the viewers and the shortcuts popup. */
export function RemoveQuickLabelText({ slot }: { slot: number }) {
  return (
    <>
      Remove quick label from key
      <Kbd>{String(slot)}</Kbd>
    </>
  );
}

export function QuickLabelButtons({
  projectId,
  labelOptions,
  labelOptionsLoading,
  scope,
  disabled,
  onApply,
}: QuickLabelButtonsProps) {
  const { shortcutLabels, updateShortcutLabels } = useShortcutLabels(projectId);
  // The slot whose label is being chosen (a chevron, or the add row).
  const [editSlot, setEditSlot] = useState<number | null>(null);
  const freeSlot = SHORTCUT_SLOTS.find((n) => !shortcutLabels[n]);

  return (
    <>
      {SHORTCUT_SLOTS
        .filter((n) => shortcutLabels[n])
        .map((n) => (
          <div key={n} className="flex w-full">
            <Button
              variant="outline"
              size="sm"
              className="min-w-0 flex-1 justify-center rounded-r-none border-r-0"
              disabled={disabled}
              onClick={() => onApply(shortcutLabels[n])}
              title={`Relabel ${scope} to ${shortcutLabels[n].displayName} and verify`}
            >
              <Bookmark className="h-4 w-4 mr-1" />
              <span className="truncate">Set to {shortcutLabels[n].displayName}</span>
              <Kbd>{String(n)}</Kbd>
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="w-9 shrink-0 rounded-l-none px-0 text-muted-foreground"
              onClick={() => setEditSlot(n)}
              title={`Choose the label for key ${n}`}
            >
              <ChevronDown className="h-3.5 w-3.5" />
            </Button>
          </div>
        ))}
      {freeSlot !== undefined && (
        <Button
          variant="outline"
          size="sm"
          className="w-full justify-center text-muted-foreground"
          onClick={() => setEditSlot(freeSlot)}
          title={`Save a label on key ${freeSlot}`}
        >
          <Plus className="h-4 w-4 mr-1" />
          Add a quick label
          <Kbd>{String(freeSlot)}</Kbd>
        </Button>
      )}

      {/* Headless: only the search dialog, portaled, so it takes no
          room among the buttons. */}
      <LabelPicker
        headless
        value={editSlot !== null ? shortcutLabels[editSlot]?.value ?? null : null}
        displayName={editSlot !== null ? shortcutLabels[editSlot]?.displayName : undefined}
        onSelect={(option) => {
          const n = editSlot;
          setEditSlot(null);
          if (n !== null) updateShortcutLabels((prev) => ({ ...prev, [n]: option }));
        }}
        // A set slot can also be emptied; its button then leaves the card.
        removeAction={
          editSlot !== null && shortcutLabels[editSlot]
            ? {
                label: <RemoveQuickLabelText slot={editSlot} />,
                onSelect: () => updateShortcutLabels((prev) => withoutSlot(prev, editSlot)),
              }
            : undefined
        }
        options={labelOptions}
        isLoading={labelOptionsLoading}
        forceOpen={editSlot !== null}
        onOpenChange={(open) => {
          if (!open) setEditSlot(null);
        }}
        projectId={projectId}
      />
    </>
  );
}
