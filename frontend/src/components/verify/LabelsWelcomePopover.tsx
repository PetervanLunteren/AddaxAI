/**
 * One-time welcome popover shown on first visit to the Labels tab.
 * Introduces the labels workflow and mentions the help button.
 */

import { CircleHelp } from "lucide-react";
import { Button } from "../ui/button";

interface LabelsWelcomePopoverProps {
  open: boolean;
  onDismiss: () => void;
}

export function LabelsWelcomePopover({ open, onDismiss }: LabelsWelcomePopoverProps) {
  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
      <div className="bg-white rounded-lg shadow-xl max-w-md mx-4 p-6 space-y-4">
        <h2 className="text-lg font-semibold">Welcome to label verification</h2>
        <div className="space-y-3 text-sm text-muted-foreground">
          <p>
            Each tile is a single AI detection. Tiles are sorted by visual
            similarity (not time), so look-alikes sit next to each other.
            That is what makes mislabels easy to spot.
          </p>
          <p>
            Click a tile to select it. Shift-click another to select the
            range between them. Verify or relabel the whole selection at
            once. Double-click a tile to open it for closer inspection.
          </p>
          <p>
            Click <CircleHelp className="inline h-3.5 w-3.5 align-text-bottom" /> in
            the toolbar once dismissed for the full guide and keyboard
            shortcuts.
          </p>
        </div>
        <div className="flex justify-end">
          <Button onClick={onDismiss}>Got it</Button>
        </div>
      </div>
    </div>
  );
}
