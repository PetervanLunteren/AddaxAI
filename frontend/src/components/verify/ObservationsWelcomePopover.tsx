/**
 * One-time welcome popover shown on first visit to the Observations tab.
 * Introduces the observations workflow and mentions the help button.
 */

import { CircleHelp } from "lucide-react";
import { Button } from "../ui/button";

interface ObservationsWelcomePopoverProps {
  open: boolean;
  onDismiss: () => void;
}

export function ObservationsWelcomePopover({ open, onDismiss }: ObservationsWelcomePopoverProps) {
  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
      <div className="bg-white rounded-lg shadow-xl max-w-md mx-4 p-6 space-y-4">
        <h2 className="text-lg font-semibold">Welcome to observations verification</h2>
        <div className="space-y-3 text-sm text-muted-foreground">
          <p>
            Observations are sorted by visual similarity so similar-looking
            crops appear together. This makes it easy to spot mislabels and
            verify in bulk: click to select, click again to select a range,
            then verify or relabel.
          </p>
          <p>
            Click <CircleHelp className="inline h-3.5 w-3.5 align-text-bottom" /> in
            the toolbar for a full guide covering search mode, suspicious
            labels, keyboard shortcuts, and more.
          </p>
        </div>
        <div className="flex justify-end">
          <Button onClick={onDismiss}>Got it</Button>
        </div>
      </div>
    </div>
  );
}
