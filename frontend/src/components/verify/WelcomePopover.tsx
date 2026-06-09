/**
 * One-time welcome popover shown on first visit to the event detail modal.
 * Introduces the verification workflow and mentions keyboard shortcuts.
 */

import { CircleHelp } from "lucide-react";
import { Button } from "../ui/button";

interface WelcomePopoverProps {
  open: boolean;
  onDismiss: () => void;
}

export function WelcomePopover({ open, onDismiss }: WelcomePopoverProps) {
  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
      <div className="bg-white rounded-lg shadow-xl max-w-md mx-4 p-6 space-y-4">
        <h2 className="text-lg font-semibold">Welcome to the counts page</h2>
        <div className="space-y-3 text-sm text-muted-foreground">
          <p>
            An event groups files captured close together in time, treated as
            one observation. Each card opens to the MaxN frame: the moment
            when the most animals were visible, so you can confirm the
            species and count in one shot.
          </p>
          <p>
            The strip below the image is the rest of the event. Click any
            frame to inspect it.
          </p>
          <p>
            Press <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Enter</code> to confirm and jump to the next unconfirmed event. Click <CircleHelp className="inline h-3.5 w-3.5 align-text-bottom" /> in the toolbar once dismissed for the full guide and keyboard shortcuts.
          </p>
        </div>
        <div className="flex justify-end">
          <Button onClick={onDismiss}>Got it</Button>
        </div>
      </div>
    </div>
  );
}
