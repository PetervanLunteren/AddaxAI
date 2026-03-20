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
        <h2 className="text-lg font-semibold">Welcome to Verification</h2>
        <div className="space-y-3 text-sm text-muted-foreground">
          <p>
            Each event opens to its MaxN frame, the image where the peak count
            for each species was observed. Confirm or correct the labels, then
            press <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Enter</code> to verify and advance.
          </p>
          <p>
            Click <CircleHelp className="inline h-3.5 w-3.5 align-text-bottom" /> in the toolbar for a full guide, or <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Show keyboard shortcuts</code> at the bottom of the sidebar.
          </p>
        </div>
        <div className="flex justify-end">
          <Button onClick={onDismiss}>Got it</Button>
        </div>
      </div>
    </div>
  );
}
