/**
 * One-time welcome popover shown on first visit to the Media tab.
 * Introduces the per-file verification workflow.
 */

import { CircleHelp } from "lucide-react";
import { Button } from "../ui/button";

interface MediaWelcomePopoverProps {
  open: boolean;
  onDismiss: () => void;
}

export function MediaWelcomePopover({ open, onDismiss }: MediaWelcomePopoverProps) {
  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
      <div className="bg-white rounded-lg shadow-xl max-w-md mx-4 p-6 space-y-4">
        <h2 className="text-lg font-semibold">Welcome to media verification</h2>
        <div className="space-y-3 text-sm text-muted-foreground">
          <p>
            Each tile is one file: a still photo or a video. Open a tile
            to verify or correct what the AI detected.
          </p>
          <p>
            Want to work by event instead? The Events tab groups files
            captured close together in time. The Observations tab is one
            tile per detection.
          </p>
          <p>
            Press <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Enter</code> to verify and jump to the next unverified file. Click <CircleHelp className="inline h-3.5 w-3.5 align-text-bottom" /> in the toolbar once dismissed for the full guide and keyboard shortcuts.
          </p>
        </div>
        <div className="flex justify-end">
          <Button onClick={onDismiss}>Got it</Button>
        </div>
      </div>
    </div>
  );
}
