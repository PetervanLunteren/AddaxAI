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
        <h2 className="text-lg font-semibold">Check the AI's counts</h2>
        <div className="space-y-3 text-sm text-muted-foreground">
          <p>
            The AI already counted the species in each event. Here you can
            check those counts and confirm them. It is optional, but the AI
            makes mistakes, so confirming the important ones makes your data
            more reliable.
          </p>
          <p>
            An event groups files captured close together in time, treated as
            one observation. Each card opens to the MaxN frame: the moment the
            most animals are visible, so you can check the species and count in
            one look. The strip below is the rest of the event; for a video it
            shows frames across the clip. Click any to inspect.
          </p>
          <p>
            Press <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Enter</code> to confirm and jump to the next unconfirmed event. Click <CircleHelp className="inline h-3.5 w-3.5 align-text-bottom" /> in the toolbar any time for the full guide and keyboard shortcuts.
          </p>
        </div>
        <div className="flex justify-end">
          <Button onClick={onDismiss}>Got it</Button>
        </div>
      </div>
    </div>
  );
}
