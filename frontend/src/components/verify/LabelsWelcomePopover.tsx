/**
 * One-time welcome popover shown on first visit to the Labels tab.
 * Introduces the labels workflow and mentions the help button.
 */

import { CircleHelp, ExternalLink } from "lucide-react";
import { Button } from "../ui/button";

const GUIDE_URL = "https://docs.addaxai.com/docs/guides/check-labels/";

interface LabelsWelcomePopoverProps {
  open: boolean;
  onDismiss: () => void;
}

export function LabelsWelcomePopover({ open, onDismiss }: LabelsWelcomePopoverProps) {
  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
      <div className="bg-white rounded-lg shadow-xl max-w-md mx-4 p-6 space-y-4">
        <h2 className="text-lg font-semibold">Check the AI's labels</h2>
        <div className="space-y-3 text-sm text-muted-foreground">
          <p>
            The AI already labelled every detection. Here you can check those
            labels and fix any that are wrong. It is optional, but the AI
            makes mistakes, so a quick pass makes your data more reliable.
          </p>
          <p>
            Each tile is one detection, sorted by visual similarity (not time),
            so look-alikes sit next to each other and wrong labels stand out.
            Click a tile to select it, shift-click another for the range
            between, then verify or relabel the whole selection. Double-click a
            tile to open it.
          </p>
          <p>
            Click <CircleHelp className="inline h-3.5 w-3.5 align-text-bottom" /> in
            the toolbar any time to open the full guide.
          </p>
        </div>
        <div className="flex items-center justify-between">
          <a
            href={GUIDE_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-primary underline hover:opacity-80 inline-flex items-center gap-1"
          >
            Watch the video tutorial
            <ExternalLink className="h-3 w-3" />
          </a>
          <Button onClick={onDismiss}>Got it</Button>
        </div>
      </div>
    </div>
  );
}
