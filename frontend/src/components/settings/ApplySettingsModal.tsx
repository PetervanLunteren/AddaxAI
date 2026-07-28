/**
 * Blocking "Applying settings" progress dialog shown while a settings
 * PATCH + reprocess job runs. Extracted from the project Settings page
 * so the folder-run Labels step's analysis panel shows the identical
 * modal (one source of truth for the reprocess UX).
 */

import { RefreshCw } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "../ui/dialog";

export function ApplySettingsModal({
  open,
  message,
  progress,
  fallbackMessage = "Starting...",
}: {
  open: boolean;
  /** Latest progress message from the job's WebSocket, if any. */
  message: string | null | undefined;
  /** Job progress in [0, 1]; the bar hides at 0 and 1. */
  progress: number;
  /** Shown before the job has produced any message. */
  fallbackMessage?: string;
}) {
  return (
    <Dialog open={open}>
      <DialogContent
        className="max-w-md"
        onInteractOutside={(e) => e.preventDefault()}
      >
        <DialogTitle className="sr-only">Applying settings</DialogTitle>
        <DialogDescription className="sr-only">
          AddaxAI is applying your settings. This may take a moment.
        </DialogDescription>
        <div className="flex flex-col items-center gap-4 py-4">
          <div className="rounded-full bg-primary/10 p-3">
            <RefreshCw className="h-6 w-6 text-primary animate-spin" />
          </div>
          <div className="text-center space-y-2">
            <h3 className="font-semibold text-lg">Applying settings</h3>
            <p className="text-sm text-muted-foreground">
              {message || fallbackMessage}
            </p>
          </div>
          {progress > 0 && progress < 1 && (
            <div className="w-full space-y-1">
              <div className="h-2 w-full bg-secondary rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary rounded-full transition-all duration-300"
                  style={{ width: `${Math.round(progress * 100)}%` }}
                />
              </div>
              <p className="text-xs text-muted-foreground text-center">
                {Math.round(progress * 100)}%
              </p>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
