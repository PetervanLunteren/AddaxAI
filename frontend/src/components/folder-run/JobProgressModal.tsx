/**
 * Blocking progress modal for folder-run jobs.
 *
 * Wraps a job's progress UI (passed in as ``children``) in a dialog
 * the user can't dismiss except by hitting Cancel. Same shell for
 * both the Analysis step (Run) and the Save step (Output), so the
 * two long-running operations feel like the same pattern.
 *
 * Rules:
 * - Not dismissable by clicking the backdrop or pressing ESC
 * - ``beforeunload`` warning while the modal is open
 * - Only the Cancel button is interactive; Cancel switches to a
 *   muted "Cancelling..." state until the worker actually stops
 *   and the parent unmounts the modal
 */

import { useEffect } from "react";
import { Loader2, X } from "lucide-react";

import { Button } from "../ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";

interface JobProgressModalProps {
  open: boolean;
  title: string;
  isCancelling: boolean;
  onCancel: () => void;
  children: React.ReactNode;
}

export function JobProgressModal({
  open,
  title,
  isCancelling,
  onCancel,
  children,
}: JobProgressModalProps) {
  // Warn the user before closing the tab while the job is running.
  // Without this, a refresh or accidental close would silently leave
  // the worker running with no UI to come back to.
  useEffect(() => {
    if (!open) return;
    const handler = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [open]);

  return (
    <Dialog open={open}>
      <DialogContent className="max-w-lg" nonDismissable>
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            {isCancelling ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : null}
            {title}
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-3 py-2">{children}</div>

        <div className="flex justify-end">
          <Button
            variant="outline"
            size="sm"
            onClick={onCancel}
            disabled={isCancelling}
            className="gap-2"
          >
            <X className="h-4 w-4" />
            {isCancelling ? "Cancelling..." : "Cancel"}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
