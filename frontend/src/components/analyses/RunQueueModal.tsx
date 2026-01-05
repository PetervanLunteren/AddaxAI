/**
 * Run Queue Modal Component
 *
 * Blocking modal that shows progress while processing queue.
 * Connects to WebSocket for real-time progress updates.
 */

import { useState, useEffect } from "react";
import { Loader2, CheckCircle2, XCircle } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { useTaskProgress } from "@/hooks/useTaskProgress";

interface RunQueueModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  queueCount: number;
  jobIds: string[];
}

export function RunQueueModal({ open, onOpenChange, queueCount, jobIds }: RunQueueModalProps) {
  const [hasError, setHasError] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [isComplete, setIsComplete] = useState(false);

  // Reset state when modal closes
  useEffect(() => {
    if (!open) {
      setHasError(false);
      setErrorMessage("");
      setIsComplete(false);
    }
  }, [open]);

  // Auto-close modal shortly after success
  useEffect(() => {
    if (open && isComplete && !hasError) {
      const timer = setTimeout(() => {
        onOpenChange(false);
      }, 2000);
      return () => clearTimeout(timer);
    }
  }, [open, isComplete, hasError, onOpenChange]);

  // Track progress of the single job (sequential processing)
  // The backend processes all queue entries sequentially, we just track the one job
  const jobId = jobIds[0] || null;
  const { progress, message, isConnected } = useTaskProgress({
    taskId: jobId,
    onComplete: () => {
      setIsComplete(true);
    },
    onError: (msg) => {
      setHasError(true);
      setErrorMessage(msg);
    },
  });

  const hasJob = Boolean(jobId);

  // Calculate overall status
  const isWaitingForJob = !hasError && !isComplete && !hasJob;
  const isProcessing = !isComplete && !hasError && hasJob;
  const progressPercent = progress * 100;

  // Display message and icon
  let displayMessage = message || "Initializing...";
  let displayIcon = <Loader2 className="h-5 w-5 text-blue-600 animate-spin" />;

  if (hasError) {
    displayMessage = errorMessage || "Processing failed";
    displayIcon = <XCircle className="h-5 w-5 text-red-600" />;
  } else if (isComplete) {
    displayMessage = `Queue processing complete! Processed ${queueCount} deployment${queueCount > 1 ? 's' : ''}.`;
    displayIcon = <CheckCircle2 className="h-5 w-5 text-green-600" />;
  } else if (isWaitingForJob) {
    displayMessage = "Starting queue job...";
  } else if (isProcessing && message) {
    displayMessage = message;
  }

  // Debug logging to see what's being rendered with timestamp
  const timestamp = new Date().toISOString().split('T')[1].slice(0, -1); // HH:MM:SS.mmm
  console.log(`[${timestamp}] [RunQueueModal] Render - isProcessing: ${isProcessing}, isComplete: ${isComplete}, message: "${message}", progress: ${progress}`);
  console.log(`[${timestamp}] [RunQueueModal] Displaying: "${displayMessage}" at ${progressPercent.toFixed(1)}%`);

  return (
    <Dialog open={open} onOpenChange={isComplete || hasError ? onOpenChange : undefined}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Processing queue</DialogTitle>
          <DialogDescription>
            {isComplete
              ? "All deployments have been processed successfully."
              : isWaitingForJob
                ? "Preparing the deployment queue..."
                : `Processing ${queueCount} deployment${queueCount > 1 ? 's' : ''} sequentially...`}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          {/* Status */}
          <div className="flex items-center gap-3">
            {displayIcon}
            <span className="text-sm font-medium">{displayMessage}</span>
          </div>

          {/* Progress bar */}
          {(isProcessing || isWaitingForJob) && (
            <div className="space-y-2">
              <Progress value={isProcessing ? progressPercent : undefined} className="h-2" />
              <p className="text-xs text-gray-500 text-center">
                {isProcessing ? `${progressPercent.toFixed(0)}%` : "Waiting for job to start..."}
              </p>
            </div>
          )}

          {/* Connection status */}
          {isProcessing && !isConnected && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
              <p className="text-xs text-yellow-800">
                <strong>Connecting to progress updates...</strong>
              </p>
            </div>
          )}
        </div>

        <DialogFooter>
          {isComplete || hasError ? (
            <Button onClick={() => onOpenChange(false)}>Close</Button>
          ) : null}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
