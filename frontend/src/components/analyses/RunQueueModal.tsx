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
  const { progress, message, phase, phaseProgress, isConnected } = useTaskProgress({
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
  const phaseProgressPercent = (phaseProgress ?? 0) * 100;

  // Determine what to display based on phase
  const showSpinner = phase === "init" || phase === "finalize" || isWaitingForJob;
  const showDetectionBar = phase === "detection";
  const showClassificationBar = phase === "classification";

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
          {/* Error State */}
          {hasError && (
            <div className="flex items-center gap-3">
              <XCircle className="h-5 w-5 text-red-600" />
              <span className="text-sm font-medium text-red-600">{errorMessage || "Processing failed"}</span>
            </div>
          )}

          {/* Complete State */}
          {isComplete && !hasError && (
            <div className="flex items-center gap-3">
              <CheckCircle2 className="h-5 w-5 text-green-600" />
              <span className="text-sm font-medium text-green-600">
                Queue processing complete! Processed {queueCount} deployment{queueCount > 1 ? 's' : ''}.
              </span>
            </div>
          )}

          {/* Processing States */}
          {!isComplete && !hasError && (
            <>
              {/* Spinner phases (init, finalize, waiting) */}
              {showSpinner && (
                <div className="flex items-center gap-3">
                  <Loader2 className="h-5 w-5 text-blue-600 animate-spin" />
                  <span className="text-sm font-medium">{message || "Initializing..."}</span>
                </div>
              )}

              {/* Detection Phase */}
              {showDetectionBar && (
                <div className="space-y-2">
                  <p className="text-xs font-medium text-gray-700">Detection</p>
                  <Progress value={phaseProgressPercent} className="h-2" />
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">{message}</span>
                    <span className="text-xs text-gray-500">{phaseProgressPercent.toFixed(0)}%</span>
                  </div>
                </div>
              )}

              {/* Classification Phase */}
              {showClassificationBar && (
                <div className="space-y-2">
                  <p className="text-xs font-medium text-gray-700">Classification</p>
                  <Progress value={phaseProgressPercent} className="h-2" />
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">{message}</span>
                    <span className="text-xs text-gray-500">{phaseProgressPercent.toFixed(0)}%</span>
                  </div>
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
            </>
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
