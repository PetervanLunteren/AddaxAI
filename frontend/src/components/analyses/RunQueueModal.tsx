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

  // Phase ordering for progress calculation
  const phaseOrder = ["init", "video_detection", "video_classification", "image_detection", "image_classification", "finalize"];
  const currentPhaseIndex = phase ? phaseOrder.indexOf(phase) : -1;

  // Calculate progress for each phase
  const getPhaseProgress = (targetPhase: string): number => {
    const targetIndex = phaseOrder.indexOf(targetPhase);
    if (currentPhaseIndex < targetIndex) return 0; // Not started yet
    if (currentPhaseIndex > targetIndex) return 100; // Already completed
    return phase === targetPhase ? phaseProgressPercent : 0; // Currently active
  };

  const videoDetectionProgress = getPhaseProgress("video_detection");
  const videoClassificationProgress = getPhaseProgress("video_classification");
  const imageDetectionProgress = getPhaseProgress("image_detection");
  const imageClassificationProgress = getPhaseProgress("image_classification");

  // Determine status messages
  const getPhaseStatus = (targetPhase: string, defaultWaiting: string, messagePrefix: string): string => {
    const targetIndex = phaseOrder.indexOf(targetPhase);
    if (currentPhaseIndex < targetIndex) return defaultWaiting;
    if (currentPhaseIndex > targetIndex) return "Complete";
    return phase === targetPhase ? (message || "").replace(`${messagePrefix}: `, "") : defaultWaiting;
  };

  const videoDetectionStatus = getPhaseStatus("video_detection", "Waiting...", "Video detection");
  const videoClassificationStatus = getPhaseStatus("video_classification", "Waiting...", "Video classification");
  const imageDetectionStatus = getPhaseStatus("image_detection", "Waiting...", "Image detection");
  const imageClassificationStatus = getPhaseStatus("image_classification", "Waiting...", "Image classification");

  const showSpinner = phase === "init" || phase === "finalize" || isWaitingForJob;

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

          {/* Processing States - Show BOTH progress bars */}
          {!isComplete && !hasError && (
            <>
              {/* Spinner for init/finalize phases */}
              {showSpinner && (
                <div className="flex items-center gap-3">
                  <Loader2 className="h-5 w-5 text-blue-600 animate-spin" />
                  <span className="text-sm font-medium">{message || "Initializing..."}</span>
                </div>
              )}

              {/* Video Detection Progress Bar */}
              {!showSpinner && (
                <div className="space-y-2">
                  <p className="text-xs font-medium text-gray-700">Video Detection</p>
                  <Progress value={videoDetectionProgress} className="h-2" />
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">{videoDetectionStatus}</span>
                    <span className="text-xs text-gray-500">{videoDetectionProgress.toFixed(0)}%</span>
                  </div>
                </div>
              )}

              {/* Video Classification Progress Bar */}
              {!showSpinner && (
                <div className="space-y-2">
                  <p className="text-xs font-medium text-gray-700">Video Classification</p>
                  <Progress value={videoClassificationProgress} className="h-2" />
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">{videoClassificationStatus}</span>
                    <span className="text-xs text-gray-500">{videoClassificationProgress.toFixed(0)}%</span>
                  </div>
                </div>
              )}

              {/* Image Detection Progress Bar */}
              {!showSpinner && (
                <div className="space-y-2">
                  <p className="text-xs font-medium text-gray-700">Image Detection</p>
                  <Progress value={imageDetectionProgress} className="h-2" />
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">{imageDetectionStatus}</span>
                    <span className="text-xs text-gray-500">{imageDetectionProgress.toFixed(0)}%</span>
                  </div>
                </div>
              )}

              {/* Image Classification Progress Bar */}
              {!showSpinner && (
                <div className="space-y-2">
                  <p className="text-xs font-medium text-gray-700">Image Classification</p>
                  <Progress value={imageClassificationProgress} className="h-2" />
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">{imageClassificationStatus}</span>
                    <span className="text-xs text-gray-500">{imageClassificationProgress.toFixed(0)}%</span>
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
