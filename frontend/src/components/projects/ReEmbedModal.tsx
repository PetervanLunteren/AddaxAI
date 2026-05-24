/**
 * Re-Embed Modal Component
 *
 * Blocking modal that shows progress while re-embedding detections.
 * Simplified version of RunQueueModal for single-phase (embedding) jobs.
 */

import { useState, useEffect } from "react";
import { Loader2, CheckCircle2, XCircle, Clock } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { useTaskProgress } from "@/hooks/useTaskProgress";
import { humanizeTqdmTime } from "@/lib/duration";

interface ReEmbedModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  jobId: string | null;
  onComplete?: () => void;
  onError?: (message: string) => void;
}

export function ReEmbedModal({ open, onOpenChange, jobId, onComplete, onError }: ReEmbedModalProps) {
  const [hasError, setHasError] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [isComplete, setIsComplete] = useState(false);

  // Reset state when modal opens/closes
  useEffect(() => {
    setHasError(false);
    setErrorMessage("");
    setIsComplete(false);
  }, [open]);

  // Reset state when a new job starts
  useEffect(() => {
    if (jobId) {
      setIsComplete(false);
      setHasError(false);
      setErrorMessage("");
    }
  }, [jobId]);

  // Auto-close modal shortly after success
  useEffect(() => {
    if (open && isComplete && !hasError) {
      const timer = setTimeout(() => {
        onOpenChange(false);
      }, 2000);
      return () => clearTimeout(timer);
    }
  }, [open, isComplete, hasError, onOpenChange]);

  const { message, phase, metrics, computeDevice } = useTaskProgress({
    taskId: jobId,
    onComplete: () => {
      setIsComplete(true);
      onComplete?.();
    },
    onError: (msg) => {
      setHasError(true);
      setErrorMessage(msg);
      onError?.(msg);
    },
  });

  const hasJob = Boolean(jobId);
  const isWaitingForJob = !hasError && !isComplete && !hasJob;
  const isProcessing = !isComplete && !hasError && hasJob;

  // Calculate embedding progress from tqdm metrics
  const getEmbeddingProgress = (): number => {
    if (isComplete) return 100;
    if (phase !== "embedding") return 0;
    if (metrics?.current !== undefined && metrics?.total !== undefined && metrics.total > 0) {
      return (metrics.current / metrics.total) * 100;
    }
    return 0;
  };

  const embeddingProgress = getEmbeddingProgress();

  const getEmbeddingStatus = (): string => {
    if (isComplete) return "Complete";
    if (phase !== "embedding") return "Waiting...";
    if (metrics?.current !== undefined && metrics?.total !== undefined && metrics.current >= metrics.total) {
      return "Complete";
    }
    return "Starting up...";
  };

  const embeddingStatus = getEmbeddingStatus();

  const renderStatusWithIcon = (status: string) => {
    if (status === "Waiting...") {
      return (
        <div className="flex items-center gap-2">
          <Clock className="h-3.5 w-3.5" style={{ color: '#156065' }} />
          <span>{status}</span>
        </div>
      );
    } else if (status === "Starting up...") {
      return (
        <div className="flex items-center gap-2">
          <Loader2 className="h-3.5 w-3.5 animate-spin" style={{ color: '#156065' }} />
          <span>{status}</span>
        </div>
      );
    } else if (status === "Complete") {
      return (
        <div className="flex items-center gap-2">
          <CheckCircle2 className="h-3.5 w-3.5" style={{ color: '#156065' }} />
          <span>{status}</span>
        </div>
      );
    }
    return <span>{status}</span>;
  };

  const showInfoCard = phase === "embedding" &&
    metrics?.current !== undefined && metrics?.total !== undefined &&
    metrics.current < metrics.total;

  return (
    <Dialog open={open} onOpenChange={isComplete || hasError ? onOpenChange : undefined}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>
            {isComplete ? "Re-embedding complete" : "Re-embedding observations"}
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-4 py-2">
          {/* Error State */}
          {hasError && (
            <div className="flex items-center gap-3">
              <XCircle className="h-5 w-5 text-red-600" />
              <span className="text-sm font-medium text-red-600">{errorMessage || "Re-embedding failed"}</span>
            </div>
          )}

          {/* Complete State */}
          {isComplete && !hasError && (
            <div className="flex items-center gap-3">
              <CheckCircle2 className="h-5 w-5" style={{ color: '#156065' }} />
              <span className="text-sm font-medium" style={{ color: '#156065' }}>
                {message || "Re-embedding complete!"}
              </span>
            </div>
          )}

          {/* Processing States */}
          {!isComplete && !hasError && (
            <>
              {/* Spinner while waiting for job */}
              {isWaitingForJob && (
                <div className="flex items-center gap-3">
                  <Loader2 className="h-5 w-5 animate-spin" style={{ color: '#0f6064' }} />
                  <span className="text-sm font-medium">{message || "Initializing..."}</span>
                </div>
              )}

              {/* Embedding progress card. Re-embedding is logically a
                  single operation regardless of how many deployments
                  the worker iterates internally, so we show only the
                  crop-level progress, not the deployment counter. */}
              {isProcessing && (
                <div className="border rounded-lg p-4 space-y-2">
                  <div className="flex items-center justify-between">
                    <p className="text-xs font-medium text-gray-700">Embedding</p>
                    <span className="text-xs text-gray-500 font-mono">{embeddingProgress.toFixed(0)}%</span>
                  </div>
                  <Progress value={embeddingProgress} className="h-2" />

                  {showInfoCard ? (
                    <div className="text-[11px] space-y-0.5 rounded-md bg-gray-50 p-3 font-mono text-gray-600">
                      <div className="flex justify-between">
                        <span>Processing {metrics.unit || 'crops'}:</span>
                        <span>{metrics.current?.toLocaleString()} of {metrics.total?.toLocaleString()}</span>
                      </div>
                      {metrics.elapsed && (
                        <div className="flex justify-between">
                          <span>Elapsed time:</span>
                          <span>{humanizeTqdmTime(metrics.elapsed)}</span>
                        </div>
                      )}
                      {metrics.remaining && (
                        <div className="flex justify-between">
                          <span>Remaining time:</span>
                          <span>{humanizeTqdmTime(metrics.remaining, true)}</span>
                        </div>
                      )}
                      {metrics.rate && metrics.unit && (
                        <div className="flex justify-between">
                          <span>{metrics.unit.charAt(0).toUpperCase() + metrics.unit.slice(1)} per second:</span>
                          <span>{metrics.rate.toFixed(1)}</span>
                        </div>
                      )}
                      <div className="flex justify-between">
                        <span>Running on:</span>
                        <span className={computeDevice ? "" : "text-gray-400"}>
                          {computeDevice ?? "detecting..."}
                        </span>
                      </div>
                    </div>
                  ) : (
                    <div className="text-[11px] rounded-md bg-gray-50 p-3 font-mono text-gray-600">
                      {renderStatusWithIcon(embeddingStatus)}
                    </div>
                  )}
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
