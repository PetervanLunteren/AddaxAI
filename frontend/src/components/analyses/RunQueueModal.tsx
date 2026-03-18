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
import { Separator } from "@/components/ui/separator";
import { useTaskProgress, type TqdmMetrics } from "@/hooks/useTaskProgress";

interface PhaseRowProps {
  label: string;
  phaseName: string;
  progress: number;
  currentPhase: string | null;
  phaseProgress: number | undefined;
  metrics: TqdmMetrics | null;
  computeDevice: string | null;
}

function PhaseRow({ label, phaseName, progress, currentPhase, phaseProgress, metrics, computeDevice }: PhaseRowProps) {
  const isActive = phaseName === currentPhase;
  const hasValidMetrics = isActive && metrics?.current !== undefined && metrics?.total !== undefined && metrics.current < metrics.total;
  const isStartingUp = isActive && !hasValidMetrics && (phaseProgress === undefined || phaseProgress < 1.0);

  const unit = metrics?.unit || "items";
  const capitalizedUnit = unit.charAt(0).toUpperCase() + unit.slice(1);

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <p className="text-xs font-medium text-gray-700">{label}</p>
        <span className="text-xs text-gray-500 font-mono">{progress.toFixed(0)}%</span>
      </div>
      <Progress value={progress} className="h-2" />

      {hasValidMetrics && metrics && (
        <div className="text-[11px] space-y-0.5 rounded-md bg-gray-50 p-3 font-mono text-gray-600">
          <div className="flex justify-between">
            <span>Processing {unit}:</span>
            <span>{metrics.current} of {metrics.total}</span>
          </div>
          {metrics.elapsed && (
            <div className="flex justify-between">
              <span>Elapsed time:</span>
              <span>{metrics.elapsed}</span>
            </div>
          )}
          {metrics.remaining && (
            <div className="flex justify-between">
              <span>Remaining time:</span>
              <span>{metrics.remaining}</span>
            </div>
          )}
          {metrics.rate && (
            <div className="flex justify-between">
              <span>{capitalizedUnit} per second:</span>
              <span>{metrics.rate.toFixed(2)}</span>
            </div>
          )}
          <div className="flex justify-between">
            <span>Running on:</span>
            <span className={computeDevice ? "" : "text-gray-400"}>
              {computeDevice ?? "detecting..."}
            </span>
          </div>
        </div>
      )}

      {isStartingUp && !hasValidMetrics && (
        <div className="flex items-center gap-2 text-[11px] font-mono text-gray-500 px-1">
          <Loader2 className="h-3 w-3 animate-spin" style={{ color: '#156065' }} />
          <span>Starting up...</span>
        </div>
      )}
    </div>
  );
}

interface RunQueueModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  queueCount: number;
  jobIds: string[];
  onAnalysisComplete?: () => void;
}

export function RunQueueModal({ open, onOpenChange, queueCount, jobIds, onAnalysisComplete }: RunQueueModalProps) {
  const [hasError, setHasError] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [isComplete, setIsComplete] = useState(false);

  // Reset state when modal opens or closes
  useEffect(() => {
    setHasError(false);
    setErrorMessage("");
    setIsComplete(false);
  }, [open]);

  // Track progress of the single job (sequential processing)
  // The backend processes all queue entries sequentially, we just track the one job
  const jobId = jobIds[0] || null;

  // Reset state when a new job starts (fixes stale completion message issue)
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
  const { progress, message, phase, phaseProgress, isConnected, deploymentContext, metrics, computeDevice } = useTaskProgress({
    taskId: jobId,
    onComplete: () => {
      setIsComplete(true);
      onAnalysisComplete?.();
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

  // Phase ordering for progress calculation
  const phaseOrder = ["init", "video_detection", "video_classification", "image_detection", "image_classification", "saving", "embedding", "finalize"];
  const currentPhaseIndex = phase ? phaseOrder.indexOf(phase) : -1;

  // Calculate progress for each phase based on TQDM metrics
  const getPhaseProgress = (targetPhase: string): number => {
    const targetIndex = phaseOrder.indexOf(targetPhase);
    if (currentPhaseIndex < targetIndex) return 0; // Not started yet
    if (currentPhaseIndex > targetIndex) return 100; // Already completed

    // Currently active phase - use TQDM metrics if available
    if (phase === targetPhase) {
      // If phase is complete (phaseProgress >= 1.0), show 100%
      if (phaseProgress !== undefined && phaseProgress >= 1.0) {
        return 100;
      }
      // Otherwise use TQDM metrics if available
      if (metrics?.current !== undefined && metrics?.total !== undefined && metrics.total > 0) {
        return (metrics.current / metrics.total) * 100;
      }
      return 0; // Phase active but no metrics yet
    }
    return 0;
  };

  const showSpinner = isWaitingForJob;

  return (
    <Dialog open={open} onOpenChange={isComplete || hasError ? onOpenChange : undefined}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>
            {isComplete ? "Analysis complete" : "Analyzing"}
          </DialogTitle>
          <DialogDescription>
            {isComplete
              ? "All deployments have been processed successfully."
              : isWaitingForJob
                ? "Preparing the deployment queue..."
                : "This analysis is resource intensive. Please avoid other heavy tasks while it runs."}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-2">
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
              <CheckCircle2 className="h-5 w-5" style={{ color: '#156065' }} />
              <span className="text-sm font-medium" style={{ color: '#156065' }}>
                Queue processing complete! Processed {queueCount} deployment{queueCount > 1 ? 's' : ''}.
              </span>
            </div>
          )}

          {/* Processing States */}
          {!isComplete && !hasError && (
            <>
              {/* Spinner while waiting for job */}
              {showSpinner && (
                <div className="flex items-center gap-3">
                  <Loader2 className="h-5 w-5 animate-spin" style={{ color: '#0f6064' }} />
                  <span className="text-sm font-medium">{message || "Initializing..."}</span>
                </div>
              )}

              {/* Progress bars - show always when we have deployment context */}
              {!showSpinner && deploymentContext && (
                <>
                  {/* Progress bars card */}
                  <div className="border rounded-lg p-4 space-y-4">
                    {/* Deployment count badge */}
                    <div className="flex items-center gap-2 pb-2">
                      <span className="text-xs font-medium text-gray-600">Deployment</span>
                      <span className="inline-flex items-center rounded-md bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-800">
                        {deploymentContext.deploymentIndex} of {deploymentContext.totalDeployments}
                      </span>
                    </div>

                    {[
                      deploymentContext.videoCount > 0 && { label: "Video detection", phase: "video_detection" },
                      deploymentContext.videoCount > 0 && deploymentContext.hasClassifier && { label: "Video classification", phase: "video_classification" },
                      deploymentContext.imageCount > 0 && { label: "Image detection", phase: "image_detection" },
                      deploymentContext.imageCount > 0 && deploymentContext.hasClassifier && { label: "Image classification", phase: "image_classification" },
                      deploymentContext.hasEmbedding && { label: "Embedding", phase: "embedding" },
                    ].filter(Boolean).map((entry, i) => {
                      const { label: phaseLabel, phase: phaseName } = entry as { label: string; phase: string };
                      return (
                        <div key={phaseName}>
                          <Separator className="mb-4" />
                          <PhaseRow label={phaseLabel} phaseName={phaseName} progress={getPhaseProgress(phaseName)} currentPhase={phase} phaseProgress={phaseProgress} metrics={metrics} computeDevice={computeDevice} />
                        </div>
                      );
                    })}
                  </div>
                </>
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
