/**
 * Run Queue Modal Component
 *
 * Blocking modal that shows progress while processing queue.
 * Connects to WebSocket for real-time progress updates.
 */

import { useState, useEffect } from "react";
import { Loader2, CheckCircle2, XCircle, Clock } from "lucide-react";
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

  // Reset state when modal opens or closes
  useEffect(() => {
    if (open) {
      console.log(`[RunQueueModal ${new Date().toISOString()}] Modal opened`);
      // Reset when opening (clears stale state from previous run)
      setHasError(false);
      setErrorMessage("");
      setIsComplete(false);
    } else {
      // Also reset when closing (cleanup)
      setHasError(false);
      setErrorMessage("");
      setIsComplete(false);
    }
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
  const { progress, message, phase, phaseProgress, isConnected, deploymentContext, metrics } = useTaskProgress({
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

  // Debug: Log when deployment context is received (triggers progress bars to show)
  useEffect(() => {
    if (deploymentContext) {
      console.log(`[RunQueueModal ${new Date().toISOString()}] Deployment context received, progress bars should now be visible:`, deploymentContext);
    }
  }, [deploymentContext]);

  // Calculate overall status
  const isWaitingForJob = !hasError && !isComplete && !hasJob;
  const isProcessing = !isComplete && !hasError && hasJob;
  const phaseProgressPercent = (phaseProgress ?? 0) * 100;

  // Phase ordering for progress calculation
  const phaseOrder = ["init", "video_detection", "video_classification", "image_detection", "image_classification", "finalize"];
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

  const videoDetectionProgress = getPhaseProgress("video_detection");
  const videoClassificationProgress = getPhaseProgress("video_classification");
  const imageDetectionProgress = getPhaseProgress("image_detection");
  const imageClassificationProgress = getPhaseProgress("image_classification");

  // Determine status messages (raw tqdm output from backend)
  const getPhaseStatus = (targetPhase: string, defaultWaiting: string): string => {
    const targetIndex = phaseOrder.indexOf(targetPhase);
    const finalizeIndex = phaseOrder.indexOf("finalize");

    if (currentPhaseIndex < targetIndex) return defaultWaiting;

    // If we're in finalize phase and this phase is completed, show "Finalizing..."
    if (currentPhaseIndex === finalizeIndex && currentPhaseIndex > targetIndex) return "Finalizing...";

    if (currentPhaseIndex > targetIndex) return "Complete";

    // If this is the current phase
    if (phase === targetPhase) {
      // Check if metrics show 100% complete (current === total)
      if (metrics?.current !== undefined && metrics?.total !== undefined && metrics.current >= metrics.total) {
        return "Finalizing...";
      }
      // Fallback: Check if phase_progress is 100% (for phases without detailed metrics)
      if (phaseProgress !== undefined && phaseProgress >= 1.0) {
        return "Finalizing...";
      }
      // Otherwise show "Starting up..." (will display until metrics card takes over)
      return "Starting up...";
    }
    return defaultWaiting;
  };

  const videoDetectionStatus = getPhaseStatus("video_detection", "Waiting...");
  const videoClassificationStatus = getPhaseStatus("video_classification", "Waiting...");
  const imageDetectionStatus = getPhaseStatus("image_detection", "Waiting...");
  const imageClassificationStatus = getPhaseStatus("image_classification", "Waiting...");

  // Helper function to render status with icon
  const renderStatusWithIcon = (status: string) => {
    if (status === "Waiting...") {
      return (
        <div className="flex items-center gap-2">
          <Clock className="h-3.5 w-3.5" style={{ color: '#156065' }} />
          <span>{status}</span>
        </div>
      );
    } else if (status === "Starting up..." || status === "Finalizing...") {
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

  const showSpinner = phase === "finalize" || isWaitingForJob;

  return (
    <Dialog open={open} onOpenChange={isComplete || hasError ? onOpenChange : undefined}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>
            {isComplete ? "Processing complete" : "Processing"}
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
              {/* Spinner for finalize phase */}
              {showSpinner && (
                <div className="flex items-center gap-3">
                  <Loader2 className="h-5 w-5 text-blue-600 animate-spin" />
                  <span className="text-sm font-medium">{message || "Initializing..."}</span>
                </div>
              )}

              {/* Progress bars - show always when we have deployment context */}
              {!showSpinner && deploymentContext && (
                <>
                  {/* Progress bars card */}
                  <div className="border rounded-lg p-4 space-y-4">
                    {/* Deployment count badge */}
                    <div className="flex items-center gap-2 pb-2 border-b">
                      <span className="text-xs font-medium text-gray-600">Deployment</span>
                      <span className="inline-flex items-center rounded-md bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-800">
                        {deploymentContext.deploymentIndex} of {deploymentContext.totalDeployments}
                      </span>
                    </div>

                    {/* Video Detection - only if videos present */}
                    {deploymentContext.videoCount > 0 && (
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <p className="text-sm font-medium text-gray-700">Video detection</p>
                          <span className="text-sm text-gray-500 font-mono">{videoDetectionProgress.toFixed(0)}%</span>
                        </div>
                        <Progress value={videoDetectionProgress} className="h-2" />

                        {/* Info card - only for active phase and not at 100% */}
                        {phase === "video_detection" && metrics?.current !== undefined && metrics?.total !== undefined && metrics.current < metrics.total && (
                          <div className="text-xs space-y-0.5 rounded-md bg-gray-50 p-3 font-mono text-gray-600">
                            <div className="flex justify-between">
                              <span>Processing {metrics.unit || 'items'}:</span>
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
                            {metrics.rate && metrics.unit && (
                              <div className="flex justify-between">
                                <span>{metrics.unit.charAt(0).toUpperCase() + metrics.unit.slice(1)} per second:</span>
                                <span>{metrics.rate.toFixed(2)}</span>
                              </div>
                            )}
                            <div className="flex justify-between">
                              <span>Running on:</span>
                              <span className="text-gray-400">[detecting...]</span>
                            </div>
                          </div>
                        )}

                        {/* Status for non-active phases OR active phase without metrics yet OR at 100% */}
                        {(phase !== "video_detection" || !(metrics?.current !== undefined && metrics?.total !== undefined && metrics.current < metrics.total)) && (
                          <div className="text-xs rounded-md bg-gray-50 p-3 font-mono text-gray-600">
                            {renderStatusWithIcon(videoDetectionStatus)}
                          </div>
                        )}
                      </div>
                    )}

                    {/* Video Classification - only if videos AND classifier */}
                    {deploymentContext.videoCount > 0 && deploymentContext.hasClassifier && (
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <p className="text-sm font-medium text-gray-700">Video classification</p>
                          <span className="text-sm text-gray-500 font-mono">{videoClassificationProgress.toFixed(0)}%</span>
                        </div>
                        <Progress value={videoClassificationProgress} className="h-2" />

                        {/* Info card - only for active phase and not at 100% */}
                        {phase === "video_classification" && metrics?.current !== undefined && metrics?.total !== undefined && metrics.current < metrics.total && (
                          <div className="text-xs space-y-0.5 rounded-md bg-gray-50 p-3 font-mono text-gray-600">
                            <div className="flex justify-between">
                              <span>Processing {metrics.unit || 'items'}:</span>
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
                            {metrics.rate && metrics.unit && (
                              <div className="flex justify-between">
                                <span>{metrics.unit.charAt(0).toUpperCase() + metrics.unit.slice(1)} per second:</span>
                                <span>{metrics.rate.toFixed(2)}</span>
                              </div>
                            )}
                            <div className="flex justify-between">
                              <span>Running on:</span>
                              <span className="text-gray-400">[detecting...]</span>
                            </div>
                          </div>
                        )}
                        {(phase !== "video_classification" || !(metrics?.current !== undefined && metrics?.total !== undefined && metrics.current < metrics.total)) && (
                          <div className="text-xs rounded-md bg-gray-50 p-3 font-mono text-gray-600">
                            {renderStatusWithIcon(videoClassificationStatus)}
                          </div>
                        )}
                      </div>
                    )}

                    {/* Image Detection - only if images present */}
                    {deploymentContext.imageCount > 0 && (
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <p className="text-sm font-medium text-gray-700">Image detection</p>
                          <span className="text-sm text-gray-500 font-mono">{imageDetectionProgress.toFixed(0)}%</span>
                        </div>
                        <Progress value={imageDetectionProgress} className="h-2" />

                        {/* Info card - only for active phase and not at 100% */}
                        {phase === "image_detection" && metrics?.current !== undefined && metrics?.total !== undefined && metrics.current < metrics.total && (
                          <div className="text-xs space-y-0.5 rounded-md bg-gray-50 p-3 font-mono text-gray-600">
                            <div className="flex justify-between">
                              <span>Processing {metrics.unit || 'items'}:</span>
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
                            {metrics.rate && metrics.unit && (
                              <div className="flex justify-between">
                                <span>{metrics.unit.charAt(0).toUpperCase() + metrics.unit.slice(1)} per second:</span>
                                <span>{metrics.rate.toFixed(2)}</span>
                              </div>
                            )}
                            <div className="flex justify-between">
                              <span>Running on:</span>
                              <span className="text-gray-400">[detecting...]</span>
                            </div>
                          </div>
                        )}
                        {(phase !== "image_detection" || !(metrics?.current !== undefined && metrics?.total !== undefined && metrics.current < metrics.total)) && (
                          <div className="text-xs rounded-md bg-gray-50 p-3 font-mono text-gray-600">
                            {renderStatusWithIcon(imageDetectionStatus)}
                          </div>
                        )}
                      </div>
                    )}

                    {/* Image Classification - only if images AND classifier */}
                    {deploymentContext.imageCount > 0 && deploymentContext.hasClassifier && (
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <p className="text-sm font-medium text-gray-700">Image classification</p>
                          <span className="text-sm text-gray-500 font-mono">{imageClassificationProgress.toFixed(0)}%</span>
                        </div>
                        <Progress value={imageClassificationProgress} className="h-2" />

                        {/* Info card - only for active phase and not at 100% */}
                        {phase === "image_classification" && metrics?.current !== undefined && metrics?.total !== undefined && metrics.current < metrics.total && (
                          <div className="text-xs space-y-0.5 rounded-md bg-gray-50 p-3 font-mono text-gray-600">
                            <div className="flex justify-between">
                              <span>Processing {metrics.unit || 'items'}:</span>
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
                            {metrics.rate && metrics.unit && (
                              <div className="flex justify-between">
                                <span>{metrics.unit.charAt(0).toUpperCase() + metrics.unit.slice(1)} per second:</span>
                                <span>{metrics.rate.toFixed(2)}</span>
                              </div>
                            )}
                            <div className="flex justify-between">
                              <span>Running on:</span>
                              <span className="text-gray-400">[detecting...]</span>
                            </div>
                          </div>
                        )}
                        {(phase !== "image_classification" || !(metrics?.current !== undefined && metrics?.total !== undefined && metrics.current < metrics.total)) && (
                          <div className="text-xs rounded-md bg-gray-50 p-3 font-mono text-gray-600">
                            {renderStatusWithIcon(imageClassificationStatus)}
                          </div>
                        )}
                      </div>
                    )}
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
