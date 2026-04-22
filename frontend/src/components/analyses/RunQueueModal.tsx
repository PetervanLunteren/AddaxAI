/**
 * Run Queue Modal Component
 *
 * Blocking modal that shows progress while processing queue and the
 * end-of-run "receipt" (success + a unified log table covering warnings
 * and errors) with a download option. Queue entries from this run are
 * deleted on Close.
 */

import { useState, useEffect } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Loader2, CheckCircle2, Download } from "lucide-react";
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
import {
  deploymentQueueApi,
  type DeploymentQueueEntry,
} from "@/api/deployment-queue";
import { downloadTextFile } from "@/lib/download";

type Severity = "warning" | "error";

interface LogRow {
  severity: Severity;
  type: string;
  typeLabel: string;
  deployment: string;
  detail: string;
}

// Keep this in one place so adding a new warning/error kind is a
// one-line change. Backend emits the `type` key; frontend maps to a
// human label.
const TYPE_LABELS: Record<string, string> = {
  missing_timestamp: "No capture timestamp",
  job_failed: "Deployment failed",
};

// Used to split skipped-file warnings into image vs video buckets for
// the success message. Anything not matched as video is counted as
// an image (the queue only enqueues image/video media).
const IMAGE_VIDEO_RE = {
  video: /\.(mp4|mov|avi|mkv|m4v|wmv|flv|webm|mts|m2ts|3gp)$/i,
};

function labelForType(type: string): string {
  return TYPE_LABELS[type] ?? type;
}

function deploymentNameOf(path: string): string {
  return path.split("/").pop() || path;
}

interface StoredWarning {
  type?: string;
  path?: string;
  message?: string;
}

function parseWarnings(raw: string | null): StoredWarning[] {
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) return parsed as StoredWarning[];
  } catch {
    // Legacy format: newline-joined paths, all missing_timestamp.
    return raw
      .split("\n")
      .filter(Boolean)
      .map((p) => ({ type: "missing_timestamp", path: p }));
  }
  return [];
}

function buildLogRows(entries: DeploymentQueueEntry[]): LogRow[] {
  const rows: LogRow[] = [];
  for (const entry of entries) {
    const name = deploymentNameOf(entry.folder_path);

    for (const w of parseWarnings(entry.warnings)) {
      const type = w.type || "warning";
      rows.push({
        severity: "warning",
        type,
        typeLabel: labelForType(type),
        deployment: name,
        detail: w.path || w.message || "",
      });
    }

    if (entry.status === "failed" && entry.error) {
      rows.push({
        severity: "error",
        type: "job_failed",
        typeLabel: labelForType("job_failed"),
        deployment: name,
        detail: entry.error,
      });
    }
  }
  return rows;
}

function csvEscape(value: string): string {
  if (/[",\r\n]/.test(value)) {
    return `"${value.replace(/"/g, '""')}"`;
  }
  return value;
}

function formatLogCsv(rows: LogRow[]): string {
  const header = ["severity", "type", "deployment", "detail"].join(",");
  const lines = rows.map((r) =>
    [r.severity, r.type, r.deployment, r.detail].map(csvEscape).join(","),
  );
  return [header, ...lines].join("\n");
}

function timestampSuffix(): string {
  const d = new Date();
  const pad = (n: number) => n.toString().padStart(2, "0");
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
}

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
  const isFinalizing = isActive && progress >= 100;
  const hasValidMetrics = isActive && !isFinalizing && metrics?.current !== undefined && metrics?.total !== undefined && metrics.current < metrics.total;
  const isStartingUp = isActive && !isFinalizing && !hasValidMetrics && (phaseProgress === undefined || phaseProgress < 1.0);

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

      {isStartingUp && (
        <div className="flex items-center gap-2 text-[11px] font-mono text-gray-500 px-1">
          <Loader2 className="h-3 w-3 animate-spin" style={{ color: '#156065' }} />
          <span>Starting up...</span>
        </div>
      )}

      {isFinalizing && (
        <div className="flex items-center gap-2 text-[11px] font-mono text-gray-500 px-1">
          <Loader2 className="h-3 w-3 animate-spin" style={{ color: '#156065' }} />
          <span>Finalizing...</span>
        </div>
      )}
    </div>
  );
}

interface LogTableProps {
  rows: LogRow[];
}

function severityBadge(severity: Severity) {
  if (severity === "error") {
    return (
      <span className="inline-flex items-center rounded-md bg-red-100 text-red-700 px-1.5 py-0.5 text-[11px] font-medium">
        Error
      </span>
    );
  }
  return (
    <span className="inline-flex items-center rounded-md bg-amber-100 text-amber-800 px-1.5 py-0.5 text-[11px] font-medium">
      Warning
    </span>
  );
}

function LogTable({ rows }: LogTableProps) {
  const warningCount = rows.filter((r) => r.severity === "warning").length;
  const errorCount = rows.filter((r) => r.severity === "error").length;

  const handleDownload = () => {
    downloadTextFile(`run-log-${timestampSuffix()}.csv`, formatLogCsv(rows));
  };

  return (
    <div className="rounded-lg border border-gray-200 bg-white">
      <div className="flex items-center justify-between gap-3 px-3 py-2 border-b border-gray-200">
        <p className="text-sm font-medium text-gray-900">
          {rows.length} issue{rows.length === 1 ? "" : "s"}
          <span className="text-xs font-normal text-gray-500 ml-2">
            {warningCount > 0 && (
              <span className="text-amber-700">
                {warningCount} warning{warningCount === 1 ? "" : "s"}
              </span>
            )}
            {warningCount > 0 && errorCount > 0 && <span> · </span>}
            {errorCount > 0 && (
              <span className="text-red-700">
                {errorCount} error{errorCount === 1 ? "" : "s"}
              </span>
            )}
          </span>
        </p>
        <Button variant="outline" onClick={handleDownload}>
          <Download className="h-4 w-4 mr-2" />
          Download CSV
        </Button>
      </div>

      <div className="max-h-64 overflow-auto">
        <table className="w-full text-left text-xs">
          <thead className="bg-gray-50 text-[11px] uppercase tracking-wide text-gray-500 sticky top-0">
            <tr>
              <th className="px-3 py-2 font-medium">Severity</th>
              <th className="px-3 py-2 font-medium">Type</th>
              <th className="px-3 py-2 font-medium">Deployment</th>
              <th className="px-3 py-2 font-medium">Detail</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {rows.map((r, i) => (
              <tr key={i} className="align-top">
                <td className="px-3 py-2">{severityBadge(r.severity)}</td>
                <td className="px-3 py-2 text-gray-900">{r.typeLabel}</td>
                <td className="px-3 py-2 text-gray-700">{r.deployment}</td>
                <td className="px-3 py-2 text-gray-700 font-mono break-all">{r.detail}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

interface RunQueueModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  queueCount: number;
  jobIds: string[];
  projectId: string;
  queueEntryIds: string[];
  onAnalysisComplete?: () => void;
}

export function RunQueueModal({
  open,
  onOpenChange,
  queueCount,
  jobIds,
  projectId,
  queueEntryIds,
  onAnalysisComplete,
}: RunQueueModalProps) {
  const queryClient = useQueryClient();
  const [hasError, setHasError] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [isComplete, setIsComplete] = useState(false);
  const [isClosing, setIsClosing] = useState(false);

  useEffect(() => {
    setHasError(false);
    setErrorMessage("");
    setIsComplete(false);
    setIsClosing(false);
  }, [open]);

  const jobId = jobIds[0] || null;

  useEffect(() => {
    if (jobId) {
      setIsComplete(false);
      setHasError(false);
      setErrorMessage("");
    }
  }, [jobId]);

  const { message, phase, phaseProgress, isConnected, deploymentContext, metrics, computeDevice } = useTaskProgress({
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

  // Once terminal, fetch fresh queue entries so we can inspect per-entry
  // warnings/errors without relying on stale data in the list view.
  const { data: allEntries } = useQuery({
    queryKey: ["deployment-queue", projectId],
    queryFn: () => deploymentQueueApi.list(projectId),
    enabled: open && (isComplete || hasError),
  });

  const runEntries = (allEntries || []).filter((e) => queueEntryIds.includes(e.id));
  const logRows = buildLogRows(runEntries);

  // Synthesize a row for a job-level crash (no per-entry error recorded).
  if ((isComplete || hasError) && hasError && logRows.every((r) => r.severity !== "error")) {
    logRows.push({
      severity: "error",
      type: "job_failed",
      typeLabel: labelForType("job_failed"),
      deployment: "",
      detail: errorMessage || "Unknown error",
    });
  }

  const hasJob = Boolean(jobId);
  const isWaitingForJob = !hasError && !isComplete && !hasJob;
  const isProcessing = !isComplete && !hasError && hasJob;

  const phaseOrder = ["init", "video_detection", "video_classification", "image_detection", "image_classification", "saving", "embedding", "finalize"];
  const currentPhaseIndex = phase ? phaseOrder.indexOf(phase) : -1;

  const getPhaseProgress = (targetPhase: string): number => {
    const targetIndex = phaseOrder.indexOf(targetPhase);
    if (currentPhaseIndex < targetIndex) return 0;
    if (currentPhaseIndex > targetIndex) return 100;

    if (phase === targetPhase) {
      if (phaseProgress !== undefined && phaseProgress >= 1.0) {
        return 100;
      }
      if (metrics?.current !== undefined && metrics?.total !== undefined && metrics.total > 0) {
        return (metrics.current / metrics.total) * 100;
      }
      return 0;
    }
    return 0;
  };

  const showSpinner = isWaitingForJob;

  const handleClose = async () => {
    if (isClosing) return;
    setIsClosing(true);
    try {
      await Promise.all(
        queueEntryIds.map((id) =>
          deploymentQueueApi.remove(id).catch(() => null),
        ),
      );
    } finally {
      void queryClient.invalidateQueries({
        queryKey: ["deployment-queue", projectId],
      });
      onOpenChange(false);
    }
  };

  const inTerminalState = isComplete || hasError;
  const completedEntries = runEntries.filter((e) => e.status === "completed");
  const successCount = completedEntries.length;
  const completedImageTotal = completedEntries.reduce(
    (sum, e) => sum + (e.image_count || 0),
    0,
  );
  const completedVideoTotal = completedEntries.reduce(
    (sum, e) => sum + (e.video_count || 0),
    0,
  );
  const warningRows = logRows.filter((r) => r.severity === "warning");
  const skippedVideoCount = warningRows.filter((r) =>
    IMAGE_VIDEO_RE.video.test(r.detail),
  ).length;
  const skippedImageCount = warningRows.length - skippedVideoCount;
  const savedImageCount = Math.max(0, completedImageTotal - skippedImageCount);
  const savedVideoCount = Math.max(0, completedVideoTotal - skippedVideoCount);
  const showLogTable = inTerminalState && logRows.length > 0;

  return (
    <Dialog open={open} onOpenChange={inTerminalState ? onOpenChange : undefined}>
      <DialogContent className={showLogTable ? "sm:max-w-3xl" : "sm:max-w-lg"}>
        <DialogHeader>
          <DialogTitle>
            {isComplete ? "Analysis complete" : hasError ? "Analysis failed" : "Analyzing"}
          </DialogTitle>
          <DialogDescription>
            {isComplete
              ? "Review the results below, then close to clear the queue."
              : hasError
                ? "The run stopped before finishing. Review the details below."
                : isWaitingForJob
                  ? "Preparing the deployment queue..."
                  : "This analysis is resource intensive. Please avoid other heavy tasks while it runs."}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3">
          {isComplete && !hasError && (
            <div className="flex items-start gap-3">
              <CheckCircle2
                className="h-5 w-5 shrink-0 mt-0.5"
                style={{ color: '#156065' }}
              />
              <div className="text-sm font-medium" style={{ color: '#156065' }}>
                <div>
                  Processed {successCount > 0 ? successCount : queueCount} deployment
                  {(successCount || queueCount) === 1 ? '' : 's'}.
                </div>
                {(savedImageCount > 0 || savedVideoCount > 0) && (
                  <div>
                    Successfully analysed{" "}
                    {savedImageCount > 0 && (
                      <>
                        {savedImageCount} image{savedImageCount === 1 ? '' : 's'}
                      </>
                    )}
                    {savedImageCount > 0 && savedVideoCount > 0 && " and "}
                    {savedVideoCount > 0 && (
                      <>
                        {savedVideoCount} video{savedVideoCount === 1 ? '' : 's'}
                      </>
                    )}
                    .
                  </div>
                )}
              </div>
            </div>
          )}

          {showLogTable && <LogTable rows={logRows} />}

          {!isComplete && !hasError && (
            <>
              {showSpinner && (
                <div className="flex items-center gap-3">
                  <Loader2 className="h-5 w-5 animate-spin" style={{ color: '#0f6064' }} />
                  <span className="text-sm font-medium">{message || "Initializing..."}</span>
                </div>
              )}

              {!showSpinner && deploymentContext && (
                <div className="border rounded-lg p-4 space-y-4">
                  <div className="flex items-center gap-2">
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
                  ].filter(Boolean).map((entry) => {
                    const { label: phaseLabel, phase: phaseName } = entry as { label: string; phase: string };
                    return (
                      <div key={phaseName}>
                        <Separator className="mb-4" />
                        <PhaseRow label={phaseLabel} phaseName={phaseName} progress={getPhaseProgress(phaseName)} currentPhase={phase} phaseProgress={phaseProgress} metrics={metrics} computeDevice={computeDevice} />
                      </div>
                    );
                  })}
                </div>
              )}

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

        {inTerminalState && (
          <DialogFooter>
            <Button onClick={handleClose} disabled={isClosing}>
              {isClosing ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Closing...
                </>
              ) : (
                "Close"
              )}
            </Button>
          </DialogFooter>
        )}
      </DialogContent>
    </Dialog>
  );
}
