/**
 * Run Queue Modal Component
 *
 * Blocking modal that shows progress while processing queue and the
 * end-of-run "receipt" (success + a unified log table covering warnings
 * and errors) with a download option. Queue entries from this run are
 * deleted on Close.
 */

import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Loader2,
  CheckCircle2,
  Download,
  Ban,
  Info,
  Tag,
  Tally5,
  LayoutDashboard,
  ChevronRight,
} from "lucide-react";
import { basename } from "@/lib/path-utils";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { useTaskProgress } from "@/hooks/useTaskProgress";
import { AnalysisProgress } from "./AnalysisProgress";
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
  video_processing_failure: "Could not be read",
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
  return basename(path) || path;
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

// PhaseRow + per-phase computation moved to ./AnalysisProgress.tsx,
// shared across every run-progress screen.

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

/**
 * A clickable "what next" row in the completion modal: icon + a short
 * action title + a one-line description of where it takes you. Used so
 * the post-analysis step is self-explanatory instead of a bare button.
 */
function NextStepRow({
  icon: Icon,
  title,
  description,
  onClick,
  disabled,
}: {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  description: string;
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className="flex w-full items-start gap-3 rounded-lg border p-3 text-left transition-colors hover:bg-accent disabled:pointer-events-none disabled:opacity-50"
    >
      <Icon className="mt-0.5 h-5 w-5 shrink-0 text-primary" />
      <div className="flex-1">
        <p className="text-sm font-medium">{title}</p>
        <p className="text-xs text-muted-foreground">{description}</p>
      </div>
      <ChevronRight className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
    </button>
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
        <table className="w-full table-fixed text-left text-xs">
          <thead className="bg-gray-50 text-[11px] uppercase tracking-wide text-gray-500 sticky top-0">
            <tr>
              <th className="w-[92px] px-3 py-2 font-medium">Severity</th>
              <th className="w-[150px] px-3 py-2 font-medium">Type</th>
              <th className="w-[130px] px-3 py-2 font-medium">Deployment</th>
              <th className="px-3 py-2 font-medium">Detail</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {rows.map((r, i) => (
              <tr key={i}>
                <td className="px-3 py-2">{severityBadge(r.severity)}</td>
                <td className="truncate px-3 py-2 text-gray-900" title={r.typeLabel}>
                  {r.typeLabel}
                </td>
                <td className="truncate px-3 py-2 text-gray-700" title={r.deployment}>
                  {r.deployment}
                </td>
                <td className="px-3 py-2 text-gray-700 font-mono" title={r.detail}>
                  {/* Truncate from the start so the filename (end of the
                      path) stays visible, with a leading ellipsis. */}
                  <span
                    className="block overflow-hidden text-ellipsis whitespace-nowrap"
                    style={{ direction: "rtl", textAlign: "left" }}
                  >
                    {r.detail}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/** Terminal-state info passed to ``renderTerminalFooter`` so the caller
 * can render an appropriate continue / retry button row. ``close``
 * runs the modal's standard close logic (including queue-entry cleanup
 * when enabled) and resolves when done. */
export type RunQueueTerminalKind = "completed" | "failed" | "cancelled";
export interface RunQueueTerminalInfo {
  kind: RunQueueTerminalKind;
  close: () => Promise<void>;
  isClosing: boolean;
}

interface RunQueueModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  queueCount: number;
  jobIds: string[];
  projectId: string;
  queueEntryIds: string[];
  onAnalysisComplete?: () => void;
  /** Override the default terminal-state footer (New analysis +
   * Verify + Dashboard). Receives the terminal kind and a close
   * callback so the caller can chain navigation after close. When
   * omitted, the projects-mode default is rendered. */
  renderTerminalFooter?: (info: RunQueueTerminalInfo) => React.ReactNode;
  /** Whether to delete completed/failed queue entries on close.
   * Projects mode wants this (true, default) because the queue is a
   * scratch list. Folder-run mode wants this off so the entry stays
   * available for the "you analysed this folder before" lookup. */
  deleteQueueEntriesOnClose?: boolean;
  /** Caller context. Drives wording in the terminal-state UI:
   * projects mode talks about "deployments" (the run is a batch of
   * N items from the queue), folder-run mode talks about a single
   * folder and drops the deployment count entirely. Defaults to
   * "projects" so existing callers stay unchanged. */
  mode?: "projects" | "folder-run";
}

export function RunQueueModal({
  open,
  onOpenChange,
  queueCount,
  jobIds,
  projectId,
  queueEntryIds,
  onAnalysisComplete,
  renderTerminalFooter,
  deleteQueueEntriesOnClose = true,
  mode = "projects",
}: RunQueueModalProps) {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const [hasError, setHasError] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [isComplete, setIsComplete] = useState(false);
  const [isClosing, setIsClosing] = useState(false);
  const [hasCancelled, setHasCancelled] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);

  useEffect(() => {
    setHasError(false);
    setErrorMessage("");
    setIsComplete(false);
    setIsClosing(false);
    setHasCancelled(false);
    setIsCancelling(false);
  }, [open]);

  const jobId = jobIds[0] || null;

  useEffect(() => {
    if (jobId) {
      setIsComplete(false);
      setHasError(false);
      setErrorMessage("");
      setHasCancelled(false);
      setIsCancelling(false);
    }
  }, [jobId]);

  const { message, phase, phaseProgress, isConnected, deploymentContext, metrics, computeDevice, cancel } = useTaskProgress({
    taskId: jobId,
    onComplete: () => {
      setIsComplete(true);
      onAnalysisComplete?.();
    },
    onError: (msg) => {
      setHasError(true);
      setErrorMessage(msg);
      // Even on failure some deployments may have completed before the
      // crash — refresh so the UI reflects whatever did land.
      onAnalysisComplete?.();
    },
    onCancelled: () => {
      setHasCancelled(true);
      setIsCancelling(false);
      // Run is done and DB state has updated; let the rest of the app
      // refresh just like on normal completion.
      onAnalysisComplete?.();
    },
  });

  // Once terminal, fetch fresh queue entries so we can inspect per-entry
  // warnings/errors without relying on stale data in the list view.
  const { data: allEntries } = useQuery({
    queryKey: ["deployment-queue", projectId],
    queryFn: () => deploymentQueueApi.list(projectId),
    enabled: open && (isComplete || hasError || hasCancelled),
  });

  const runEntries = (allEntries || []).filter((e) => queueEntryIds.includes(e.id));
  // Files with no capture date were still detected and classified and live
  // in the database; they are NOT skipped or failed. Keep them out of the
  // issues table and the skipped tally, and surface them as a calm note.
  const allRows = buildLogRows(runEntries);
  const datelessCount = allRows.filter((r) => r.type === "missing_timestamp").length;
  const logRows = allRows.filter((r) => r.type !== "missing_timestamp");

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
  const isWaitingForJob = !hasError && !isComplete && !hasCancelled && !hasJob;
  const isProcessing =
    !isComplete && !hasError && !hasCancelled && hasJob && !isCancelling;

  // Phase order + per-phase progress logic moved to AnalysisProgress.tsx.

  const showSpinner = isWaitingForJob;

  const handleClose = async () => {
    if (isClosing) return;
    setIsClosing(true);
    try {
      if (deleteQueueEntriesOnClose) {
        // Only delete entries that reached a terminal state in this run.
        // After a cancel, entries reset back to "pending" must survive so
        // the user can re-run them without re-adding the folders.
        const terminalStatuses = new Set(["completed", "failed"]);
        const idsToDelete = runEntries
          .filter((e) => terminalStatuses.has(e.status))
          .map((e) => e.id);
        await Promise.all(
          idsToDelete.map((id) =>
            deploymentQueueApi.remove(id).catch(() => null),
          ),
        );
      }
    } finally {
      void queryClient.invalidateQueries({
        queryKey: ["deployment-queue", projectId],
      });
      onOpenChange(false);
    }
  };

  const inTerminalState = isComplete || hasError || hasCancelled;
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
      <DialogContent
        className={`${showLogTable ? "sm:max-w-3xl" : "sm:max-w-xl"} [&>button.absolute]:hidden`}
      >
        <DialogHeader>
          <DialogTitle>
            {isComplete
              ? "Analysis complete"
              : hasCancelled
                ? "Analysis cancelled"
                : hasError
                  ? "Analysis failed"
                  : isCancelling
                    ? "Cancelling..."
                    : "Analyzing"}
          </DialogTitle>
          <DialogDescription>
            {isComplete
              ? mode === "folder-run"
                ? "Your folder has been analysed. AddaxAI suggested a species and a count for everything it found. The next steps let you review and correct them before saving."
                : "AddaxAI filled in a suggested species label and a count for everything it found. You can accept these as they are, but the AI makes mistakes, so a quick review is recommended."
              : hasCancelled
                ? mode === "folder-run"
                  ? "The run was stopped before finishing."
                  : "Review what finished before the run was stopped."
                : hasError
                  ? "The run stopped before finishing. Review the details below."
                  : isCancelling
                    ? mode === "folder-run"
                      ? "Stopping the analysis..."
                      : "Stopping the current deployment..."
                    : isWaitingForJob
                      ? mode === "folder-run"
                        ? "Preparing the analysis..."
                        : "Preparing the deployment queue..."
                      : "This analysis is resource intensive. Please avoid other heavy tasks while it runs."}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3">
          {isComplete && !hasError && (() => {
            const failureCount = logRows.filter((r) => r.severity === "error").length;
            const warningCount = logRows.filter((r) => r.severity === "warning").length;
            const totalAttempted = successCount + failureCount;
            const stillLoading = runEntries.length === 0;

            const mediaParts: string[] = [];
            if (savedImageCount > 0) {
              mediaParts.push(
                `${savedImageCount} image${savedImageCount === 1 ? '' : 's'}`,
              );
            }
            if (savedVideoCount > 0) {
              mediaParts.push(
                `${savedVideoCount} video${savedVideoCount === 1 ? '' : 's'}`,
              );
            }
            const mediaText = mediaParts.join(' and ');

            const hasIssues = !stillLoading && (warningCount > 0 || failureCount > 0);

            // Body string: projects mode talks about deployments, folder
            // mode drops the deployment count (always 1) and uses media
            // counts as the headline number.
            let mainText: string;
            if (mode === "folder-run") {
              const prefix = hasIssues ? "Processed" : "Successfully processed";
              mainText = mediaText
                ? `${prefix} ${mediaText}.`
                : `${prefix} the folder.`;
            } else {
              const deploymentN = stillLoading ? queueCount : successCount;
              const deploymentsText =
                failureCount > 0
                  ? `${successCount} of ${totalAttempted} deployments`
                  : `${deploymentN} deployment${deploymentN === 1 ? '' : 's'}`;
              const withMedia = mediaText ? ` with ${mediaText}` : '';
              const prefix = hasIssues ? "Processed" : "Successfully processed";
              mainText = `${prefix} ${deploymentsText}${withMedia}.`;
            }

            // Hint at the log table so users know where the details are.
            const issueBits: string[] = [];
            if (warningCount > 0) {
              issueBits.push(`${warningCount} file${warningCount === 1 ? '' : 's'} skipped`);
            }
            if (failureCount > 0) {
              if (mode === "folder-run") {
                issueBits.push(
                  `${failureCount} error${failureCount === 1 ? '' : 's'}`,
                );
              } else {
                issueBits.push(
                  `${failureCount} deployment${failureCount === 1 ? '' : 's'} failed`,
                );
              }
            }
            const issueText = issueBits.length > 0 ? ` See details below: ${issueBits.join(', ')}.` : '';

            const iconColor = failureCount > 0
              ? '#882000'
              : warningCount > 0
                ? '#b45309'
                : '#156065';

            return (
              <div className="flex items-start gap-3">
                <CheckCircle2
                  className="h-5 w-5 shrink-0 mt-0.5"
                  style={{ color: iconColor }}
                />
                <div className="text-sm font-medium" style={{ color: iconColor }}>
                  {mainText}{issueText}
                </div>
              </div>
            );
          })()}

          {hasCancelled && (() => {
            const completedCount = runEntries.filter(
              (e) => e.status === "completed",
            ).length;
            const pendingCount = runEntries.filter(
              (e) => e.status === "pending",
            ).length;
            const totalInRun = runEntries.length || queueCount;

            const parts: string[] = [];
            if (mode === "folder-run") {
              parts.push(
                completedCount > 0
                  ? "The folder was partly processed before the run was stopped."
                  : "The run was stopped before the folder finished processing.",
              );
            } else {
              parts.push(
                `${completedCount} of ${totalInRun} deployment${totalInRun === 1 ? '' : 's'} completed`,
              );
              if (pendingCount > 0) {
                parts.push(
                  `${pendingCount} returned to the queue`,
                );
              }
            }
            return (
              <div className="flex items-start gap-3 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-amber-900">
                <Ban className="h-5 w-5 shrink-0 mt-0.5" />
                <div className="text-sm font-medium">
                  {parts.join('. ')}{parts[parts.length - 1]?.endsWith('.') ? '' : '.'}
                </div>
              </div>
            );
          })()}

          {showLogTable && <LogTable rows={logRows} />}

          {inTerminalState && datelessCount > 0 && (
            <div className="flex items-start gap-2 rounded-md border bg-muted/40 px-3 py-2 text-xs text-muted-foreground">
              <Info className="h-4 w-4 shrink-0 mt-0.5" />
              <span>
                {datelessCount} file{datelessCount === 1 ? "" : "s"} had no
                capture date. They were still detected and classified and are in
                your data, just left out of time-based stats and charts.
              </span>
            </div>
          )}

          {/* What next? comes last: the user reads what happened and any
              issues first, then decides the next step. */}
          {isComplete && mode !== "folder-run" && successCount > 0 && (
            <div className="space-y-2 pt-1">
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                What next?
              </p>
              <NextStepRow
                icon={Tag}
                title="Check the labels"
                description="Correct the species the AI assigned to each animal."
                disabled={isClosing}
                onClick={async () => {
                  await handleClose();
                  navigate(`/projects/${projectId}/labels`);
                }}
              />
              <NextStepRow
                icon={Tally5}
                title="Confirm the counts"
                description="Check how many individuals the AI counted per observation."
                disabled={isClosing}
                onClick={async () => {
                  await handleClose();
                  navigate(`/projects/${projectId}/counts`);
                }}
              />
              <NextStepRow
                icon={LayoutDashboard}
                title="Open the dashboard"
                description="See an overview of what was found."
                disabled={isClosing}
                onClick={async () => {
                  await handleClose();
                  navigate(`/projects/${projectId}/dashboard`);
                }}
              />
            </div>
          )}

          {!inTerminalState && (
            <>
              {showSpinner && (
                <div className="flex items-center gap-3">
                  <Loader2 className="h-5 w-5 animate-spin" style={{ color: '#0f6064' }} />
                  <span className="text-sm font-medium">{message || "Initializing..."}</span>
                </div>
              )}

              {isCancelling && (
                <div className="flex items-center gap-3 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-amber-900">
                  <Loader2 className="h-4 w-4 animate-spin shrink-0" />
                  <span className="text-sm font-medium">
                    {mode === "folder-run"
                      ? "Stopping the analysis..."
                      : "Stopping the current deployment..."}
                  </span>
                </div>
              )}

              {!showSpinner && (
                <AnalysisProgress
                  phase={phase}
                  phaseProgress={phaseProgress}
                  metrics={metrics}
                  computeDevice={computeDevice}
                  deploymentContext={deploymentContext}
                  message={message}
                />
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

        {inTerminalState ? (
          <DialogFooter>
            {renderTerminalFooter ? (
              renderTerminalFooter({
                kind: isComplete
                  ? "completed"
                  : hasCancelled
                    ? "cancelled"
                    : "failed",
                close: handleClose,
                isClosing,
              })
            ) : (
              <Button
                variant="outline"
                onClick={handleClose}
                disabled={isClosing}
              >
                {isClosing ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Closing...
                  </>
                ) : isComplete ? (
                  "Analyse more data"
                ) : (
                  "Close"
                )}
              </Button>
            )}
          </DialogFooter>
        ) : hasJob && !isCancelling ? (
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setIsCancelling(true);
                cancel();
              }}
            >
              Cancel
            </Button>
          </DialogFooter>
        ) : null}
      </DialogContent>
    </Dialog>
  );
}
