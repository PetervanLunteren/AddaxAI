/**
 * The one dialog every project export opens. Same parts, same order:
 * which deployments, the export's own choices (the body), where to save
 * it, Start; then progress and the result in the same window.
 *
 * The shell owns the run: start, follow the job over its WebSocket (or
 * wait for a plain request), cancel when the export can stop, and show
 * what was written with a way to open the folder. Each export supplies
 * only its body, how to start, and how to describe its result.
 *
 * Mount it only while it is open (ExportPage does), so every opening
 * starts from the defaults.
 */

import { useEffect, useState, type ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { CheckCircle2, FolderOpen } from "lucide-react";

import { deploymentsApi } from "../../api/deployments";
import type { ExportScope } from "../../api/export";
import { useTaskProgress } from "../../hooks/useTaskProgress";
import { FolderSelector } from "../analyses/FolderSelector";
import { Button } from "../ui/button";
import { Callout } from "../ui/callout";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";
import { NextStepRow } from "../ui/next-step-row";
import { ExportScopeSelect } from "./ExportScopeSelect";
import { JobMetricsProgress } from "./JobMetricsProgress";

/** The two columns of a wide body: the form, and a side panel such as
 * the image copies preview. The shell puts Save in under the left column
 * with the same grid, so it lines up with the form. */
export const WIDE_COLUMNS = "grid gap-6 lg:grid-cols-[minmax(0,1fr)_360px] lg:items-start";

/** One part of an export dialog: a heading, an optional caption, the
 * control. Deployments, the export's choices and Save in all use it, so
 * every dialog reads the same way from top to bottom. */
export function ExportSection({
  title,
  caption,
  children,
}: {
  title: string;
  caption?: string;
  children: ReactNode;
}) {
  return (
    <div className="space-y-2">
      <div>
        <span className="block text-sm font-semibold">{title}</span>
        {caption && (
          <span className="mt-0.5 block text-xs text-muted-foreground">{caption}</span>
        )}
      </div>
      {children}
    </div>
  );
}

/** How an export started: a job to follow, or a request that already
 * finished with its result. */
export type ExportStart = { jobId: string } | { result: unknown };

/** What the result screen shows. `folder` gets the "Open folder" row. */
export interface ExportResultView {
  title: string;
  folder?: string;
  details?: ReactNode;
}

interface ExportDialogProps {
  projectId: string;
  title: string;
  description: ReactNode;
  onClose: () => void;
  /** Without these the shell shows no picker: for a body that filters
   * deployments among its own options (image copies). */
  scope?: ExportScope;
  onScopeChange?: (scope: ExportScope | undefined) => void;
  /** Off for an export that writes into the deployment folders. */
  needsFolder?: boolean;
  /** Only for an export that does something unexpected in the picked
   * folder: replaces an earlier export, or writes a folder, not files.
   * Keeping an existing file (the `_2` rule) is what users expect. */
  folderCaption?: string;
  /** Extra condition on Start, on top of a folder and a selection. */
  canStart?: boolean;
  start: (outputDir: string) => Promise<ExportStart>;
  describeResult: (data: unknown) => ExportResultView;
  cancellable?: boolean;
  /** Two-column bodies (image copies with its folder preview), laid out
   * with `WIDE_COLUMNS`. */
  wide?: boolean;
  /** The export's own choices. A function gets the chosen folder, for a
   * body that previews what it will write there. */
  children?: ReactNode | ((outputDir: string) => ReactNode);
}

type Phase =
  | { kind: "form"; error: string | null }
  | { kind: "running"; jobId: string | null }
  | { kind: "done"; view: ExportResultView };

export function ExportDialog({
  projectId,
  title,
  description,
  onClose,
  scope,
  onScopeChange,
  needsFolder = true,
  folderCaption,
  canStart = true,
  start,
  describeResult,
  cancellable = false,
  wide = false,
  children,
}: ExportDialogProps) {
  const [phase, setPhase] = useState<Phase>({ kind: "form", error: null });
  const [outputDir, setOutputDir] = useState("");
  const [cancelling, setCancelling] = useState(false);

  const { data: deployments } = useQuery({
    queryKey: ["deployments", projectId],
    queryFn: () => deploymentsApi.list({ projectId }),
  });
  const showPicker = (deployments?.length ?? 0) > 1;
  const nothingSelected = scope?.deploymentIds?.length === 0;

  const finish = (data: unknown) => setPhase({ kind: "done", view: describeResult(data) });
  const fail = (message: string) => {
    setCancelling(false);
    setPhase({ kind: "form", error: message });
  };

  const jobId = phase.kind === "running" ? phase.jobId : null;
  const progress = useTaskProgress({
    taskId: jobId,
    onComplete: (data) => finish(data),
    onError: (message) => fail(message),
    onCancelled: () => fail("Export cancelled. Nothing was kept."),
  });

  const running = phase.kind === "running";
  // A refresh or a closed window mid-run would leave the job without a
  // screen to come back to, as the other job dialogs warn.
  useEffect(() => {
    if (!running) return;
    const handler = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [running]);

  const onStart = async () => {
    setPhase({ kind: "running", jobId: null });
    try {
      const started = await start(outputDir);
      if ("jobId" in started) setPhase({ kind: "running", jobId: started.jobId });
      else finish(started.result);
    } catch (err) {
      fail(err instanceof Error ? err.message : "Export failed");
    }
  };

  const ready =
    canStart && !nothingSelected && (!needsFolder || outputDir.length > 0);
  const openFolder =
    phase.kind === "done" && phase.view.folder && window.electronAPI?.openPath
      ? phase.view.folder
      : null;

  return (
    <Dialog open onOpenChange={(open) => !open && !running && onClose()}>
      <DialogContent
        className={`max-h-[90vh] overflow-y-auto ${wide ? "max-w-7xl" : "max-w-2xl"}`}
        nonDismissable={running}
      >
        {phase.kind === "done" ? (
          <>
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2">
                <CheckCircle2 className="h-5 w-5 text-primary-ink" />
                {phase.view.title}
              </DialogTitle>
            </DialogHeader>
            {phase.view.folder && (
              <div className="rounded-md border bg-muted/30 p-3 text-xs">
                <p className="mb-1 text-muted-foreground">Saved in</p>
                <code
                  className="block truncate text-left font-mono [direction:rtl]"
                  title={phase.view.folder}
                >
                  <bdi>{phase.view.folder}</bdi>
                </code>
              </div>
            )}
            {phase.view.details}
            {openFolder && (
              <NextStepRow
                icon={FolderOpen}
                title="Open folder"
                description="Show what was written in your file manager."
                onClick={() => void window.electronAPI?.openPath(openFolder)}
              />
            )}
            <DialogFooter>
              <Button type="button" onClick={onClose}>
                Close
              </Button>
            </DialogFooter>
          </>
        ) : (
          <>
            <DialogHeader>
              <DialogTitle>{title}</DialogTitle>
              <DialogDescription>{description}</DialogDescription>
            </DialogHeader>

            {running && (
              <JobMetricsProgress
                message={progress.message || "Writing..."}
                metrics={progress.metrics}
                phaseProgress={progress.progress}
              />
            )}
            {/* Hidden, not unmounted, while running: the deployment picker
                keeps its own selection, and an error or a cancel must hand
                back the form exactly as the user left it. */}
            <div className={running ? "hidden" : "space-y-5"}>
              {showPicker && onScopeChange && (
                <ExportSection title="Deployments">
                  <ExportScopeSelect
                    projectId={projectId}
                    onChange={onScopeChange}
                    className="w-full min-w-0"
                  />
                </ExportSection>
              )}
              {typeof children === "function" ? children(outputDir) : children}
              {needsFolder && (
                <div className={wide ? WIDE_COLUMNS : undefined}>
                  <ExportSection title="Save in" caption={folderCaption}>
                    <FolderSelector
                      value={outputDir || null}
                      onChange={(dir) => {
                        // The last error was about the last try; a new folder starts over.
                        setOutputDir(dir);
                        setPhase({ kind: "form", error: null });
                      }}
                      hideLabel
                      hideScanResult
                      noScan
                      pickerTitle="Save in"
                    />
                  </ExportSection>
                </div>
              )}
              {phase.kind === "form" && phase.error && (
                <Callout variant="error" size="compact">
                  {phase.error}
                </Callout>
              )}
            </div>

            <DialogFooter>
              {running ? (
                cancellable &&
                jobId && (
                  <Button
                    type="button"
                    variant="outline"
                    disabled={cancelling}
                    onClick={() => {
                      setCancelling(true);
                      progress.cancel();
                    }}
                  >
                    {cancelling ? "Cancelling..." : "Cancel"}
                  </Button>
                )
              ) : (
                <>
                  <Button type="button" variant="outline" onClick={onClose}>
                    Cancel
                  </Button>
                  <Button type="button" onClick={() => void onStart()} disabled={!ready}>
                    Start
                  </Button>
                </>
              )}
            </DialogFooter>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}
