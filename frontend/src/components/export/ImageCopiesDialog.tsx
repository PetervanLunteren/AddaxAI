/**
 * Image copies: the folder-run media card and its tree preview, for a
 * project, in one dialog on the Export page.
 *
 * Nothing here is the dialog's own except the output folder and the
 * Start button: the options, the preview and the persistence come from
 * `useMediaOptions`, the hook the folder-run Save step composes too, so
 * a user learns one form. Starting runs the same save-outputs job a
 * folder run uses; progress and the result screen reuse its pieces.
 *
 * The form is mounted only while the dialog is open, so every opening
 * starts from the remembered layout and look with the snapshot filters
 * (deployments, labels, dates, liked, verified) cleared. The job state
 * lives outside it and survives the dialog closing while a job runs.
 */

import { useState } from "react";
import { AlertTriangle, CheckCircle2, FolderOpen } from "lucide-react";

import { exportApi } from "../../api/export";
import type { SaveOutputsResult } from "../../api/folder-runs";
import { collectIssues } from "../../pages/folder-run/save/media-options";
import { OutputPreviewPanel } from "../../pages/folder-run/save/OutputPreviewPanel";
import {
  IssuesPanel,
  MediaBody,
  OutputFolderField,
} from "../../pages/folder-run/save/SaveShared";
import { useMediaOptions } from "../../pages/folder-run/save/useMediaOptions";
import { JobProgressModal } from "../folder-run/JobProgressModal";
import { SaveOutputsProgress } from "../folder-run/SaveOutputsProgress";
import { useSaveOutputsProgress } from "../folder-run/useSaveOutputsProgress";
import { Button } from "../ui/button";
import { Callout } from "../ui/callout";
import { Card, CardContent } from "../ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";
import { NextStepRow } from "../ui/next-step-row";

interface ImageCopiesDialogProps {
  projectId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function ImageCopiesDialog({
  projectId,
  open,
  onOpenChange,
}: ImageCopiesDialogProps) {
  const [jobId, setJobId] = useState<string | null>(null);
  const [result, setResult] = useState<SaveOutputsResult | null>(null);
  // An error from the job itself, shown on the next opening of the form.
  const [startError, setStartError] = useState<string | null>(null);

  const progress = useSaveOutputsProgress({
    jobId,
    onComplete: (data) => {
      setResult(data);
      setJobId(null);
    },
    onError: (msg) => {
      setResult(null);
      setJobId(null);
      setStartError(msg);
    },
    onCancelled: () => setJobId(null),
  });

  const running = jobId !== null;

  const close = (next: boolean) => {
    if (running) return;
    if (!next) {
      setResult(null);
      setStartError(null);
    }
    onOpenChange(next);
  };

  const openResults = async () => {
    if (!result || !window.electronAPI?.openPath) return;
    await window.electronAPI.openPath(result.output_dir);
  };

  // What landed on disk: separation always runs for this export and
  // counts one placement per copied file, with the label filter and the
  // empties toggle already applied. `source_file_count` is the scope
  // before those, so it overstates.
  const copied = result?.separate_folders?.written_count ?? 0;

  return (
    <>
      <Dialog open={open && !running} onOpenChange={close}>
        {/* Same width as the folder-run step content (max-w-7xl in
            FolderRunLayout) and the same two columns below, so the dialog
            is the Save step the user already knows, in a frame. */}
        <DialogContent className="max-h-[90vh] max-w-7xl overflow-y-auto">
          {result ? (
            <>
              <DialogHeader>
                <DialogTitle className="flex items-center gap-2">
                  {copied > 0 ? (
                    <CheckCircle2 className="h-5 w-5 text-primary-ink" />
                  ) : (
                    <AlertTriangle className="h-5 w-5 text-muted-foreground" />
                  )}
                  {copied > 0 ? "Copies written" : "Nothing copied"}
                </DialogTitle>
                <DialogDescription>
                  {copied.toLocaleString()} {copied === 1 ? "file" : "files"}{" "}
                  copied into addaxai-media.
                </DialogDescription>
              </DialogHeader>
              <div className="rounded-md border bg-muted/30 p-3 text-xs">
                <p className="mb-1 text-muted-foreground">Saved to</p>
                <code
                  className="block truncate text-left font-mono [direction:rtl]"
                  title={result.output_dir}
                >
                  <bdi>{result.output_dir}</bdi>
                </code>
              </div>
              {collectIssues(result).length > 0 && (
                <IssuesPanel issues={collectIssues(result)} />
              )}
              {window.electronAPI?.openPath && (
                <NextStepRow
                  icon={FolderOpen}
                  title="Open output folder"
                  description="Show the copies AddaxAI wrote."
                  onClick={() => void openResults()}
                />
              )}
              <DialogFooter>
                <Button type="button" onClick={() => close(false)}>
                  Close
                </Button>
              </DialogFooter>
            </>
          ) : (
            open && (
              <ImageCopiesForm
                projectId={projectId}
                initialError={startError}
                onStarted={(id) => {
                  setStartError(null);
                  setJobId(id);
                }}
                onCancel={() => close(false)}
              />
            )
          )}
        </DialogContent>
      </Dialog>

      <JobProgressModal
        open={running}
        title="Copying images"
        isCancelling={progress.isCancelling}
        onCancel={progress.cancel}
      >
        <SaveOutputsProgress
          modules={progress.modules}
          currentModule={progress.currentModule}
          moduleIndex={progress.moduleIndex}
          totalModules={progress.totalModules}
          message={progress.message}
          phaseProgress={progress.phaseProgress}
        />
      </JobProgressModal>
    </>
  );
}

/** The options and the Start button. Mounted per opening: its state is
 * one export's worth of choices. */
function ImageCopiesForm({
  projectId,
  initialError,
  onStarted,
  onCancel,
}: {
  projectId: string;
  initialError: string | null;
  onStarted: (jobId: string) => void;
  onCancel: () => void;
}) {
  const [outputDir, setOutputDir] = useState("");
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(initialError);
  const media = useMediaOptions({ projectId, alwaysEnabled: true });

  const canStart = !!outputDir && media.hasFilesToCopy && !starting;

  const start = async () => {
    setError(null);
    setStarting(true);
    media.remember();
    try {
      const { job_id } = await exportApi.prepareImageCopies(projectId, {
        ...media.requestFields,
        output_dir: outputDir,
      });
      onStarted(job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Export failed");
    } finally {
      setStarting(false);
    }
  };

  // Nothing in scope can only mean the deployment picker, which shows
  // with more than one deployment; an empty label pick is "all".
  const emptyMessage = "Pick at least one deployment to see the preview";

  return (
    <>
      <DialogHeader>
        <DialogTitle>Image copies</DialogTitle>
        <DialogDescription>
          Copies of your images in species folders, with your labels in the
          image metadata. Originals are never touched.
        </DialogDescription>
      </DialogHeader>

      {/* Same two columns as the folder-run Save step: options left, the
          tree they produce right. */}
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_360px] lg:items-start">
        <div className="space-y-6">
          <OutputFolderField
            value={outputDir}
            onChange={setOutputDir}
            caption="Copies go into an addaxai-media folder inside the folder you pick. Exporting again replaces it. Your originals are never touched."
          />
          <Card>
            <CardContent className="p-6">
              <MediaBody form={media} />
            </CardContent>
          </Card>
          {error && (
            <Callout variant="error" size="compact">
              {error}
            </Callout>
          )}
        </div>

        <OutputPreviewPanel
          media={media}
          outputDir={outputDir}
          fallbackName="output-folder"
          preview={media.preview}
          isLoading={media.previewLoading}
          emptyMessage={emptyMessage}
        />
      </div>

      <DialogFooter>
        <Button type="button" variant="outline" onClick={onCancel}>
          Cancel
        </Button>
        <Button type="button" onClick={() => void start()} disabled={!canStart}>
          {starting ? "Starting..." : "Start"}
        </Button>
      </DialogFooter>
    </>
  );
}
