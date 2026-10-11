/**
 * Confirm, write, report: the recognition-file export in one dialog.
 *
 * Unlike the other exports this one writes into the user's own folders
 * (one addaxai-recognitions.json per deployment folder, where Timelapse
 * looks for it) and replaces a file that is already there, so it asks
 * first. The backend answers per deployment, written or skipped with a
 * reason, and that list is the result the user needs to see: a
 * deployment on an unplugged drive is skipped, not an error.
 */

import { useState } from "react";
import { Loader2 } from "lucide-react";

import {
  exportApi,
  type ExportScope,
  type RecognitionJsonExportResult,
} from "../../api/export";
import { useTaskProgress } from "../../hooks/useTaskProgress";
import { basename } from "../../lib/path-utils";
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
import { Progress } from "../ui/progress";

interface RecognitionJsonExportDialogProps {
  projectId: string;
  scope: ExportScope | undefined;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function RecognitionJsonExportDialog({
  projectId,
  scope,
  open,
  onOpenChange,
}: RecognitionJsonExportDialogProps) {
  const [jobId, setJobId] = useState<string | null>(null);
  const [result, setResult] = useState<RecognitionJsonExportResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const writing = jobId !== null;

  // Per-deployment progress from the worker; the written/skipped list
  // rides on the completion message, so no second request is needed.
  const { metrics } = useTaskProgress({
    taskId: jobId,
    onComplete: (data) => {
      setResult((data as unknown as RecognitionJsonExportResult) ?? null);
      setJobId(null);
    },
    onError: (msg) => {
      setError(msg);
      setJobId(null);
    },
  });
  const current = metrics?.current ?? 0;
  const total = metrics?.total ?? 0;
  const pct = total > 0 ? Math.round((current / total) * 100) : 0;

  const close = (next: boolean) => {
    if (writing) return;
    if (!next) {
      setResult(null);
      setError(null);
    }
    onOpenChange(next);
  };

  const run = async () => {
    setError(null);
    try {
      const { job_id } = await exportApi.prepareRecognitionJson(projectId, scope);
      setJobId(job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Export failed");
    }
  };

  const folderWord = (n: number) => (n === 1 ? "folder" : "folders");

  return (
    <Dialog open={open} onOpenChange={close}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>
            {result ? "Recognition files written" : "Write recognition files"}
          </DialogTitle>
          <DialogDescription>
            {result
              ? "One file per deployment folder, ready for Timelapse."
              : "The file Timelapse reads, with your labels and corrections."}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3 text-sm">
          {!result && !writing && (
            <p>
              AddaxAI writes <code>addaxai-recognitions.json</code> into the
              folder of every deployment in your selection, with paths
              relative to that folder. A file already there is replaced.
              Open Timelapse on a deployment folder to import it.
            </p>
          )}

          {writing && (
            <div className="space-y-2">
              <div className="flex items-center gap-3">
                <Loader2 className="h-5 w-5 shrink-0 animate-spin text-primary-ink" />
                <span className="font-medium">
                  {total > 0
                    ? `Deployment ${current.toLocaleString()} of ${total.toLocaleString()}`
                    : "Starting..."}
                </span>
                {total > 0 && (
                  <span className="ml-auto font-mono text-xs text-muted-foreground">
                    {pct}%
                  </span>
                )}
              </div>
              {total > 0 && <Progress value={pct} className="h-2" />}
            </div>
          )}

          {result && (
            <>
              <p>
                Written into {result.written.length}{" "}
                {folderWord(result.written.length)}.
              </p>
              {result.skipped.length > 0 && (
                <Callout variant="warning" size="compact">
                  <p>
                    Skipped {result.skipped.length}{" "}
                    {folderWord(result.skipped.length)}:
                  </p>
                  <ul className="mt-1 list-disc pl-5">
                    {result.skipped.map((s) => (
                      <li key={s.deployment_id}>
                        {s.folder_path ? basename(s.folder_path) : "(no folder)"}
                        : {s.reason}
                      </li>
                    ))}
                  </ul>
                </Callout>
              )}
            </>
          )}

          {error && (
            <Callout variant="error" size="compact">
              {error}
            </Callout>
          )}
        </div>

        <DialogFooter>
          {result ? (
            <Button type="button" onClick={() => close(false)}>
              Close
            </Button>
          ) : (
            <>
              <Button
                type="button"
                variant="outline"
                onClick={() => close(false)}
                disabled={writing}
              >
                Cancel
              </Button>
              <Button type="button" onClick={() => void run()} disabled={writing}>
                {writing ? "Writing..." : "Write files"}
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
