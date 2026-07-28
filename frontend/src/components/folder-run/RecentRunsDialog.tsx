/**
 * "Show recent runs" — the step-1 re-entry list.
 *
 * A folder run persists its step, but the only way back into one used to be
 * remembering its folder and re-browsing to it. This lists recent runs so the
 * user can click straight back in.
 *
 * Clicking a row navigates to `/folder-runs/{id}`; the existing
 * FolderRunResumeIndex redirect drops the user on that run's persisted step —
 * no resume logic lives here.
 *
 * Runs whose folder has moved, been deleted, or sits on an unplugged drive
 * come back with `folder_exists: false`. Those are shown greyed and are not
 * resumable (there'd be nothing to open), but they can still be deleted —
 * which is exactly when you'd want to.
 *
 * Delete is irreversible: it takes the run's DB rows and its on-disk
 * `.addaxai` cache, so it sits behind a confirm naming the folder.
 *
 * The API returns every run, so the list starts at INITIAL_ROWS and the rest
 * are one click away. Nothing is hidden silently: a truncated list reads as
 * "these are your runs", and a user whose run is missing would reasonably
 * conclude it is gone and re-run work they already did. Since the full list is
 * already in the browser, revealing it in pages would buy nothing but clicks.
 */

import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { FolderOpen, Trash2 } from "lucide-react";

import { folderRunsApi, type FolderRunSummary } from "../../api/folder-runs";
import { formatAuditWhen } from "../../lib/auditTime";
import { cn } from "../../lib/utils";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "../ui/alert-dialog";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";

/**
 * The row's sub-line: what a run is, at a glance.
 *
 * `file_count` counts File rows, which are only written once results load
 * (`ml/json_pipeline.py`), so it is really "how much has been analysed", not
 * "how big is the folder". That makes it the signal for whether the run ran
 * at all, and it is why an unanalysed run reports no count: "0 files" would
 * describe the database, not the folder the user picked, which is full of
 * images. The folder's own size isn't known here, so the row stays quiet
 * about it rather than guessing.
 */
function formatRunFacts(run: FolderRunSummary, when: string): string {
  // Date first: it is the one fact every row has, so the column reads down
  // cleanly whether or not a run has been analysed.
  if (run.file_count === 0) return `${when} · not analysed yet`;
  const files = `${run.file_count.toLocaleString()} files`;
  return `${when} · ${files} · ${formatReview(run)}`;
}

/**
 * How far the review got. Only called once results exist, so "no detections"
 * unambiguously means analysed-and-empty rather than never-run: those are
 * different facts, and calling an empty result "not analysed" would invite
 * the user to re-run work they already did.
 *
 * "Labels verified" is the app's canonical review metric (verified detections
 * over the detections the run shows), the same fraction the dashboard's
 * VerificationProgressChart and RerunConfirmDialog report. The wording is
 * shorter here because it shares a line with the file count and the date; the
 * metric itself is defined by the backend fields, so the two cannot drift.
 */
function formatReview(run: FolderRunSummary): string {
  const total = run.detection_count;
  if (total === 0) return "no detections";
  const verified = run.verified_detection_count;
  if (verified === total) return "all labels verified";
  return `${Math.round((verified / total) * 100)}% labels verified`;
}

/** Rows shown before "Show all". The rest are one click away, not gone. */
const INITIAL_ROWS = 20;

interface RecentRunsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function RecentRunsDialog({ open, onOpenChange }: RecentRunsDialogProps) {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [pendingDelete, setPendingDelete] = useState<FolderRunSummary | null>(
    null,
  );
  const [showAll, setShowAll] = useState(false);

  const { data: runs = [], isLoading } = useQuery({
    queryKey: ["folder-runs"],
    queryFn: folderRunsApi.list,
    enabled: open,
  });

  // Reopening the dialog starts from the top again, so a one-off dig through
  // old runs doesn't leave a long list for every visit afterwards.
  useEffect(() => {
    if (!open) setShowAll(false);
  }, [open]);

  const visibleRuns = showAll ? runs : runs.slice(0, INITIAL_ROWS);
  const hiddenCount = runs.length - visibleRuns.length;

  const remove = useMutation({
    mutationFn: (runId: string) => folderRunsApi.remove(runId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["folder-runs"] });
      setPendingDelete(null);
    },
  });

  return (
    <>
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Recent runs</DialogTitle>
            <DialogDescription>
              Pick up where you left off. Opening a run takes you back to the
              step it was left on.
            </DialogDescription>
          </DialogHeader>

          <div className="max-h-96 space-y-1.5 overflow-y-auto rounded-lg border bg-muted/30 p-2">
            {isLoading ? (
              <p className="py-6 text-center text-sm text-muted-foreground">
                Loading recent runs…
              </p>
            ) : runs.length === 0 ? (
              <p className="py-6 text-center text-sm text-muted-foreground">
                No previous runs yet.
              </p>
            ) : (
              visibleRuns.map((run) => {
                const when = formatAuditWhen(run.updated_at_utc);
                return (
                  <div
                    key={run.id}
                    className={cn(
                      "flex items-center gap-3 rounded-lg border bg-white p-2.5",
                      run.folder_exists
                        ? "hover:border-[#0f6064]/40"
                        : "opacity-60",
                    )}
                  >
                    <button
                      type="button"
                      disabled={!run.folder_exists}
                      onClick={() => {
                        onOpenChange(false);
                        navigate(`/folder-runs/${run.id}`);
                      }}
                      className="flex min-w-0 flex-1 items-center gap-3 text-left disabled:cursor-not-allowed"
                    >
                      <FolderOpen className="h-4 w-4 shrink-0 text-muted-foreground" />
                      <span className="min-w-0 flex-1">
                        {/* Truncate from the START: the tail (the folder you
                            recognise) is the useful part of a path. `direction:
                            rtl` moves the ellipsis to the left; the LRM prefix
                            keeps the leading "/" from being reordered to the
                            far end by the bidi algorithm. */}
                        <span
                          dir="rtl"
                          className="block truncate text-left text-sm font-medium"
                        >
                          {"‎" + run.folder_path}
                        </span>
                        <span className="block truncate text-xs text-muted-foreground">
                          {run.folder_exists
                            ? formatRunFacts(run, when.rel)
                            : "Folder not found: it moved, was deleted, or its drive isn't connected"}
                        </span>
                      </span>
                    </button>
                    <button
                      type="button"
                      title="Delete this run"
                      aria-label={`Delete the run for ${run.folder_path}`}
                      onClick={() => setPendingDelete(run)}
                      className="shrink-0 rounded p-1.5 text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                );
              })
            )}
          </div>

          {/* Only when something is actually hidden: with nothing left to
              reveal, a count would just be noise. */}
          {hiddenCount > 0 && (
            <div className="flex items-center justify-between px-1 text-xs text-muted-foreground">
              <span>
                Showing {visibleRuns.length} of {runs.length} runs
              </span>
              <button
                type="button"
                onClick={() => setShowAll(true)}
                className="font-medium text-primary hover:underline"
              >
                Show all
              </button>
            </div>
          )}
        </DialogContent>
      </Dialog>

      <AlertDialog
        open={pendingDelete !== null}
        onOpenChange={(o) => !o && setPendingDelete(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete this run?</AlertDialogTitle>
            <AlertDialogDescription>
              Deletes the analysis of{" "}
              <span className="font-medium text-foreground">
                {pendingDelete?.folder_path}
              </span>
              , including any labels you verified. Your images and videos are
              not touched. This cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={remove.isPending}>
              Cancel
            </AlertDialogCancel>
            <AlertDialogAction
              disabled={remove.isPending}
              onClick={(e) => {
                // Keep the dialog up until the delete resolves.
                e.preventDefault();
                if (pendingDelete) remove.mutate(pendingDelete.id);
              }}
            >
              {remove.isPending ? "Deleting…" : "Delete run"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
