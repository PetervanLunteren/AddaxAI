/**
 * Destructive confirm for "Re-run analysis on this folder".
 *
 * The folder run already finished once and the user clicked Re-run on
 * the Setup page. The dialog shows what the existing run holds (date,
 * models, counts, verification progress) so the user can see exactly
 * what they're about to discard, then spells out the destructive
 * effect with the verified count called out specifically.
 *
 * A failed run (killed by a crash or a power cut) gets a different
 * dialog: there is nothing to weigh, because nothing of it reached the
 * database. The summary box and the backups Callout are dropped for it.
 * "Analysed 21 Aug · 0 files" is nonsense for a run that never finished,
 * and the backups line sent a user into a loop of restores looking for
 * results that were never saved anywhere.
 *
 * What can survive is detection progress: MegaDetector saves a checkpoint
 * every few hundred images, and a finished detection file counts as one
 * too. When the lookup reports that (`detection_resume`) and the form
 * still holds the detection settings it was made under (`canContinue`),
 * the failed dialog offers Continue next to Start over. The steps after
 * detection always run again; only detection itself is picked up.
 */

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
import type { DetectionResume, FolderRunLookup } from "../../api/folder-runs";

interface RerunConfirmDialogProps {
  open: boolean;
  /** The run being re-run (the current run, or a previous run matched
   * on this folder). Null while the lookup is still resolving. */
  run: FolderRunLookup | null;
  /** True when the run's queue entry is "failed": the previous run was
   * interrupted and has no results to discard. */
  failed: boolean;
  /** True when the failed run left a detection checkpoint that is valid
   * for the detection settings currently in the form. Adds Continue. */
  canContinue: boolean;
  isBusy: boolean;
  onCancel: () => void;
  /** `keepCheckpoint` is true for Continue, false for Start over and for
   * a plain re-run. */
  onConfirm: (keepCheckpoint: boolean) => void;
}

export function RerunConfirmDialog({
  open,
  run,
  failed,
  canContinue,
  isBusy,
  onCancel,
  onConfirm,
}: RerunConfirmDialogProps) {
  const hasProgress =
    !failed &&
    ((run?.verified_detection_count ?? 0) > 0 ||
      (run?.confirmed_event_count ?? 0) > 0);
  const resume = failed && canContinue ? run?.detection_resume ?? null : null;

  return (
    // The dialog stays up while the wipe runs (see `confirmRerun`), so it
    // must not be dismissable in that window: cancelling would hide the only
    // progress signal without stopping anything.
    <Dialog
      open={open}
      onOpenChange={(v) => {
        if (!v && !isBusy) onCancel();
      }}
    >
      <DialogContent nonDismissable={isBusy}>
        <DialogHeader>
          <DialogTitle>
            {resume
              ? "Continue the analysis?"
              : failed
                ? "Run the analysis again?"
                : "Re-run analysis?"}
          </DialogTitle>
          <DialogDescription>
            {resume
              ? formatResume(resume)
              : failed
                ? "The previous run was interrupted. Its results cannot be recovered, so the analysis starts from the beginning."
                : "Re-running deletes the existing analysis output and starts fresh."}
          </DialogDescription>
        </DialogHeader>

        {/* Neutral summary of what the run holds, so the user can weigh
            the cost. Not a Callout: Callout is for advisories, not a
            status readout (see its docstring). */}
        {run && !failed && (
          <div className="rounded-md border bg-muted/30 p-3 text-xs text-muted-foreground">
            <p>{formatRunSummary(run)}</p>
            <p className="mt-1">{formatRunCounts(run)}</p>
            {formatLabelsVerified(run) && (
              <p className="mt-1">{formatLabelsVerified(run)}</p>
            )}
            {formatCountsConfirmed(run) && (
              <p className="mt-1">{formatCountsConfirmed(run)}</p>
            )}
          </div>
        )}

        {/* Consequence + reassurance follow the app's destructive-confirm
            shape (see DeleteSiteDialog): a warning Callout for what's
            lost, an info Callout for the safety net. Both only when there
            is progress to lose: with nothing to go back to, pointing at
            the backups folder only invites a restore hunt. */}
        {hasProgress && (
          <>
            <Callout variant="warning">
              Your verification and count progress will be lost.
            </Callout>
            <Callout variant="info">
              A database snapshot from earlier today is in your backups
              folder if you change your mind.
            </Callout>
          </>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={onCancel} disabled={isBusy}>
            Cancel
          </Button>
          {resume && (
            <Button
              variant="outline"
              onClick={() => onConfirm(false)}
              disabled={isBusy}
            >
              Start over
            </Button>
          )}
          <Button onClick={() => onConfirm(!!resume)} disabled={isBusy}>
            {isBusy
              ? "Clearing previous results…"
              : resume
                ? "Continue"
                : failed
                  ? "Run analysis"
                  : "Re-run analysis"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function formatResume(resume: DetectionResume): string {
  const total = resume.images_total.toLocaleString();
  if (resume.images_done >= resume.images_total) {
    return `Detection finished for all ${total} images before the run was interrupted. Continue skips detection. The steps after detection run again.`;
  }
  return `Detection got to ${resume.images_done.toLocaleString()} of ${total} images before the run was interrupted. Continue picks up there. The steps after detection run again.`;
}

function formatRunSummary(run: FolderRunLookup): string {
  const date = formatDate(run.updated_at_utc);
  const models = [
    run.detection_model_name ?? run.detection_model_id,
    run.classification_model_name ?? run.classification_model_id,
  ]
    .filter(Boolean)
    .join(" + ");
  return models ? `Analysed ${date} · ${models}` : `Analysed ${date}`;
}

function formatRunCounts(run: FolderRunLookup): string {
  const parts: string[] = [
    `${run.file_count.toLocaleString()} file${
      run.file_count === 1 ? "" : "s"
    }`,
  ];
  if (run.detection_count > 0) {
    parts.push(
      `${run.detection_count.toLocaleString()} observation${
        run.detection_count === 1 ? "" : "s"
      }`,
    );
  }
  if (run.species_count > 0) {
    parts.push(`${run.species_count.toLocaleString()} species`);
  }
  return parts.join(" · ");
}

// The app frames verification as two metrics (see the dashboard's
// VerificationProgressChart): "Labels verified" (detections checked on the
// Labels page) and "Counts confirmed" (events confirmed on the Counts page).
// Re-running discards both, so the dialog shows both.

function formatLabelsVerified(run: FolderRunLookup): string | null {
  const total = run.detection_count;
  if (total === 0) return null;
  const verified = run.verified_detection_count;
  if (verified === total) {
    return `All ${total.toLocaleString()} labels verified`;
  }
  const pct = Math.round((verified / total) * 100);
  return `${verified.toLocaleString()} of ${total.toLocaleString()} labels verified (${pct}%)`;
}

function formatCountsConfirmed(run: FolderRunLookup): string | null {
  const total = run.event_count;
  if (total === 0) return null;
  const confirmed = run.confirmed_event_count;
  if (confirmed === total) {
    return `All ${total.toLocaleString()} counts confirmed`;
  }
  const pct = Math.round((confirmed / total) * 100);
  return `${confirmed.toLocaleString()} of ${total.toLocaleString()} counts confirmed (${pct}%)`;
}

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      day: "numeric",
      month: "short",
      year: "numeric",
    });
  } catch {
    return iso;
  }
}
