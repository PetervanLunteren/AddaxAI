/**
 * Slim status row for the Setup step, shown when the folder run is
 * already in a terminal state (queue entry completed or failed) and
 * the user is still looking at its source folder.
 *
 * Sits where the standalone "Start analysis" button normally lives,
 * so the action area collapses to one row regardless of run state:
 *
 *   [ icon + status text ]                       [ Skip ] [ Re-run ]
 *
 * Two actions:
 * - **Skip analysis**: jump straight to Verification. Always the
 *   same destination so the mental model stays "skip or run again".
 * - **Re-run analysis**: open the destructive confirm dialog that
 *   wipes the analysis output and re-processes under the current
 *   form values.
 *
 * Metadata (file counts, verification progress, run date) lives on
 * the Verification / Summary / Output pages and inside the Re-run
 * destructive confirm dialog. We keep this row light because the
 * user is already inside the run and the destructive button has its
 * own confirm with the full breakdown.
 */

import {
  AlertCircle,
  ArrowRight,
  CheckCircle2,
  RotateCcw,
} from "lucide-react";
import { Button } from "../ui/button";

interface CompletedRunNoticeProps {
  /** True when the queue entry's status is "failed"; flips the icon
   * + status text. Otherwise the row reads as a completed run. */
  failed: boolean;
  isBusy: boolean;
  /** False when a selected model needs setup. Disables Re-run (which
   * would otherwise spawn a job that can't run); Skip stays enabled
   * because viewing results doesn't touch the models. */
  canRerun: boolean;
  onSkipAnalysis: () => void;
  onRerun: () => void;
}

export function CompletedRunNotice({
  failed,
  isBusy,
  canRerun,
  onSkipAnalysis,
  onRerun,
}: CompletedRunNoticeProps) {
  const Icon = failed ? AlertCircle : CheckCircle2;
  const iconClass = failed ? "text-destructive" : "text-primary";
  const message = failed
    ? "Previous run did not finish"
    : "This folder is already analysed";

  return (
    <div className="flex items-center justify-between gap-3">
      <div className="flex items-center gap-2 text-sm">
        <Icon className={`h-4 w-4 shrink-0 ${iconClass}`} />
        <span className="font-medium">{message}</span>
      </div>
      <div className="flex items-center gap-2">
        {/* type="button" is essential: this row lives inside the Setup
            <form>, so a default-type button would also submit the form
            (firing the create-or-resume Start path) on top of its own
            handler. */}
        <Button
          type="button"
          variant="ghost"
          onClick={onRerun}
          disabled={isBusy || !canRerun}
          size="lg"
          className="gap-2"
        >
          <RotateCcw className="h-4 w-4" />
          Re-run analysis
        </Button>
        {/* Skip is only meaningful when there is a finished analysis to
            view. A failed / interrupted run has no complete results, so
            skipping leads to an empty Edit page — drop the option and
            steer the user to Re-run. */}
        {!failed && (
          <Button
            type="button"
            onClick={onSkipAnalysis}
            disabled={isBusy}
            size="lg"
            className="gap-2"
          >
            Skip analysis
            <ArrowRight className="h-4 w-4" />
          </Button>
        )}
      </div>
    </div>
  );
}
